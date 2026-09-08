"""Train one configuration, one seed.

    python -m src.train --config configs/model/d1.yaml --seed 42

Writes everything a result needs to be reproduced and audited:
``config.yaml``, ``env.json``, ``curve.csv``, ``metrics.json``,
``predictions.npz`` and (on request) ``diagnostics.npz``.

Two protocol rules are enforced here rather than left to discipline.

**The test split is touched once**, at the end, after early stopping has already
chosen the checkpoint on validation. Nothing in the training loop reads it.

**Predictions are saved, not just scores.** Every model runs on an identical
origin set derived from the frozen split, so any two saved runs are row-aligned
and the paired bootstrap is valid without further work. On this dataset that is
not a nicety: one test year holds 53 independent weeks, absolute MAE carries a
+/-12.6 % interval, and models cannot be ranked by headline number.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data import contract
from src.data.windows import Batch, WindowBatcher, build_batchers
from src.metrics import Predictions, format_table, summary_table
from src.models.head import build_loss
from src.models.sdgt import ModelConfig, build_model
from src.utils.manifest import capture_environment, write_json
from src.utils.seeding import seed_everything

RUNS_ROOT = Path("experiments/runs")
REPORT_HORIZONS = (1, 6, 12, 24)


# --------------------------------------------------------------------- config
def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config, merging it over ``configs/base.yaml``."""
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    base_path = Path("configs/base.yaml")
    if base_path.exists() and path.resolve() != base_path.resolve():
        with open(base_path, encoding="utf-8") as handle:
            base = yaml.safe_load(handle) or {}
        config = _deep_merge(base, config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def model_config_from(config: dict, dataset) -> ModelConfig:
    data, model = config["data"], config["model"]
    graph = model.get("graph", {})
    return ModelConfig(
        n_features=dataset.n_features,
        n_stations=dataset.n_stations,
        coords=dataset.coords,
        lookback=data["lookback"],
        horizon=data["horizon"],
        patch_length=data["patch_length"],
        patch_stride=data["patch_stride"],
        d_model=model["d_model"],
        n_heads=model["n_heads"],
        dropout=model["dropout"],
        temporal=model["temporal"]["type"],
        temporal_layers=model["temporal"].get("n_layers", 3),
        ffn_mult=model["temporal"].get("ffn_mult", 4),
        graph=graph.get("type", "none"),
        spatial=model["spatial"]["type"],
        spatial_layers=model["spatial"].get("n_layers", 2),
        edge_dropout=model["spatial"].get("edge_dropout", 0.1),
        fusion=model["fusion"]["type"],
        head_hidden=model["head"].get("hidden", 256),
        graph_options={k: v for k, v in graph.items() if k != "type"},
    )


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batcher: WindowBatcher,
    loss_fn,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (loss, predictions, truth, mask, origins) in *scaled* units."""
    model.eval()
    total, count = 0.0, 0
    predictions, truths, masks, origins = [], [], [], []

    for batch in batcher:
        batch = batch.to(device)
        output, _ = model(batch.features, batch.wind_uv)
        loss = loss_fn(output, batch.target, batch.target_mask.float())
        total += float(loss) * len(batch)
        count += len(batch)

        predictions.append(output.cpu().numpy())
        truths.append(batch.target.cpu().numpy())
        masks.append(batch.target_mask.cpu().numpy())
        origins.append(batch.origins.cpu().numpy())

    return (
        total / max(count, 1),
        np.concatenate(predictions),
        np.concatenate(truths),
        np.concatenate(masks),
        np.concatenate(origins),
    )


def to_predictions(
    dataset, scaled_pred, scaled_truth, mask, origins, model_name: str, seed: int
) -> Predictions:
    """Invert scaling so every reported metric is in ug/m3."""
    return Predictions(
        origins=origins,
        pred=dataset.target_scaler.inverse(scaled_pred).astype(np.float32),
        truth=dataset.target_scaler.inverse(scaled_truth).astype(np.float32),
        mask=mask.astype(bool),
        model=model_name,
        seed=seed,
    )


def validation_score(predictions: Predictions) -> float:
    """Mean MAE across the reported horizons -- the early-stopping criterion.

    Averaging across horizons rather than optimising one keeps the model honest
    about the whole forecast, not just the easy first hour.
    """
    from src.metrics import masked_mae

    values = [
        masked_mae(
            predictions.pred[:, :, h - 1],
            predictions.truth[:, :, h - 1],
            predictions.mask[:, :, h - 1],
        )
        for h in REPORT_HORIZONS
    ]
    return float(np.mean(values))


# -------------------------------------------------------------------- training
def train(
    config: dict,
    seed: int,
    runs_root: Path,
    device_name: str | None = None,
    smoke: int | None = None,
) -> Path:
    seed_everything(seed)

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = contract.load(config["data"]["city"])
    data, training = config["data"], config["train"]

    batchers = build_batchers(
        dataset,
        lookback=data["lookback"],
        horizon=data["horizon"],
        batch_size=training["batch_size"],
        seed=seed,
        device=device,
    )

    if smoke:
        # Wiring check only: subsample every split so the whole loop -- including
        # the per-epoch validation pass, which is the real cost -- finishes in
        # seconds. Never use for a reported result.
        for batcher in batchers.values():
            batcher.origins = batcher.origins[:smoke]

    model = build_model(model_config_from(config, dataset)).to(device)
    counts = model.parameter_counts()

    name = config["name"]
    run_dir = runs_root / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 68}\n{name}  seed={seed}  device={device}\n{'=' * 68}")
    print("  parameters: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(
        "  origins: "
        + "  ".join(f"{split}={b.n_origins}" for split, b in batchers.items())
    )

    loss_fn = build_loss(training["loss"]["name"], training["loss"].get("delta", 1.0))
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=training["optimizer"]["lr"],
        weight_decay=training["optimizer"]["weight_decay"],
    )
    epochs = training["epochs"]
    warmup = training["scheduler"].get("warmup_epochs", 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda epoch: _lr_scale(epoch, warmup, epochs)
    )

    patience = training["early_stopping"]["patience"]
    best_score, best_epoch, best_state = float("inf"), -1, None
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in batchers["train"]:
            batch = batch.to(device)
            output, _ = model(batch.features, batch.wind_uv)
            loss = loss_fn(output, batch.target, batch.target_mask.float())

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimiser.step()

            running += float(loss) * len(batch)
            seen += len(batch)
        scheduler.step()

        val_loss, *val_arrays = evaluate(model, batchers["val"], loss_fn, device)
        val_predictions = to_predictions(dataset, *val_arrays, name, seed)
        score = validation_score(val_predictions)

        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "val_loss": val_loss,
                "val_mae": score,
                "lr": scheduler.get_last_lr()[0],
                "seconds": time.time() - started,
            }
        )
        marker = ""
        if score < best_score - 1e-6:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  *"
        print(
            f"  epoch {epoch:3d}  train {history[-1]['train_loss']:.4f}  "
            f"val {val_loss:.4f}  val_mae {score:6.2f}{marker}"
        )

        if epoch - best_epoch >= patience:
            print(f"  early stop: no improvement for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - started

    # ---- the test split is read exactly once, here, after selection is frozen
    _, *test_arrays = evaluate(model, batchers["test"], loss_fn, device)
    test_predictions = to_predictions(dataset, *test_arrays, name, seed)

    rows = summary_table(test_predictions)
    print(f"\n{format_table(rows)}")

    test_predictions.save(run_dir / "predictions.npz")
    torch.save({"state_dict": model.state_dict(), "config": config}, run_dir / "checkpoint.pt")
    write_json(
        run_dir / "metrics.json",
        {
            "rows": rows,
            "best_epoch": best_epoch,
            "best_val_mae": best_score,
            "epochs_run": len(history),
            "train_seconds": elapsed,
            "parameters": counts,
        },
    )
    write_json(run_dir / "env.json", capture_environment())
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with open(run_dir / "curve.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print(f"\n  best epoch {best_epoch} (val MAE {best_score:.2f}), {elapsed / 60:.1f} min")
    print(f"  wrote {run_dir}")
    return run_dir


def _lr_scale(epoch: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base rate."""
    if warmup and epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override for smoke tests")
    parser.add_argument(
        "--smoke", type=int, default=None,
        help="subsample every split to N origins; wiring checks only, never a result",
    )
    parser.add_argument("--threads", type=int, default=None, help="torch CPU thread count")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    train(config, seed, args.runs_root, args.device, args.smoke)


if __name__ == "__main__":
    main()
