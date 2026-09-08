"""Re-evaluate a trained checkpoint and dump the interpretability diagnostics.

    python -m src.evaluate --run experiments/runs/d1_wind_crossview_seed42

Training deliberately runs with diagnostics off so nothing extra is retained in
the loop. This loads the checkpoint back and does one pass with them on, writing
``diagnostics.npz`` next to the predictions.

Diagnostics are large -- the attention tensor alone is ``[n, L, N, N]``, which at
8,737 test windows is several gigabytes. Only a sampled subset of windows is
retained, chosen to span the wind range so the calm-versus-windy comparison in
``src/figures/diagnostics.py`` has both ends to draw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import contract
from src.data.windows import build_batchers
from src.metrics import Predictions, format_table, summary_table
from src.models.sdgt import build_model
from src.train import model_config_from
from src.utils.manifest import write_json
from src.utils.seeding import seed_everything

DEFAULT_SAMPLES = 256


def pick_windows(wind_speed: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """Choose windows spanning the wind range, not just a random slice.

    A uniform sample of Beijing hours is mostly calm -- 52 % sit below 1.5 m/s --
    so a plain random draw would rarely contain the strong-wind hours the graph
    figures need. This stratifies by wind speed decile.
    """
    order = np.argsort(wind_speed)
    buckets = np.array_split(order, 10)
    per_bucket = max(1, n_samples // len(buckets))

    rng = np.random.default_rng(seed)
    chosen = [
        rng.choice(bucket, size=min(per_bucket, len(bucket)), replace=False)
        for bucket in buckets
        if len(bucket)
    ]
    return np.sort(np.concatenate(chosen))


@torch.no_grad()
def run(run_dir: Path, n_samples: int = DEFAULT_SAMPLES, device_name: str | None = None) -> None:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed = config.get("seed", 42)
    seed_everything(seed)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = contract.load(config["data"]["city"])
    data = config["data"]

    model = build_model(model_config_from(config, dataset)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    batchers = build_batchers(
        dataset,
        lookback=data["lookback"],
        horizon=data["horizon"],
        batch_size=config["train"]["batch_size"],
        seed=seed,
        device=device,
    )
    test = batchers["test"]

    # Mean wind speed over each window, used both to stratify and to label.
    speed = np.hypot(dataset.wind_uv[..., 0], dataset.wind_uv[..., 1]).mean(axis=1)
    origins = test.origins.numpy()
    selected = pick_windows(speed[origins], n_samples, seed)
    keep = set(origins[selected].tolist())

    print(f"{config['name']} seed={seed} device={device}")
    print(f"  test windows {len(origins)}, retaining diagnostics for {len(keep)}")

    predictions, truths, masks, all_origins = [], [], [], []
    collected: dict[str, list[np.ndarray]] = {}
    scalars: dict[str, list[float]] = {}

    for batch in test:
        batch = batch.to(device)
        wanted = torch.tensor(
            [int(o) in keep for o in batch.origins.cpu().numpy()], device=device
        )
        output, diagnostics = model(
            batch.features, batch.wind_uv, return_diagnostics=bool(wanted.any())
        )

        predictions.append(output.cpu().numpy())
        truths.append(batch.target.cpu().numpy())
        masks.append(batch.target_mask.cpu().numpy())
        all_origins.append(batch.origins.cpu().numpy())

        if not wanted.any():
            continue
        rows = wanted.nonzero(as_tuple=True)[0]
        for key, value in diagnostics.items():
            if value.ndim == 0:
                scalars.setdefault(key, []).append(float(value))
            elif value.shape[0] == len(batch):
                collected.setdefault(key, []).append(value[rows].cpu().numpy())

        speeds = torch.hypot(batch.wind_uv[..., 0], batch.wind_uv[..., 1])
        collected.setdefault("wind_speed", []).append(speeds[rows].mean(-1).cpu().numpy())

    result = Predictions(
        origins=np.concatenate(all_origins),
        pred=dataset.target_scaler.inverse(np.concatenate(predictions)).astype(np.float32),
        truth=dataset.target_scaler.inverse(np.concatenate(truths)).astype(np.float32),
        mask=np.concatenate(masks).astype(bool),
        model=config["name"],
        seed=seed,
    )
    print("\n" + format_table(summary_table(result)))

    payload = {key: np.concatenate(chunks) for key, chunks in collected.items()}
    payload["origins"] = origins[selected]
    np.savez_compressed(run_dir / "diagnostics.npz", **payload)

    summary = {key: float(np.mean(values)) for key, values in scalars.items()}
    if summary:
        write_json(run_dir / "graph_diagnostics.json", summary)
        print("\n  graph diagnostics: " + "  ".join(f"{k}={v:.4f}" for k, v in summary.items()))

    total = sum(v.nbytes for v in payload.values()) / 1e6
    print(f"  wrote diagnostics.npz ({total:.1f} MB): {sorted(payload)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="an experiments/runs/<name> dir")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.run, args.samples, args.device)


if __name__ == "__main__":
    main()
