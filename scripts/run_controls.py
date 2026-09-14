"""Run the negative-control suite against trained checkpoints.

    python scripts/run_controls.py --runs experiments/runs/d1_wind_crossview_seed*

Each control re-scores the test split through an unmodified checkpoint with one
thing broken, and reports the damage as a paired weekly block bootstrap against
the same checkpoint's unperturbed run. Positive numbers mean the perturbation
made the forecast **worse**, which is what a model that genuinely depends on the
perturbed quantity should do.

Read the two reference rows first. ``zero_correction`` says what the learned
correction on top of the persistence anchor is worth in total, and
``history_shuffle`` says how much of that any input perturbation can actually
reach. Every other row is only interpretable as a fraction of those: a control
reporting "no effect" against a reference that also reports "no effect" has
proved nothing about the model and everything about the test having no power.

Nothing here trains. It needs ``checkpoint.pt``, which the Kaggle notebook now
brings back for every run -- earlier bundles excluded weights, which is why Run 2
could not be controlled after the fact.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running "python scripts/foo.py" puts scripts/ on sys.path, not the repo root,
# so "import src" fails unless the package happens to be pip-installed. Adding
# the root here makes every script work from a fresh clone with no install step
# -- which is what a Kaggle notebook actually does.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np
import torch

from src import bootstrap, controls as controls_module
from src.data import contract
from src.data.windows import build_batchers
from src.metrics import masked_mae
from src.models.sdgt import build_model
from src.train import model_config_from
from src.utils.manifest import write_json
from src.utils.seeding import seed_everything

HORIZONS = (1, 6, 12, 24)
N_BOOT = 600


def load_run(run_dir: Path, device: torch.device, limit: int | None = None):
    """Rebuild a trained model from its checkpoint, plus its test batcher."""
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Controls need trained weights. Result "
            "bundles from before September 2026 excluded *.pt, so a run downloaded "
            "from one of those cannot be controlled locally -- rerun the notebook, "
            "which now keeps checkpoints, or run the controls inside the session."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed = config.get("seed", 42)
    seed_everything(seed)

    dataset = contract.load(config["data"]["city"])
    model = build_model(model_config_from(config, dataset)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    batchers = build_batchers(
        dataset,
        lookback=config["data"]["lookback"],
        horizon=config["data"]["horizon"],
        batch_size=config["train"]["batch_size"],
        seed=seed,
        device=device,
    )
    test = batchers["test"]
    if limit:
        # Wiring check only: the control machinery is exercised end to end in
        # seconds instead of one full pass per control. Never a reported result.
        test.origins = test.origins[:limit]
    return model, dataset, test, config, seed


def mae_row(predictions) -> dict[int, float]:
    return {
        h: masked_mae(
            predictions.pred[:, :, h - 1],
            predictions.truth[:, :, h - 1],
            predictions.mask[:, :, h - 1],
        )
        for h in HORIZONS
    }


def run_one(run_dir: Path, device: torch.device, occlusion: bool,
            limit: int | None = None) -> dict:
    model, dataset, test, config, seed = load_run(run_dir, device, limit)
    name = config["name"]

    print(f"\n{'=' * 78}\n{name}  seed={seed}  device={device}\n{'=' * 78}")

    baseline = controls_module.evaluate_control(
        model, test, dataset, None, device=device, model_name=name, seed=seed
    )
    base_mae = mae_row(baseline)
    print("  unperturbed  " + "  ".join(f"h{h}={base_mae[h]:.2f}" for h in HORIZONS))

    # A cheap guard against loading the wrong weights or a changed data build:
    # this pass should reproduce what training wrote. Meaningless when the test
    # split has been subsampled, so it is skipped there.
    if not limit:
        _check_reproduces(run_dir, base_mae)
    else:
        print(f"  LIMIT {limit} windows -- wiring check, not a result")

    suite = controls_module.build_controls(dataset, dataset.n_stations, seed=seed)
    applicable = [c for c in suite if c.applies_to(model)]
    skipped = [c.name for c in suite if not c.applies_to(model)]
    if skipped:
        print(f"  skipped (no graph in this configuration): {', '.join(skipped)}")

    print(
        f"\n  {'control':22s}{'':2s}"
        + "".join(f"{'h=' + str(h):>18s}" for h in HORIZONS)
    )
    print(f"  {'-' * 96}")

    results = []
    for control in applicable:
        perturbed = controls_module.evaluate_control(
            model, test, dataset, control, device=device, model_name=name, seed=seed
        )
        cells, record = [], {"control": control.name, "reads": control.reads,
                             "is_reference": control.is_reference, "horizons": {}}
        for h in HORIZONS:
            # paired_interval returns metric(baseline) - metric(candidate); flip
            # it so a positive number reads as "the perturbation made it worse".
            interval = bootstrap.paired_interval(
                baseline, perturbed, horizon=h, n_boot=N_BOOT, seed=seed
            )
            damage, low, high = -interval.point, -interval.high, -interval.low
            detected = interval.significant
            cells.append(f"{damage:>+8.3f} {'*' if detected else ' '}[{low:+.2f},{high:+.2f}]")
            record["horizons"][h] = {
                "damage": damage, "ci_low": low, "ci_high": high,
                "detected": bool(detected),
                "mae_perturbed": mae_row(perturbed)[h],
                "mae_unperturbed": base_mae[h],
            }
        marker = "REF " if control.is_reference else "    "
        print(f"  {marker}{control.name:18s}" + "".join(f"{c:>18s}" for c in cells))
        results.append(record)

    print(f"\n  * = the 95 % interval excludes zero. Positive = perturbation hurt.")
    _interpret(results)

    payload = {
        "run": str(run_dir),
        "model": name,
        "seed": seed,
        "unperturbed_mae": {str(h): base_mae[h] for h in HORIZONS},
        "controls": results,
    }

    if occlusion:
        print("\n  station occlusion (damage to the *other* stations, ug/m3)")
        rows = controls_module.station_influence(model, test, dataset, device=device)
        print(f"    {'station':18s}" + "".join(f"{'h=' + str(h):>10s}" for h in HORIZONS))
        for row in sorted(rows, key=lambda r: -r["delta_h24"]):
            print(
                f"    {row['station']:18s}"
                + "".join(f"{row[f'delta_h{h}']:>10.3f}" for h in HORIZONS)
            )
        payload["station_occlusion"] = rows

    # A subsampled run must never overwrite a real one.
    name_out = "controls_limited.json" if limit else "controls.json"
    write_json(run_dir / name_out, payload)
    print(f"\n  wrote {run_dir / name_out}")
    return payload


def _check_reproduces(run_dir: Path, base_mae: dict[int, float]) -> None:
    """Warn loudly if this evaluation disagrees with the run's own metrics."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return
    import json

    rows = json.load(open(metrics_path, encoding="utf-8")).get("rows", [])
    saved = {int(r["horizon"]): float(r["mae"]) for r in rows}
    drift = {h: abs(saved[h] - base_mae[h]) for h in HORIZONS if h in saved}
    worst = max(drift.values(), default=0.0)
    if worst > 0.01:
        print(
            f"  WARNING: unperturbed MAE differs from metrics.json by up to "
            f"{worst:.3f} ug/m3. The checkpoint, the config or the data build "
            f"has changed since the run; controls computed here are not "
            f"comparable to the reported grid."
        )


def _interpret(results: list[dict], horizon: int = 24) -> None:
    """State the conclusion the numbers support, or refuse to state one.

    The percentages here are only meaningful when the reference denominator is
    itself solid. If the learned correction is worth nothing measurable, every
    ratio against it is noise divided by noise, and printing "0.8 % of it" would
    dress that up as a finding. In that case say what is wrong instead.
    """
    index = {r["control"]: r for r in results}
    reference = index.get("zero_correction")
    if reference is None:
        return

    cell = reference["horizons"][horizon]
    denominator, sound = cell["damage"], cell["detected"] and cell["damage"] > 0

    print()
    if not sound:
        print(
            f"  CANNOT INTERPRET at h={horizon}. Zeroing the head changed MAE by "
            f"{denominator:+.3f} ug/m3"
        )
        print(
            f"  (95 % CI [{cell['ci_low']:+.2f}, {cell['ci_high']:+.2f}]), so the "
            f"learned correction on top of"
        )
        print(
            "  the persistence anchor is not measurably useful. With no scale to "
            "read them"
        )
        print(
            "  against, the controls below say nothing about the graph -- they say "
            "the model"
        )
        print("  has little to perturb. Fix that before drawing any conclusion.")
    else:
        print(
            f"  At h={horizon} the learned correction is worth {denominator:.2f} "
            f"ug/m3 over the persistence"
        )
        print("  anchor. Every control below is a fraction of that:")

    for record in results:
        if record["is_reference"]:
            continue
        row = record["horizons"][horizon]
        damage = row["damage"]
        if not row["detected"]:
            verdict = "no effect"
        elif damage > 0:
            verdict = "degrades"
        else:
            verdict = "IMPROVES -- investigate"
        share = (
            f"({100.0 * damage / denominator:>6.1f} % of it)" if sound else " " * 16
        )
        print(f"    {record['control']:22s}{damage:>+8.3f}  {share}   {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, nargs="+", required=True,
        help="one or more experiments/runs/<name> directories containing checkpoint.pt",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--occlusion", action="store_true",
        help="also run per-station occlusion (N+1 extra passes over the test split)",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="subsample the test split to N windows; wiring checks only",
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    payloads = [
        run_one(run_dir, device, args.occlusion, args.limit) for run_dir in args.runs
    ]

    if len(payloads) > 1:
        _across_seeds(payloads)


def _across_seeds(payloads: list[dict]) -> None:
    """A control that only fires for one seed has not found anything."""
    print(f"\n{'=' * 78}\nACROSS {len(payloads)} RUNS, damage at h=24\n{'=' * 78}")
    names = [c["control"] for c in payloads[0]["controls"]]
    print(f"  {'control':22s}{'mean':>10s}{'sd':>9s}{'detected':>12s}")
    for name in names:
        values, detected = [], 0
        for payload in payloads:
            record = next((c for c in payload["controls"] if c["control"] == name), None)
            if record is None:
                continue
            values.append(record["horizons"][24]["damage"])
            detected += bool(record["horizons"][24]["detected"])
        if not values:
            continue
        array = np.asarray(values)
        sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
        print(
            f"  {name:22s}{array.mean():>+10.3f}{sd:>9.3f}"
            f"{detected:>8d}/{len(values):<3d}"
        )


if __name__ == "__main__":
    main()
