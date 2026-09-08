"""Run every baseline and write predictions plus a metrics table.

    python scripts/run_baselines.py --city beijing

This is the Phase 2 gate: one command from the processed dataset to scored
predictions and a figure. Every baseline writes the same ``predictions.npz``
format on the same origin set, so any pair is directly comparable and the paired
bootstrap is valid without further alignment.
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

from src.baselines import linear, naive
from src.data import contract
from src.data.windows import valid_origins
from src.metrics import format_table, summary_table
from src.utils.manifest import capture_environment, write_json
from src.utils.seeding import seed_everything

RUNS_ROOT = Path("experiments/runs")


def split_origins(dataset, lookback: int, horizon: int) -> dict[str, np.ndarray]:
    """Valid forecast origins per split, filtered to windows with an observed target."""
    return {
        name: valid_origins(
            split=split,
            n_hours=dataset.n_hours,
            lookback=lookback,
            horizon=horizon,
            target_mask=dataset.target_mask,
        )
        for name, split in dataset.splits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="beijing")
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--skip-trees", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    dataset = contract.load(args.city)
    print(dataset.describe())

    origins = split_origins(dataset, args.lookback, args.horizon)
    print(
        "\norigins: "
        + "  ".join(f"{name}={len(idx)}" for name, idx in origins.items())
        + f"   (lookback={args.lookback}, horizon={args.horizon})"
    )

    results = []
    print("\nrunning baselines...")

    results.append(naive.persistence(dataset, origins["test"], args.horizon, args.seed))
    print("  persistence")
    results.append(naive.seasonal_naive(dataset, origins["test"], args.horizon, 24, args.seed))
    print("  seasonal naive 24h")
    results.append(naive.seasonal_naive(dataset, origins["test"], args.horizon, 168, args.seed))
    print("  seasonal naive 168h")
    results.append(naive.climatology(dataset, origins["test"], args.horizon, args.seed))
    print("  climatology")

    results.append(
        linear.ridge(
            dataset, origins, lookback=args.lookback, horizon=args.horizon, seed=args.seed
        )
    )
    print("  ridge")
    results.append(
        linear.dlinear(
            dataset, origins, lookback=args.lookback, horizon=args.horizon, seed=args.seed
        )
    )
    print("  dlinear")

    if not args.skip_trees:
        try:
            from src.baselines import trees

            results.append(
                trees.lightgbm_direct(
                    dataset, origins, lookback=args.lookback, horizon=args.horizon, seed=args.seed
                )
            )
            print("  lightgbm")
        except ImportError:
            print("  lightgbm not installed -- skipping (pip install lightgbm)")

    rows = []
    for predictions in results:
        run_dir = args.runs_root / f"{predictions.model}_seed{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions.save(run_dir / "predictions.npz")
        table = summary_table(predictions)
        write_json(run_dir / "metrics.json", {"rows": table})
        write_json(run_dir / "env.json", capture_environment())
        rows.extend(table)

    print("\n" + format_table(rows))

    combined = args.runs_root / f"baselines_{args.city}_seed{args.seed}.json"
    write_json(combined, {"rows": rows, "lookback": args.lookback, "horizon": args.horizon})
    print(f"\nwrote {len(results)} baselines to {args.runs_root}")
    print(f"metrics table: {combined}")


if __name__ == "__main__":
    main()
