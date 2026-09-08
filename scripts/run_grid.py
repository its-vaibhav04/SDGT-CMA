"""Run the experiment grid: five configurations across several seeds.

    python scripts/run_grid.py --seeds 42 43 44

Every run uses the identical split, preprocessing, target masks and tuning
budget; only ``graph.type``, ``spatial.type`` and ``fusion.type`` differ. That is
what makes each pairwise difference attributable to one component.

After the grid, the headline comparison (S0 -> D0, the graph in isolation) is
reported with a paired weekly block bootstrap. Absolute MAE is *not* used to rank
models: one test year holds 53 independent weeks and carries a +/-12.6 %
interval on any absolute figure.
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

from src import bootstrap
from src.bootstrap import seed_spread
from src.metrics import Predictions, masked_mae
from src.train import RUNS_ROOT, load_config, train

GRID = ["t0", "s0", "d0", "s1", "d1"]
HORIZONS = (1, 6, 12, 24)

# Each pair isolates exactly one component.
COMPARISONS = [
    ("s0_static_concat", "d0_wind_concat", "the graph, with fusion held fixed"),
    ("s0_static_concat", "s1_static_crossview", "the fusion, with the graph held fixed"),
    ("d0_wind_concat", "d1_wind_crossview", "the fusion, on the dynamic graph"),
    ("t0_temporal_only", "s0_static_concat", "adding a graph at all"),
]


def collect(runs_root: Path, name: str, seeds: list[int]) -> list[Predictions]:
    found = []
    for seed in seeds:
        path = runs_root / f"{name}_seed{seed}" / "predictions.npz"
        if path.exists():
            found.append(Predictions.load(path))
    return found


def report(runs_root: Path, seeds: list[int]) -> None:
    names = sorted({p.model for p in collect_all(runs_root, seeds)})
    print(f"\n{'=' * 78}\nRESULTS: mean +/- spread across {len(seeds)} seeds, MAE in ug/m3\n{'=' * 78}")
    print(f"{'model':28s}" + "".join(f"{'h=' + str(h):>12s}" for h in HORIZONS))

    for name in names:
        runs = collect(runs_root, name, seeds)
        if not runs:
            continue
        cells = []
        for h in HORIZONS:
            values = [
                masked_mae(r.pred[:, :, h - 1], r.truth[:, :, h - 1], r.mask[:, :, h - 1])
                for r in runs
            ]
            mean, spread = seed_spread(values)
            cells.append(f"{mean:7.2f}+/-{spread:4.2f}" if len(values) > 1 else f"{mean:11.2f}")
        print(f"{name:28s}" + "".join(f"{c:>12s}" for c in cells))

    print(f"\n{'=' * 78}\nPAIRED COMPARISONS (positive = the second model is better)\n{'=' * 78}")
    print(
        "Two independent sources of variance, and a claim must survive both:\n"
        "  block bootstrap -> which 53 test weeks you happened to draw\n"
        "  seed spread     -> where the optimiser happened to land\n"
        "A difference is ROBUST only when every seed agrees in sign and every\n"
        "seed's own interval excludes zero."
    )
    for baseline_name, candidate_name, description in COMPARISONS:
        baselines = {r.seed: r for r in collect(runs_root, baseline_name, seeds)}
        candidates = {r.seed: r for r in collect(runs_root, candidate_name, seeds)}
        shared = sorted(set(baselines) & set(candidates))
        if not shared:
            continue

        print(f"\n{baseline_name} -> {candidate_name}   [{description}]")
        print(f"  {'h':>3s}  {'diff (mean +/- seed sd)':>28s}  {'CIs excluding 0':>16s}  verdict")

        for h in HORIZONS:
            intervals = [
                bootstrap.paired_interval(baselines[s], candidates[s], horizon=h, n_boot=600)
                for s in shared
            ]
            points = [i.point for i in intervals]
            mean, spread = seed_spread(points)
            n_significant = sum(i.significant for i in intervals)
            same_sign = all(p > 0 for p in points) or all(p < 0 for p in points)

            if n_significant == len(shared) and same_sign:
                verdict = "robust"
            elif n_significant:
                verdict = "seed-dependent"
            else:
                verdict = "comparable"

            print(
                f"  {h:>3d}  {mean:>+16.3f} +/- {spread:<8.3f}"
                f"  {n_significant:>8d}/{len(shared):<7d}  {verdict}"
            )

        if len(shared) < 3:
            print(f"  NOTE: only {len(shared)} seed(s) -- the spread is not yet meaningful")


def collect_all(runs_root: Path, seeds: list[int]) -> list[Predictions]:
    """Every grid run for the given seeds. Baselines live alongside but are
    identified by name prefix, so they are not mixed into the grid table."""
    prefixes = {stem for stem in GRID}
    out = []
    for path in sorted(runs_root.glob("*/predictions.npz")):
        run = Predictions.load(path)
        if run.seed in seeds and run.model.split("_")[0] in prefixes:
            out.append(run)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--configs", nargs="+", default=GRID)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    if not args.report_only:
        total = len(args.configs) * len(args.seeds)
        done = 0
        for stem in args.configs:
            config = load_config(Path("configs/model") / f"{stem}.yaml")
            if args.epochs is not None:
                config["train"]["epochs"] = args.epochs
            for seed in args.seeds:
                done += 1
                print(f"\n[{done}/{total}] {stem} seed={seed}")
                train(config, seed, args.runs_root, args.device, args.smoke)

    report(args.runs_root, args.seeds)


if __name__ == "__main__":
    main()
