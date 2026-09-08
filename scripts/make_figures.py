"""Build figures from saved runs.

    python scripts/make_figures.py --city beijing

Reads ``experiments/runs/*/predictions.npz`` and writes to
``experiments/figures/``. Every figure is regenerable from saved artifacts, so
none of them requires re-running a model.

Case-study dates come from ``analysis/find_episodes.py`` and all sit inside the
test split.
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
from typing import Sequence

import numpy as np

from src.data import contract
from src.figures import diagnostics, forecast_trace, graph_map
from src.metrics import Predictions, masked_mae

RUNS_ROOT = Path("experiments/runs")

# Test-split case studies, chosen in the evidence review.
EPISODES = {
    "severe_2017_01_01": ("2016-12-29T00", "2017-01-05T00",
                          "New Year 2017 — worst episode of the test year (peak 522 μg/m³)"),
    "fireworks_2017_01_28": ("2017-01-26T00", "2017-01-31T00",
                             "Chinese New Year 2017 — fireworks spike to 607 μg/m³, "
                             "unpredictable from the feature set"),
    "clearout_2016_03_04": ("2016-03-03T00", "2016-03-07T00",
                            "4 March 2016 — strongest clear-out in the test year "
                            "(369 → 56 μg/m³ in 12 h at 3.0 m/s)"),
}


def load_runs(runs_root: Path, names: list[str] | None) -> list[Predictions]:
    runs: list[Predictions] = []
    for path in sorted(runs_root.glob("*/predictions.npz")):
        run = Predictions.load(path)
        if names is None or run.model in names:
            runs.append(run)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="beijing")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--models", nargs="*", default=None, help="restrict to these model names")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    dataset = contract.load(args.city)
    runs = load_runs(args.runs_root, args.models)
    if not runs:
        raise SystemExit(f"no predictions found under {args.runs_root}")
    print(f"loaded {len(runs)} runs: {', '.join(r.model for r in runs)}\n")

    print("error by horizon:")
    forecast_trace.error_by_horizon(runs, metric="mae", name="error_by_horizon_mae")
    forecast_trace.error_by_horizon(runs, metric="rmse", name="error_by_horizon_rmse")

    # Episode traces use a readable subset -- ten overlapping lines is not a figure.
    highlight = [r for r in runs if r.model in ("persistence", "ridge", "dlinear", "lightgbm")]
    highlight = highlight or runs[:3]

    print("\nepisode traces:")
    for slug, (start, end, title) in EPISODES.items():
        forecast_trace.plot(
            dataset,
            highlight,
            start=start,
            end=end,
            horizon=args.horizon,
            title=f"{title}\nforecast at h={args.horizon}, network mean",
            name=slug,
        )

    # The graph figures need no trained model -- the prior is a deterministic
    # function of wind, distance and two fixed physical constants. They are
    # therefore available from day one, and the direction control is a test as
    # much as a figure.
    print("\ngraph (no model required):")
    graph_map.direction_check(dataset)
    graph_map.clearout_sequence(dataset)
    diagnostics.pollution_rose(dataset)

    best = _best_run(runs)
    if best is not None:
        print("\nper-station errors:")
        diagnostics.station_error_map(dataset, best, horizon=args.horizon)

    _regime_figure(dataset, runs)
    _diagnostic_figures(dataset, args.runs_root)

    print("\ndone")


def _best_run(runs: Sequence[Predictions]) -> Predictions | None:
    """Lowest mean MAE across the reported horizons."""
    scored = [
        (np.mean([
            masked_mae(r.pred[:, :, h - 1], r.truth[:, :, h - 1], r.mask[:, :, h - 1])
            for h in (1, 6, 12, 24)
        ]), r)
        for r in runs
    ]
    return min(scored, key=lambda pair: pair[0])[1] if scored else None


def _regime_figure(dataset, runs: Sequence[Predictions]) -> None:
    """Wind-regime slices, once both halves of a comparison exist.

    Prefers the headline pair (the graph in isolation); falls back to
    persistence against the best available model so the figure exists early.
    """
    by_name = {r.model: r for r in runs}
    for baseline, candidate in (
        ("s0_static_concat", "d0_wind_concat"),
        ("persistence", "lightgbm"),
    ):
        if baseline in by_name and candidate in by_name:
            print(f"\nregime slices ({baseline} -> {candidate}):")
            diagnostics.regime_slices(dataset, by_name[baseline], by_name[candidate])
            return


def _diagnostic_figures(dataset, runs_root: Path) -> None:
    """Attention and gate plots, for every run that has dumped diagnostics."""
    for path in sorted(runs_root.glob("*/diagnostics.npz")):
        with np.load(path) as payload:
            available = dict(payload)
        print(f"\ndiagnostics from {path.parent.name}:")
        if "attention" in available and "wind_speed" in available:
            diagnostics.attention_by_regime(
                dataset, available, name=f"attention_{path.parent.name}"
            )
        if "gate" in available:
            diagnostics.gate_by_patch(available, name=f"gate_{path.parent.name}")


if __name__ == "__main__":
    main()
