"""Gates on the figure layer.

The failure these exist for: ``make_figures.py`` is the last cell of a
multi-hour Kaggle run, and it died with

    ValueError: 18 contending models exceeds the 5-colour palette

after the full grid had finished. Two mistakes were behind it. The figure drew
one line per *(configuration, seed)* rather than one per configuration, so five
configurations across three seeds became fifteen series. And the palette guard
raised instead of degrading, throwing away a completed run over a plotting
detail.

Figures are cheap to regenerate and expensive to lose. Nothing here should ever
be able to kill a pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.figures import forecast_trace
from src.metrics import Predictions

HORIZONS = (1, 6, 12, 24)


def make_run(model: str, seed: int = 42, offset: float = 0.0, n: int = 40) -> Predictions:
    rng = np.random.default_rng(abs(hash((model, seed))) % (2**32))
    truth = rng.uniform(20, 200, size=(n, 3, 24)).astype(np.float32)
    pred = truth + rng.normal(offset, 5.0, size=truth.shape).astype(np.float32)
    return Predictions(
        origins=np.arange(n, dtype=np.int64) * 168,
        pred=pred,
        truth=truth,
        mask=np.ones_like(truth, dtype=bool),
        model=model,
        seed=seed,
    )


# ------------------------------------------------------------ seed handling
def test_seeds_of_one_config_collapse_to_one_series():
    """Three seeds of one configuration is one line with a band, not three lines."""
    runs = [make_run("d1_wind_crossview", seed) for seed in (42, 43, 44)]
    grouped = forecast_trace._group_by_model(runs)
    assert list(grouped) == ["d1_wind_crossview"]
    assert [r.seed for r in grouped["d1_wind_crossview"]] == [42, 43, 44]


def test_seed_curves_report_mean_and_range():
    runs = [make_run("d1", seed, offset=o) for seed, o in zip((42, 43, 44), (0.0, 4.0, 8.0))]
    mean, low, high = forecast_trace._curves(runs, HORIZONS, "mae")
    assert len(mean) == len(low) == len(high) == len(HORIZONS)
    for m, lo, hi in zip(mean, low, high):
        assert lo <= m <= hi
    assert any(hi > lo for lo, hi in zip(low, high)), "seeds differ, so the band must be visible"


def test_single_seed_produces_a_degenerate_band():
    mean, low, high = forecast_trace._curves([make_run("t0")], HORIZONS, "mae")
    assert low == high == mean


# ------------------------------------------------- the failure that happened
def test_full_grid_across_three_seeds_does_not_raise(tmp_path):
    """The exact Kaggle situation: 5 configurations x 3 seeds plus baselines."""
    runs = []
    for config in ("t0_temporal_only", "s0_static_concat", "d0_wind_concat",
                   "s1_static_crossview", "d1_wind_crossview"):
        runs += [make_run(config, seed) for seed in (42, 43, 44)]
    runs += [make_run(name) for name in ("persistence", "lightgbm", "ridge", "dlinear",
                                         "seasonal_naive_24h", "climatology")]
    assert len(runs) == 21

    written = forecast_trace.error_by_horizon(
        runs, name="grid_test", root=tmp_path, references=("lightgbm",)
    )
    assert written and all(path.exists() for path in written)


def test_too_many_contenders_splits_rather_than_raising(tmp_path):
    """Beyond the palette, emit several figures -- never abort the pipeline."""
    runs = [make_run(f"model_{i}") for i in range(13)]
    written = forecast_trace.error_by_horizon(runs, name="many", root=tmp_path)
    assert len(written) == 3, "13 contenders over a 5-colour palette needs 3 figures"
    assert all(path.exists() for path in written)


def test_references_are_not_counted_as_contenders(tmp_path):
    """Naming a model as a reference must keep it out of the coloured series.

    Five grid configurations plus persistence and LightGBM has to stay one
    figure; if the references leaked into the contender list it would split.
    """
    runs = [make_run(f"cfg_{i}") for i in range(5)]
    runs += [make_run("persistence"), make_run("lightgbm"), make_run("climatology")]
    written = forecast_trace.error_by_horizon(
        runs, name="refs", root=tmp_path, references=("lightgbm",)
    )
    assert len(written) == 1


def test_no_categorical_hue_is_reused_for_a_reference():
    """A reference sharing a series colour reads as though the two are related."""
    from src.figures import style

    assert style.REFERENCE_STRONG not in style.SERIES
    assert style.REFERENCE not in style.SERIES
    assert style.TRUTH not in style.SERIES
    assert len(set(style.SERIES)) == len(style.SERIES), "duplicate hue in the palette"


def test_empty_run_list_is_handled(tmp_path):
    written = forecast_trace.error_by_horizon([], name="empty", root=tmp_path)
    assert len(written) == 1        # an empty axes, not a crash


@pytest.mark.parametrize("metric", ["mae", "rmse", "wape"])
def test_every_reported_metric_works(metric, tmp_path):
    runs = [make_run("a"), make_run("b"), make_run("persistence")]
    written = forecast_trace.error_by_horizon(runs, metric=metric, name=metric, root=tmp_path)
    assert written[0].exists()
