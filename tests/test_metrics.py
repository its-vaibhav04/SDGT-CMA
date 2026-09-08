"""Metric and bootstrap gates, against hand-calculated fixtures.

A metrics module that is silently wrong is worse than no metrics at all: every
downstream conclusion inherits the error and nothing looks broken. These
fixtures are small enough to verify by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import bootstrap, metrics
from src.metrics import Predictions


def make(pred, truth, mask=None, model="m", origins=None) -> Predictions:
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    if origins is None:
        origins = np.arange(len(pred), dtype=np.int64)
    return Predictions(origins=origins, pred=pred, truth=truth, mask=np.asarray(mask), model=model)


# ---------------------------------------------------------------- primitives
def test_mae_matches_hand_calculation():
    pred = np.array([[[10.0, 20.0]]])
    truth = np.array([[[13.0, 16.0]]])
    # errors 3 and 4 -> mean 3.5
    assert metrics.masked_mae(pred, truth, np.ones_like(pred, bool)) == pytest.approx(3.5)


def test_rmse_matches_hand_calculation():
    pred = np.array([[[10.0, 20.0]]])
    truth = np.array([[[13.0, 16.0]]])
    # squared errors 9 and 16 -> mean 12.5 -> sqrt = 3.5355...
    assert metrics.masked_rmse(pred, truth, np.ones_like(pred, bool)) == pytest.approx(12.5**0.5)


def test_wape_is_pooled_not_averaged():
    pred = np.array([[[9.0, 90.0]]])
    truth = np.array([[[10.0, 100.0]]])
    # sum|err| = 11, sum|truth| = 110 -> 10%
    assert metrics.masked_wape(pred, truth, np.ones_like(pred, bool)) == pytest.approx(10.0)


def test_bias_keeps_its_sign():
    pred = np.array([[[8.0, 8.0]]])
    truth = np.array([[[10.0, 10.0]]])
    assert metrics.masked_bias(pred, truth, np.ones_like(pred, bool)) == pytest.approx(-2.0)


def test_perfect_prediction_scores_zero_error_and_unit_r2():
    truth = np.array([[[1.0, 5.0, 9.0]]])
    mask = np.ones_like(truth, bool)
    assert metrics.masked_mae(truth, truth, mask) == 0.0
    assert metrics.masked_rmse(truth, truth, mask) == 0.0
    assert metrics.masked_r2(truth, truth, mask) == pytest.approx(1.0)


# ------------------------------------------------------------------ masking
def test_masked_entries_are_excluded_entirely():
    """The gate that stops imputed labels being scored as truth."""
    pred = np.array([[[10.0, 999.0]]])
    truth = np.array([[[12.0, -999.0]]])
    mask = np.array([[[True, False]]])
    assert metrics.masked_mae(pred, truth, mask) == pytest.approx(2.0)
    assert metrics.masked_rmse(pred, truth, mask) == pytest.approx(2.0)


def test_all_masked_returns_nan_not_zero():
    """An empty slice must be visibly absent, never a flattering zero."""
    pred = np.array([[[1.0]]])
    truth = np.array([[[2.0]]])
    mask = np.zeros_like(pred, bool)
    assert np.isnan(metrics.masked_mae(pred, truth, mask))
    assert np.isnan(metrics.masked_rmse(pred, truth, mask))


def test_zero_targets_do_not_blow_up_wape():
    pred = np.array([[[1.0, 1.0]]])
    truth = np.array([[[0.0, 0.0]]])
    assert np.isnan(metrics.masked_wape(pred, truth, np.ones_like(pred, bool)))


# --------------------------------------------------------------- aggregation
def test_by_horizon_indexes_from_one():
    pred = np.array([[[1.0, 2.0, 3.0]]])
    truth = np.array([[[1.0, 2.0, 30.0]]])
    result = metrics.by_horizon(make(pred, truth), horizons=(1, 3))
    assert result[1]["mae"] == pytest.approx(0.0)
    assert result[3]["mae"] == pytest.approx(27.0)


def test_by_horizon_rejects_out_of_range():
    predictions = make(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    with pytest.raises(ValueError):
        metrics.by_horizon(predictions, horizons=(4,))


def test_by_station_separates_stations():
    pred = np.array([[[1.0], [1.0]]])
    truth = np.array([[[2.0], [11.0]]])
    result = metrics.by_station(make(pred, truth), ["A", "B"], horizons=(1,))
    assert result["A"][1] == pytest.approx(1.0)
    assert result["B"][1] == pytest.approx(10.0)


def test_by_slice_restricts_to_selected_windows():
    pred = np.array([[[1.0]], [[1.0]]])
    truth = np.array([[[2.0]], [[100.0]]])
    selector = np.array([True, False])
    result = metrics.by_slice(make(pred, truth), selector, horizons=(1,))
    assert result[1]["mae"] == pytest.approx(1.0)


def test_top_decile_mask_selects_the_high_tail():
    truth = np.arange(100, dtype=np.float32).reshape(100, 1, 1)
    predictions = make(np.zeros_like(truth), truth)
    selected = metrics.top_decile_mask(predictions, quantile=0.9)
    assert selected.sum() == 10
    assert truth[selected].min() >= 89.0


# ------------------------------------------------------------- round-tripping
def test_predictions_survive_a_save_load_cycle(tmp_path):
    original = make(
        np.random.default_rng(0).normal(size=(5, 3, 4)),
        np.random.default_rng(1).normal(size=(5, 3, 4)),
        model="test_model",
    )
    path = tmp_path / "predictions.npz"
    original.save(path)
    restored = Predictions.load(path)

    np.testing.assert_allclose(restored.pred, original.pred)
    np.testing.assert_allclose(restored.truth, original.truth)
    np.testing.assert_array_equal(restored.mask, original.mask)
    np.testing.assert_array_equal(restored.origins, original.origins)
    assert restored.model == "test_model"


def test_shape_mismatch_is_rejected_at_construction():
    with pytest.raises(ValueError):
        Predictions(
            origins=np.arange(2),
            pred=np.zeros((2, 3, 4)),
            truth=np.zeros((2, 3, 5)),
            mask=np.ones((2, 3, 4), bool),
        )


# ---------------------------------------------------------------- bootstrap
def _paired_fixture(n_weeks=20, per_week=8, improvement=0.10, seed=0):
    """Two models on identical rows, where error magnitude varies a lot by week.

    This mirrors the real situation: a severe-pollution week produces far larger
    errors than a clean one, so the *absolute* error swings enormously week to
    week while the *relative* difference between two models stays steady. That is
    exactly the regime in which paired comparison wins.
    """
    rng = np.random.default_rng(seed)
    origins, base, cand, truth = [], [], [], []
    for week in range(n_weeks):
        severity = rng.uniform(20.0, 200.0)          # week-level pollution
        for slot in range(per_week):
            origins.append(week * bootstrap.HOURS_PER_WEEK + slot)
            error = rng.normal(0.0, 0.3 * severity)  # error scales with severity
            truth.append(severity)
            base.append(severity + error)
            cand.append(severity + error * (1.0 - improvement))
    origins = np.array(origins, dtype=np.int64)
    shape = (len(origins), 1, 1)
    return (
        make(np.array(base).reshape(shape), np.array(truth).reshape(shape),
             model="baseline", origins=origins),
        make(np.array(cand).reshape(shape), np.array(truth).reshape(shape),
             model="candidate", origins=origins),
    )


def test_paired_interval_detects_a_real_difference():
    baseline, candidate = _paired_fixture(improvement=0.10)
    interval = bootstrap.paired_interval(baseline, candidate, horizon=1, n_boot=400)
    assert interval.point > 0, "candidate should be better"
    assert interval.significant


def test_paired_interval_reports_no_difference_when_there_is_none():
    baseline, candidate = _paired_fixture(improvement=0.0, seed=3)
    interval = bootstrap.paired_interval(baseline, candidate, horizon=1, n_boot=400)
    assert not interval.significant


def test_pairing_resolves_what_absolute_intervals_cannot():
    """The core reason the protocol is paired: shared variance cancels.

    With week-to-week error variation far larger than the model difference, the
    two models' absolute confidence intervals overlap almost completely -- you
    could not tell them apart by comparing headline numbers. The paired interval
    on the same rows still excludes zero.

    This is the measured situation on Beijing: +/-12.6 % on an absolute MAE
    against +/-0.15 % on a paired difference.
    """
    baseline, candidate = _paired_fixture(improvement=0.10)

    base_interval = bootstrap.absolute_interval(baseline, horizon=1, n_boot=400)
    cand_interval = bootstrap.absolute_interval(candidate, horizon=1, n_boot=400)
    paired = bootstrap.paired_interval(baseline, candidate, horizon=1, n_boot=400)

    # The absolute intervals overlap, so they cannot rank the two models.
    overlap = min(base_interval.high, cand_interval.high) - max(base_interval.low, cand_interval.low)
    assert overlap > 0, "fixture should produce overlapping absolute intervals"

    # The paired comparison resolves the same difference cleanly.
    assert paired.significant
    assert paired.point > 0


def test_paired_interval_requires_identical_origin_sets():
    baseline, candidate = _paired_fixture()
    shifted = Predictions(
        origins=candidate.origins + 1,
        pred=candidate.pred,
        truth=candidate.truth,
        mask=candidate.mask,
        model="shifted",
    )
    with pytest.raises(ValueError, match="identical origin sets"):
        bootstrap.paired_interval(baseline, shifted, horizon=1, n_boot=10)


def test_week_ids_group_by_168_hours():
    origins = np.array([0, 167, 168, 335, 336])
    np.testing.assert_array_equal(bootstrap.week_ids(origins), [0, 0, 1, 1, 2])


def test_seed_spread_reports_mean_and_deviation():
    mean, spread = bootstrap.seed_spread([10.0, 12.0, 14.0])
    assert mean == pytest.approx(12.0)
    assert spread == pytest.approx(2.0)


def test_seed_spread_of_single_value_has_zero_deviation():
    mean, spread = bootstrap.seed_spread([7.0])
    assert mean == pytest.approx(7.0)
    assert spread == 0.0
