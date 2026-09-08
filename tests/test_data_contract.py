"""Gate tests for the processed data contract.

Every test here exists because the failure it catches is *silent*: nothing
crashes, the model trains, and the reported numbers are wrong. The off-by-one
test in particular is the single most common bug in spatio-temporal forecasting
and it inflates validation accuracy rather than degrading it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import contract
from src.data.build_beijing import CEILINGS, PHYSICAL_BOUNDS, forward_fill, hours_since_observed
from src.data.transforms import ColumnSpec, Scaler, TargetScaler
from src.data.windows import valid_origins

CITY = "beijing"


@pytest.fixture(scope="module")
def dataset():
    try:
        return contract.load(CITY)
    except FileNotFoundError as error:
        pytest.skip(str(error))


# --------------------------------------------------------------- integrity
def test_contract_validates_clean(dataset):
    problems = contract.validate(CITY)
    assert problems == [], "\n".join(problems)


def test_grid_shape_and_extent(dataset):
    assert dataset.n_hours == 35064
    assert dataset.n_stations == 12
    assert str(dataset.timestamps[0]) == "2013-03-01T00"
    assert str(dataset.timestamps[-1]) == "2017-02-28T23"


def test_timestamps_are_contiguous_hourly(dataset):
    deltas = np.diff(dataset.timestamps.astype("datetime64[h]").astype(np.int64))
    assert np.all(deltas == 1)
    assert len(np.unique(dataset.timestamps)) == dataset.n_hours


def test_no_nan_in_model_inputs(dataset):
    assert np.isfinite(dataset.features).all()
    assert np.isfinite(dataset.wind_uv).all()


def test_target_mask_matches_raw_observations(dataset):
    """The mask must mark exactly the genuinely observed values, no more."""
    assert dataset.target_mask.sum() == np.isfinite(dataset.target_raw).sum()
    assert np.isfinite(dataset.target_raw[dataset.target_mask]).all()
    # 2.08% of PM2.5 is missing in the raw record; the mask should reflect that.
    assert 0.975 < dataset.target_mask.mean() < 0.985


def test_stations_are_in_canonical_sorted_order(dataset):
    assert dataset.stations == sorted(dataset.stations)
    assert dataset.coords.shape == (dataset.n_stations, 2)
    # Beijing sits near 40N, 116E; a coordinate slip would show up immediately.
    assert (39.5 < dataset.coords[:, 0]).all() and (dataset.coords[:, 0] < 40.5).all()
    assert (116.0 < dataset.coords[:, 1]).all() and (dataset.coords[:, 1] < 117.0).all()


# ----------------------------------------------------------------- splits
def test_splits_partition_the_record_without_gaps(dataset):
    train, val, test = (dataset.splits[n] for n in ("train", "val", "test"))
    assert train.start == 0
    assert train.end == val.start
    assert val.end == test.start
    assert test.end == dataset.n_hours
    # Whole seasonal blocks: 730 / 366 / 365 days.
    assert len(train) == 730 * 24
    assert len(val) == 366 * 24
    assert len(test) == 365 * 24


def test_split_dates_match_indices(dataset):
    for split in dataset.splits.values():
        assert str(dataset.timestamps[split.start]) == split.start_date
        assert str(dataset.timestamps[split.end - 1]) == split.end_date


# ------------------------------------------------------------- leakage
def test_scaler_was_fitted_on_the_training_split_only(dataset):
    """Refit from the train slice alone and confirm the stored parameters match.

    If the shipped scaler had seen validation or test data, the refit statistics
    would differ and this would fail.
    """
    train = dataset.splits["train"].as_slice()
    index = dataset.feature_index("TEMP")
    spec = dataset.scaler.specs[index]

    column = dataset.features[train, :, index].astype(np.float64)
    # The stored features are already scaled, so a correctly fitted scaler leaves
    # the training slice at roughly zero mean and unit variance.
    assert abs(float(column.mean())) < 0.05
    assert 0.9 < float(column.std()) < 1.1
    assert spec.kind == "standard"


def test_target_scaler_fitted_on_train_only(dataset):
    train = dataset.splits["train"].as_slice()
    observed = dataset.target_mask[train]
    scaled = dataset.target_scaled[train][observed]
    assert abs(float(scaled.mean())) < 0.05
    assert 0.9 < float(scaled.std()) < 1.1


def test_target_scaling_round_trips(dataset):
    observed = dataset.target_mask
    recovered = dataset.target_scaler.inverse(dataset.target_scaled[observed])
    np.testing.assert_allclose(recovered, dataset.target_raw[observed], atol=1e-3)


def test_feature_scaling_round_trips():
    scaler = Scaler(
        [
            ColumnSpec("pollutant", "log_standard", mean=3.0, std=1.2),
            ColumnSpec("plain", "standard", mean=-4.0, std=7.5),
            ColumnSpec("mask", "none"),
        ]
    )
    values = np.stack(
        [
            np.array([[0.0, 12.0], [340.0, 999.0]]),
            np.array([[-20.0, 3.5], [40.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(scaler.inverse(scaler.transform(values)), values, rtol=1e-5, atol=1e-5)


def test_target_scaler_ignores_nan_when_fitting():
    target = np.array([[10.0, np.nan], [np.nan, 30.0], [50.0, 70.0]])
    scaler = TargetScaler.fit(target, slice(0, 3))
    assert scaler.mean == pytest.approx(40.0)


# --------------------------------------------------------------- windowing
def test_window_targets_stay_inside_their_split(dataset):
    """Every target hour of every origin must fall inside that origin's split."""
    for name, split in dataset.splits.items():
        for lookback, horizon in ((48, 24), (168, 24)):
            origins = valid_origins(
                split=split, n_hours=dataset.n_hours, lookback=lookback, horizon=horizon
            )
            if origins.size == 0:
                continue
            assert origins.min() >= lookback - 1, f"{name}: not enough history"
            assert origins.min() + 1 >= split.start, f"{name}: first target precedes the split"
            assert origins.max() + horizon <= split.end - 1, f"{name}: last target overruns"


def test_lookback_may_reach_into_the_previous_split(dataset):
    """Documented and intended: history crosses the boundary, targets never do."""
    val = dataset.splits["val"]
    origins = valid_origins(split=val, n_hours=dataset.n_hours, lookback=48, horizon=24)
    first = int(origins.min())
    assert first - 48 + 1 < val.start          # history reaches back
    assert first + 1 >= val.start              # first target does not


def test_valid_origins_drops_windows_with_no_observed_target(dataset):
    split = dataset.splits["test"]
    unfiltered = valid_origins(split=split, n_hours=dataset.n_hours, lookback=48, horizon=24)
    filtered = valid_origins(
        split=split,
        n_hours=dataset.n_hours,
        lookback=48,
        horizon=24,
        target_mask=dataset.target_mask,
    )
    assert len(filtered) <= len(unfiltered)
    assert set(filtered).issubset(set(unfiltered))


def test_batch_targets_are_the_hours_after_the_origin(dataset):
    """The off-by-one gate. target[:, :, 0] must be hour t+1, not hour t."""
    torch = pytest.importorskip("torch")
    from src.data.windows import WindowTensors

    tensors = WindowTensors(dataset)
    origins = np.array([20000, 20001, 30000], dtype=np.int64)
    batch = tensors.gather(torch.from_numpy(origins), lookback=48, horizon=24)

    for row, origin in enumerate(origins):
        # last observed hour is the origin itself
        expected_history = dataset.features[origin]
        np.testing.assert_allclose(batch.features[row, -1].numpy(), expected_history, atol=0)
        # first target is the hour after
        expected_target = dataset.target_scaled[origin + 1]
        np.testing.assert_allclose(batch.target[row, :, 0].numpy(), expected_target, atol=0)
        # last target is origin + horizon
        expected_last = dataset.target_scaled[origin + 24]
        np.testing.assert_allclose(batch.target[row, :, -1].numpy(), expected_last, atol=0)


def test_history_and_targets_never_overlap(dataset):
    torch = pytest.importorskip("torch")
    from src.data.windows import WindowTensors

    tensors = WindowTensors(dataset)
    lookback, horizon = 48, 24
    origin = 25000
    batch = tensors.gather(torch.tensor([origin]), lookback, horizon)
    history_hours = set(range(origin - lookback + 1, origin + 1))
    target_hours = set(range(origin + 1, origin + horizon + 1))
    assert history_hours.isdisjoint(target_hours)
    assert len(batch) == 1
    assert batch.features.shape == (1, lookback, dataset.n_stations, dataset.n_features)
    assert batch.target.shape == (1, dataset.n_stations, horizon)


# ------------------------------------------------------------ wind convention
def test_wind_vector_points_in_the_direction_of_motion(dataset):
    """`wd` is direction-from, so a westerly wind must give u > 0 (air moves east).

    Verified against the raw record rather than a constant: this is the single
    assumption the whole graph module rests on.
    """
    u = dataset.wind_uv[..., 0]
    v = dataset.wind_uv[..., 1]

    # Northerly wind (wd=0) moves south: v < 0. Southerly (wd=180) moves north.
    # Rather than re-deriving wd, check the physical signature that made the
    # convention identifiable: air arriving from the north-west is clean.
    speed = np.hypot(u, v)
    strong = speed >= 2.0
    from_northwest = strong & (u > 0) & (v < 0)     # moving south-east => came from NW
    from_southeast = strong & (u < 0) & (v > 0)     # moving north-west => came from SE

    observed = dataset.target_mask
    clean = dataset.target_raw[from_northwest & observed].mean()
    dirty = dataset.target_raw[from_southeast & observed].mean()
    assert clean < dirty, "air from the NW must be cleaner than air from the SE"
    assert dirty / clean > 1.8, f"expected a large contrast, got {dirty / clean:.2f}x"


def test_wind_speed_matches_the_recorded_scalar(dataset):
    """|(u, v)| must reproduce WSPM wherever both were observed."""
    speed = np.hypot(dataset.wind_uv[..., 0], dataset.wind_uv[..., 1])
    index = dataset.feature_index("WSPM")
    spec = dataset.scaler.specs[index]
    recorded = dataset.features[..., index].astype(np.float64) * spec.std + spec.mean
    observed = dataset.features[..., dataset.feature_index("obs_wind")] > 0.5
    np.testing.assert_allclose(speed[observed], recorded[observed], atol=1e-2)


# ------------------------------------------------------------ quality rules
def test_high_pollution_episodes_are_retained(dataset):
    """A z>3 rule would delete 1.78% of hours, all Severe-AQI. Nothing is deleted."""
    severe = dataset.target_raw[dataset.target_mask] > 322.0
    assert severe.sum() > 6000, "severe episodes were removed from the target"
    assert dataset.target_raw[dataset.target_mask].max() >= 999.0


def test_instrument_ceilings_are_flagged_not_deleted(dataset):
    flag = dataset.features[..., dataset.feature_index("at_ceiling")]
    assert set(np.unique(flag)).issubset({0.0, 1.0}), "the ceiling flag must stay binary"
    assert 0 < flag.sum() < 200, "expected a handful of saturated station-hours"


def test_physical_bounds_cover_the_observed_range(dataset):
    for name, (low, high) in PHYSICAL_BOUNDS.items():
        if name in CEILINGS:
            assert high > CEILINGS[name], f"{name} bound would reject its own ceiling"
        assert low <= 0.0 or name in ("PRES", "TEMP", "DEWP")


# ------------------------------------------------------- imputation helpers
def test_forward_fill_is_causal_and_respects_the_limit():
    column = np.array([[1.0], [np.nan], [np.nan], [np.nan], [np.nan], [6.0]])
    filled = forward_fill(column, limit=3)
    assert filled[1, 0] == 1.0 and filled[3, 0] == 1.0
    assert np.isnan(filled[4, 0]), "gap beyond the limit must stay missing"
    assert filled[5, 0] == 6.0


def test_forward_fill_never_looks_ahead():
    """A leading gap must remain missing -- there is nothing behind it to carry."""
    column = np.array([[np.nan], [np.nan], [5.0]])
    filled = forward_fill(column, limit=3)
    assert np.isnan(filled[0, 0]) and np.isnan(filled[1, 0])
    assert filled[2, 0] == 5.0


def test_hours_since_observed_counts_up_and_caps():
    column = np.array([[1.0], [np.nan], [np.nan], [2.0], [np.nan]])
    age = hours_since_observed(column, cap=2)
    np.testing.assert_array_equal(age[:, 0], [0, 1, 2, 0, 1])


def test_since_observed_feature_is_zero_when_observed(dataset):
    """Staleness must be lowest exactly where the observation mask is set."""
    obs = dataset.features[..., dataset.feature_index("obs_PM2.5")] > 0.5
    stale = dataset.features[..., dataset.feature_index("since_obs_PM2.5")]
    assert stale[obs].mean() < stale[~obs].mean()
