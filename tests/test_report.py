"""Gates on the notebook's analysis tables.

Every table the Kaggle notebook shows the mentor comes from ``src/report.py``.
These tests build a tiny synthetic run directory with *known* numbers -- a
persistence baseline at exactly 10 ug/m3, a model at exactly 8 -- so each
function can be checked for the right arithmetic rather than for "returns a
DataFrame". A skill score of 20 % is a fact about the fixture, not a hope.

The second group runs against the real Run 2 artifacts when they are present
(they are committed, minus the large ``predictions.npz`` files) and skips
cleanly otherwise, so a fresh clone still passes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import report
from src.metrics import Predictions

ROOT = Path(__file__).resolve().parent.parent
RUN2 = ROOT / "sdgt_results_run_2" / "experiments" / "runs"

HORIZONS = (1, 6, 12, 24)
N_STATIONS, HORIZON = 3, 24
N_WINDOWS = 168 * 4          # four whole weeks, so the block bootstrap has blocks


# ------------------------------------------------------------------ fixture
def _write_run(root: Path, name: str, seed: int, error: float, *, curve: bool = True,
               best_epoch: int = 3, epochs_run: int = 10, train_loss: float = 0.2,
               val_loss: float = 0.25) -> Path:
    """One run directory whose forecast is truth + ``error`` everywhere.

    Constant error makes every metric exact: MAE == |error|, bias == error, and
    every bootstrap draw is identical so the paired interval collapses onto the
    point estimate.
    """
    run_dir = root / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True)

    rows = [
        {"model": name, "seed": seed, "horizon": h, "mae": abs(error), "rmse": abs(error),
         "wape": 10.0, "bias": error}
        for h in HORIZONS
    ]
    (run_dir / "metrics.json").write_text(json.dumps({
        "rows": rows, "best_epoch": best_epoch, "best_val_mae": 30.0,
        "epochs_run": epochs_run, "train_seconds": 120.0,
        "parameters": {"total": 1000},
    }), encoding="utf-8")

    truth = np.full((N_WINDOWS, N_STATIONS, HORIZON), 50.0, dtype=np.float32)
    # A little structure in the truth so quantile bins are not degenerate.
    truth += np.arange(N_WINDOWS, dtype=np.float32)[:, None, None] / 10.0
    Predictions(
        origins=np.arange(N_WINDOWS, dtype=np.int64),
        pred=truth + np.float32(error),
        truth=truth,
        mask=np.ones_like(truth, dtype=bool),
        model=name,
        seed=seed,
    ).save(run_dir / "predictions.npz")

    if curve:
        with open(run_dir / "curve.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "lr", "seconds"])
            writer.writeheader()
            for epoch in range(epochs_run):
                writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                                 "val_mae": 30.0, "lr": 3e-4, "seconds": 12.0 * epoch})

    (run_dir / "env.json").write_text(json.dumps({
        "git_revision": "abc123", "python": "3.11", "platform": "Linux",
        "timestamp_utc": "2026-09-09T00:00:00+00:00",
        "torch": {"version": "2.10", "cuda_version": "12.8", "device_name": "Tesla T4"},
        "packages": {"numpy": "2.0"},
    }), encoding="utf-8")
    return run_dir


@pytest.fixture
def runs(tmp_path) -> Path:
    root = tmp_path / "runs"
    _write_run(root, "persistence", 42, error=10.0, curve=False)
    _write_run(root, "lightgbm", 42, error=9.0, curve=False)
    _write_run(root, "t0_temporal_only", 42, error=8.0, train_loss=0.2, val_loss=0.22)
    _write_run(root, "t0_temporal_only", 43, error=-8.0, train_loss=0.2, val_loss=0.24)
    _write_run(root, "d1_wind_crossview", 42, error=12.0, best_epoch=0)
    _write_run(root, "d1_wind_crossview", 43, error=12.0)
    return root


# ---------------------------------------------------------------- discovery
def test_list_runs_parses_name_and_seed(runs):
    found = report.list_runs(runs)
    assert {(r.name, r.seed) for r in found} == {
        ("persistence", 42), ("lightgbm", 42),
        ("t0_temporal_only", 42), ("t0_temporal_only", 43),
        ("d1_wind_crossview", 42), ("d1_wind_crossview", 43),
    }


def test_list_runs_ignores_directories_without_a_seed_suffix(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "metrics.json").write_text("{}")
    assert report.list_runs(tmp_path) == []


def test_every_table_is_empty_not_broken_on_an_empty_root(tmp_path):
    """A notebook section with nothing to show must show nothing, not die."""
    assert report.metrics_table(tmp_path).empty
    assert report.training_table(tmp_path).empty
    assert report.skill_vs_persistence(tmp_path).empty
    assert report.paired_vs_reference(tmp_path).empty
    assert report.graph_summary(tmp_path).empty
    assert report.bias_by_horizon(tmp_path).empty
    assert report.top_decile_table(tmp_path).empty
    assert report.environment_table(tmp_path).empty
    assert report.headline(tmp_path).empty
    assert report.controls_table(tmp_path) is None
    assert report.load_diagnostics(tmp_path) is None


# ------------------------------------------------------------ configuration
def test_grid_design_covers_the_five_configs_in_order():
    frame = report.grid_design()
    assert list(frame.index) == report.GRID
    assert frame.loc["t0_temporal_only", "graph"] == "none"
    assert frame.loc["d1_wind_crossview", "fusion"] == "cross-view"


def test_flatten_config_walks_nested_keys():
    frame = report.flatten_config({"a": {"b": 1, "c": {"d": "x"}}, "e": 2})
    assert frame.loc["a.b", "value"] == 1
    assert frame.loc["a.c.d", "value"] == "x"
    assert frame.loc["e", "value"] == 2


def test_environment_table_reads_the_first_env_json(runs):
    frame = report.environment_table(runs)
    assert frame.loc["device", "value"] == "Tesla T4"
    assert frame.loc["git revision", "value"] == "abc123"
    assert frame.loc["numpy", "value"] == "2.0"


# ---------------------------------------------------------------- training
def test_training_table_computes_the_val_over_train_ratio(runs):
    frame = report.training_table(runs, names=["t0_temporal_only", "d1_wind_crossview"])
    assert frame.loc[("t0_temporal_only", 42), "val / train"] == pytest.approx(1.1, abs=0.01)
    assert frame.loc[("t0_temporal_only", 43), "val / train"] == pytest.approx(1.2, abs=0.01)
    assert frame.loc[("d1_wind_crossview", 42), "best epoch"] == 0
    assert frame.loc[("d1_wind_crossview", 42), "minutes"] == pytest.approx(2.0)


def test_training_table_excludes_models_without_curves(runs):
    """Baselines have no training loop; they must not appear as blank rows."""
    frame = report.training_table(runs)
    assert "persistence" not in frame.index.get_level_values("model")


def test_load_curve_is_numeric(runs):
    curve = report.load_curve(runs / "t0_temporal_only_seed42")
    assert len(curve) == 10
    assert curve["train_loss"].dtype.kind == "f"


# ---------------------------------------------------------------- metrics
def test_metrics_table_shows_mean_and_sd_only_for_multi_seed_models(runs):
    frame = report.metrics_table(runs, "mae")
    assert frame.loc["persistence", "h=1"] == "10.00"
    assert frame.loc["t0_temporal_only", "h=24"] == "8.00 ± 0.00"


def test_metrics_table_orders_baselines_before_the_grid(runs):
    frame = report.metrics_table(runs, "mae")
    names = list(frame.index)
    assert names.index("persistence") < names.index("t0_temporal_only")
    assert names.index("lightgbm") < names.index("d1_wind_crossview")


def test_skill_vs_persistence_is_exact_on_the_fixture(runs):
    skill = report.skill_vs_persistence(runs)
    assert skill.loc["t0_temporal_only", "h=1"] == pytest.approx(20.0)   # 1 - 8/10
    assert skill.loc["lightgbm", "h=6"] == pytest.approx(10.0)          # 1 - 9/10
    assert skill.loc["d1_wind_crossview", "h=24"] == pytest.approx(-20.0)  # 1 - 12/10
    assert "persistence" not in skill.index


def test_bias_by_horizon_keeps_the_sign(runs):
    """MAE hides direction; the whole point of this table is the sign."""
    bias = report.bias_by_horizon(runs)
    assert bias.loc["persistence", "h=1"] == pytest.approx(10.0)
    # Seeds 42 (+8) and 43 (-8) average to zero: a model that is wrong in both
    # directions has no systematic bias, which is exactly what this shows.
    assert bias.loc["t0_temporal_only", "h=24"] == pytest.approx(0.0)


# ---------------------------------------------------- paired comparison
def test_paired_vs_reference_is_robust_when_every_seed_agrees(runs):
    """t0 (|err| 8) beats lightgbm (|err| 9) on every row for every seed."""
    frame = report.paired_vs_reference(runs, reference="lightgbm", n_boot=50)
    row = frame.loc[("t0_temporal_only", 24)]
    assert row["diff (ug/m3)"] == pytest.approx(1.0)
    assert row["verdict"] == "robust"
    assert row["CIs excl. 0"] == "2/2"
    assert frame.attrs["reference"] == "lightgbm"


def test_paired_vs_reference_reports_a_loss_with_negative_sign(runs):
    frame = report.paired_vs_reference(runs, reference="lightgbm", n_boot=50)
    row = frame.loc[("d1_wind_crossview", 1)]
    assert row["diff (ug/m3)"] == pytest.approx(-3.0)
    assert row["verdict"] == "robust"


def test_paired_vs_reference_is_empty_without_the_reference(runs):
    assert report.paired_vs_reference(runs, reference="no_such_model").empty


# ------------------------------------------------------- error analysis
def test_top_decile_table_averages_over_seeds(runs):
    frame = report.top_decile_table(runs)
    assert frame.loc["persistence", "top-decile MAE h=24"] == pytest.approx(10.0)
    assert frame.loc["t0_temporal_only", "top-decile MAE h=24"] == pytest.approx(8.0)
    assert frame.loc["t0_temporal_only", "top-decile bias h=24"] == pytest.approx(0.0)


def test_error_by_level_bins_the_observed_axis(runs):
    frame = report.error_by_level(runs, "t0_temporal_only", horizon=24, n_bins=4)
    assert len(frame) == 4
    np.testing.assert_allclose(frame["MAE"].to_numpy(), 8.0)
    assert frame["n"].sum() == N_WINDOWS * N_STATIONS
    assert frame.attrs["model"] == "t0_temporal_only"


def test_worst_weeks_labels_by_first_forecast_hour(runs):
    class _Dataset:
        timestamps = np.arange(
            np.datetime64("2016-03-01T00", "h"),
            np.datetime64("2016-03-01T00", "h") + np.timedelta64(N_WINDOWS + 48, "h"),
            np.timedelta64(1, "h"),
        )

    frame = report.worst_weeks(_Dataset(), runs, "t0_temporal_only", n=2)
    assert len(frame) == 2
    # Origin 0 -> first valid hour is 01:00 on 1 March, still 2016-03-01.
    assert "2016-03-01" in frame.index


# ----------------------------------------------------------- diagnostics
def _diagnostics(n=4, L=6, N=3, K=6, Np=2, d=8) -> dict:
    rng = np.random.default_rng(0)
    adjacency = rng.random((n, L, N, N)); adjacency /= adjacency.sum(-1, keepdims=True)
    attention = rng.random((n, L, N, N, K + 1)); attention /= attention.sum((-1, -2), keepdims=True)
    return {
        "adjacency": adjacency,
        "attention_by_lag": attention,
        "wind_speed": np.array([[0.5] * L, [0.8] * L, [4.0] * L, [6.0] * L]),
        "graph_calm": np.zeros((n, L, N, 1), dtype=bool),
        "gate": np.full((n, N, Np, d), 0.5),
    }


def test_attention_by_lag_shares_sum_to_100_and_report_the_peak():
    diag = _diagnostics()
    # Pile all cross-station attention onto lag 2.
    diag["attention_by_lag"][..., :] = 0.0
    diag["attention_by_lag"][..., 2] = 1.0
    frame = report.attention_by_lag(diag)
    assert frame["attention share"].sum() == pytest.approx(100.0, abs=0.2)
    assert frame.attrs["peak_lag"] == 2


def test_attention_by_lag_excludes_self_attention():
    """Self-attention at lag 0 must not masquerade as cross-station transport."""
    diag = _diagnostics()
    att = np.zeros_like(diag["attention_by_lag"])
    for i in range(att.shape[2]):
        att[:, :, i, i, 0] = 1.0            # self only, lag 0
    att[:, :, 0, 1, 3] = 1.0                # one cross edge at lag 3
    diag["attention_by_lag"] = att
    frame = report.attention_by_lag(diag)
    assert frame.attrs["peak_lag"] == 3
    assert frame.loc[0, "attention share"] == pytest.approx(0.0)


def test_gate_summary_flags_a_gate_stuck_at_initialisation():
    frame = report.gate_summary(_diagnostics())
    assert frame.loc["overall", "mean"] == pytest.approx(0.5)
    assert frame.loc["overall", "std"] == pytest.approx(0.0)
    assert "patch 0 (oldest)" in frame.index
    assert "patch 1 (newest)" in frame.index


def test_adjacency_by_regime_splits_on_wind_speed():
    frame = report.adjacency_by_regime(_diagnostics(), calm_threshold=1.5)
    assert len(frame) == 2
    assert frame["hours"].sum() == 4 * 6
    assert frame.iloc[0]["hours"] == 12       # two calm windows x 6 hours


def test_diagnostic_tables_are_empty_without_the_arrays():
    assert report.attention_by_lag({}).empty
    assert report.gate_summary({}).empty
    assert report.adjacency_by_regime({}).empty


# --------------------------------------------------------------- controls
def test_controls_table_puts_references_first_and_counts_detections(tmp_path):
    root = tmp_path / "runs"
    for seed, detected in ((42, True), (43, False)):
        run = _write_run(root, "d1_wind_crossview", seed, error=1.0)
        (run / "controls.json").write_text(json.dumps({"controls": [
            {"control": "wind_reversal", "is_reference": False,
             "horizons": {"24": {"damage": 0.1, "detected": detected}}},
            {"control": "zero_correction", "is_reference": True,
             "horizons": {"24": {"damage": 4.0, "detected": True}}},
        ]}), encoding="utf-8")

    frame = report.controls_table(root)
    assert frame is not None
    assert frame.index[0][0] == "zero_correction", "the reference must be read first"
    assert frame.loc[("wind_reversal", 24), "detected"] == "1/2"
    assert frame.loc[("zero_correction", 24), "damage"] == pytest.approx(4.0)


# ---------------------------------------------------------- real Run 2 data
def _run2_available(need_predictions: bool = False) -> bool:
    if not RUN2.exists():
        return False
    if need_predictions and not (RUN2 / "t0_temporal_only_seed42" / "predictions.npz").exists():
        return False
    return True


@pytest.mark.skipif(not _run2_available(), reason="Run 2 metrics not present")
def test_run2_metrics_table_matches_the_findings_document():
    """The numbers in SDGT-CMA_Run2_Findings.md are these, to two decimals."""
    frame = report.metrics_table(RUN2, "mae", seeds=(42, 43, 44))
    assert frame.loc["lightgbm", "h=24"] == "50.82"
    assert frame.loc["persistence", "h=1"] == "10.30"
    assert frame.loc["t0_temporal_only", "h=1"].startswith("10.51")


@pytest.mark.skipif(not _run2_available(), reason="Run 2 metrics not present")
def test_run2_training_table_reproduces_the_convergence_gate():
    frame = report.training_table(RUN2)
    assert frame["val / train"].between(1.0, 1.7).all(), "Run 2 ratios were 1.13-1.57"


@pytest.mark.skipif(not _run2_available(), reason="Run 2 metrics not present")
def test_run2_graph_summary_shows_the_physics_moved():
    frame = report.graph_summary(RUN2)
    row = frame.loc[("d1_wind_crossview", 42)]
    assert row["prior strength (init 1.0)"] == pytest.approx(0.61, abs=0.01)
    assert row["mean transport lag (h)"] == pytest.approx(2.91, abs=0.01)


@pytest.mark.skipif(not _run2_available(need_predictions=True), reason="Run 2 predictions not present")
def test_run2_paired_comparison_against_lightgbm_runs():
    frame = report.paired_vs_reference(RUN2, "lightgbm", seeds=(42, 43, 44), n_boot=50)
    assert set(frame.index.get_level_values("candidate")) == set(report.GRID)
    assert (frame["diff (ug/m3)"] < 0).all(), "LightGBM won every cell in Run 2"
