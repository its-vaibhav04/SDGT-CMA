"""Gates on the negative-control suite.

A broken control is worse than no control, because it produces a number that
looks like evidence. The failure modes guarded here are all silent ones:

- a perturbation that does not actually perturb (reversing standardised wind by
  negating the stored value, which reverses the wind *and* shifts its mean);
- a perturbation that leaks into the next control because a swapped module was
  never put back;
- a control that changes which rows are scored, breaking the pairing that the
  bootstrap depends on and quietly invalidating every interval.

The arithmetic tests run anywhere. The end-to-end tests need a built Beijing
dataset and skip without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from src import bootstrap, controls
from src.data import contract
from src.data.transforms import ColumnSpec, Scaler
from src.data.windows import Batch, build_batchers
from src.models.sdgt import ModelConfig, build_model

ROOT = Path(__file__).resolve().parent.parent
CITY = "beijing"
N_STATIONS, N_FEATURES, LOOKBACK, HORIZON = 12, 27, 48, 24
COORDS = np.array([[39.85 + 0.04 * i, 116.05 + 0.05 * i] for i in range(N_STATIONS)])


@pytest.fixture(autouse=True)
def deterministic_init():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def dataset():
    try:
        return contract.load(CITY)
    except FileNotFoundError as error:
        pytest.skip(str(error))


def make_model(**overrides):
    settings = dict(
        n_features=N_FEATURES,
        n_stations=N_STATIONS,
        coords=COORDS,
        lookback=LOOKBACK,
        horizon=HORIZON,
        d_model=32,
        n_heads=4,
        temporal_layers=2,
        graph="dynamic",
        spatial="wind_gat",
        fusion="cross_view",
    )
    settings.update(overrides)
    return build_model(ModelConfig(**settings))


def make_batch(batch_size: int = 3, seed: int = 0) -> Batch:
    generator = torch.Generator().manual_seed(seed)
    return Batch(
        features=torch.randn(batch_size, LOOKBACK, N_STATIONS, N_FEATURES, generator=generator),
        wind_uv=torch.randn(batch_size, LOOKBACK, N_STATIONS, 2, generator=generator) * 3.0,
        target=torch.randn(batch_size, N_STATIONS, HORIZON, generator=generator),
        target_mask=torch.ones(batch_size, N_STATIONS, HORIZON, dtype=torch.bool),
        origins=torch.arange(batch_size, dtype=torch.int64),
        anchor=torch.randn(batch_size, N_STATIONS, generator=generator),
    )


# --------------------------------------------------------------- derangement
def test_derangement_is_a_permutation_with_no_fixed_point():
    for seed in range(10):
        order = controls.derangement(12, seed)
        assert sorted(order.tolist()) == list(range(12)), "not a permutation"
        assert not (order == np.arange(12)).any(), "a station was left mapped to itself"


def test_derangement_is_reproducible_from_its_seed():
    assert np.array_equal(controls.derangement(12, 7), controls.derangement(12, 7))
    assert not np.array_equal(controls.derangement(12, 7), controls.derangement(12, 8))


def test_derangement_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        controls.derangement(1)


# ---------------------------------------------------------- graph rewriting
def test_edge_permutation_preserves_row_sums():
    """Weights must still form a distribution over sources after relabelling.

    If they did not, the control would be testing "a malformed graph" rather
    than "the right weights pointed at the wrong stations".
    """
    permute = controls._permute_edges(N_STATIONS, seed=3)
    adjacency = torch.rand(2, 5, N_STATIONS, N_STATIONS)
    adjacency = adjacency / adjacency.sum(-1, keepdim=True)
    lag = torch.rand(2, 5, N_STATIONS, N_STATIONS) * 6

    new_adjacency, new_lag = permute(adjacency, lag)

    torch.testing.assert_close(
        new_adjacency.sum(-1), adjacency.sum(-1), msg="row sums changed"
    )
    assert new_adjacency.shape == adjacency.shape
    assert new_lag.shape == lag.shape
    assert not torch.allclose(new_adjacency, adjacency), "permutation changed nothing"


def test_edge_permutation_moves_the_self_edge_off_the_diagonal():
    """The point of a derangement: no target keeps its own station as a source."""
    permute = controls._permute_edges(N_STATIONS, seed=1)
    adjacency = torch.eye(N_STATIONS).expand(1, 1, N_STATIONS, N_STATIONS).contiguous()
    new_adjacency, _ = permute(adjacency, torch.zeros_like(adjacency))
    diagonal = torch.diagonal(new_adjacency[0, 0])
    assert float(diagonal.sum()) == 0.0, "some station still attends to itself"


def test_identity_adjacency_removes_every_cross_station_edge():
    identity = controls._identity_adjacency(N_STATIONS)
    adjacency = torch.rand(2, 5, N_STATIONS, N_STATIONS)
    lag = torch.rand(2, 5, N_STATIONS, N_STATIONS) * 6

    new_adjacency, new_lag = identity(adjacency, lag)

    expected = torch.eye(N_STATIONS).expand_as(adjacency)
    torch.testing.assert_close(new_adjacency, expected)
    assert float(new_lag.abs().sum()) == 0.0, "identity must carry no transport lag"


# ------------------------------------------------------------ wind reversal
def test_wind_reversal_touches_only_the_graph_input():
    """The temporal branch must keep seeing the true wind.

    That separation is what makes this a test of the graph rather than of the
    model as a whole; wind_reversal_full is the one that moves both.
    """
    batch = make_batch()
    reversed_batch = controls._reverse_wind_uv(batch)

    torch.testing.assert_close(reversed_batch.wind_uv, -batch.wind_uv)
    torch.testing.assert_close(reversed_batch.features, batch.features)
    torch.testing.assert_close(reversed_batch.anchor, batch.anchor)
    torch.testing.assert_close(reversed_batch.target, batch.target)
    assert torch.equal(reversed_batch.target_mask, batch.target_mask)


@dataclass
class _StubDataset:
    """Just enough of ProcessedDataset for the reflection arithmetic."""

    scaler: Scaler
    feature_names: list[str]

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)


def test_full_wind_reversal_reflects_raw_wind_not_the_stored_value():
    """z -> -z would reverse the wind and move its mean at the same time.

    With a non-zero column mean the two differ by 2*mu/sigma, which is exactly
    the size of confound that would make a null control look like a real effect
    (or hide one). This pins the formula against a round trip through the real
    Scaler.
    """
    names = ["wind_u", "wind_v"]
    raw = np.stack(
        [
            np.linspace(-8.0, 6.0, 40),      # deliberately non-zero mean
            np.linspace(-3.0, 9.0, 40),
        ],
        axis=-1,
    ).reshape(40, 1, 2)

    scaler = Scaler.fit(raw, names, {}, slice(0, 40))
    stub = _StubDataset(scaler=scaler, feature_names=names)

    # Ground truth: scale the negated raw values.
    expected = scaler.transform(-raw)

    scaled = scaler.transform(raw)
    batch = Batch(
        features=torch.from_numpy(scaled).unsqueeze(0).float(),   # [1, T, 1, 2]
        wind_uv=torch.zeros(1, 40, 1, 2),
        target=torch.zeros(1, 1, 1),
        target_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        origins=torch.zeros(1, dtype=torch.int64),
        anchor=torch.zeros(1, 1),
    )
    flipped = controls._reverse_wind_everywhere(stub)(batch)

    np.testing.assert_allclose(
        flipped.features[0].numpy(), expected, rtol=1e-5, atol=1e-5,
        err_msg="reflected wind does not match a real re-scaling of -raw",
    )
    # And the naive version really is different, so the test has teeth.
    assert not np.allclose(-scaled, expected, atol=1e-3)


def test_full_wind_reversal_rejects_a_log_scaled_wind_column():
    """The formula assumes plain standardisation; say so rather than be wrong."""
    scaler = Scaler([ColumnSpec("wind_u", "log_standard", 1.0, 2.0),
                     ColumnSpec("wind_v", "standard", 0.0, 1.0)])
    stub = _StubDataset(scaler=scaler, feature_names=["wind_u", "wind_v"])
    with pytest.raises(ValueError, match="log_standard"):
        controls._reverse_wind_everywhere(stub)


# ----------------------------------------------------------- history shuffle
def test_history_shuffle_scrambles_time_but_keeps_the_anchor():
    """The anchor is the true y_t and must survive, or this stops being a
    measure of the learned correction and becomes a measure of persistence."""
    batch = make_batch()
    shuffled = controls._shuffle_history(seed=0)(batch)

    torch.testing.assert_close(shuffled.anchor, batch.anchor)
    assert not torch.allclose(shuffled.features, batch.features), "nothing was shuffled"
    # A permutation of the time axis, so the multiset of hours is unchanged.
    torch.testing.assert_close(
        shuffled.features.sort(dim=1).values, batch.features.sort(dim=1).values
    )


# --------------------------------------------------------- module swap safety
def test_applied_restores_the_model_afterwards():
    model = make_model()
    graph, head = model.graph, model.head

    control = controls.Control(
        name="x", description="", reads="",
        graph_fn=controls._identity_adjacency(N_STATIONS), zero_head=True,
    )
    with controls._applied(model, control):
        assert model.graph is not graph
        assert model.head is not head
    assert model.graph is graph, "graph was not restored"
    assert model.head is head, "head was not restored"


def test_applied_restores_the_model_even_when_the_body_raises():
    """A leaked swap would silently contaminate every later control."""
    model = make_model()
    graph, head = model.graph, model.head
    control = controls.Control(
        name="x", description="", reads="",
        graph_fn=controls._identity_adjacency(N_STATIONS), zero_head=True,
    )
    with pytest.raises(RuntimeError):
        with controls._applied(model, control):
            raise RuntimeError("boom")
    assert model.graph is graph and model.head is head


def test_applied_is_a_no_op_for_the_unperturbed_baseline():
    model = make_model()
    graph, head = model.graph, model.head
    with controls._applied(model, None):
        assert model.graph is graph and model.head is head


def test_graph_controls_refuse_a_model_without_a_graph():
    model = make_model(graph="none", spatial="none", fusion="none")
    suite = _suite()
    graph_controls = [c for c in suite if c.requires_graph]
    assert graph_controls, "the suite should contain graph-only controls"
    for control in graph_controls:
        assert not control.applies_to(model)
    for control in suite:
        if not control.requires_graph:
            assert control.applies_to(model)


def _suite():
    """The control suite, built against a stub scaler so it needs no dataset."""
    names = ["wind_u", "wind_v"]
    raw = np.random.default_rng(0).normal(size=(50, 1, 2))
    scaler = Scaler.fit(raw, names, {}, slice(0, 50))
    return controls.build_controls(_StubDataset(scaler, names), N_STATIONS, seed=0)


def test_every_control_declares_what_it_reads():
    for control in _suite():
        assert control.reads, f"{control.name} has no stated interpretation"
        assert control.description


def test_the_suite_contains_a_power_reference():
    """Without one, "no effect" is uninterpretable -- it could just mean the
    measurement has no power. This is the whole reason the suite is honest."""
    references = [c.name for c in _suite() if c.is_reference]
    assert "zero_correction" in references
    assert "history_shuffle" in references


# ------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def trained(dataset):
    """A small model on the real data. Untrained weights are fine here: these
    tests check the control *machinery*, not what a control concludes."""
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(
            n_features=dataset.n_features,
            n_stations=dataset.n_stations,
            coords=dataset.coords,
            lookback=LOOKBACK,
            horizon=HORIZON,
            d_model=16,
            n_heads=2,
            temporal_layers=1,
            spatial_layers=1,
            graph="dynamic",
            spatial="wind_gat",
            fusion="cross_view",
        )
    )
    model.eval()
    batchers = build_batchers(
        dataset, lookback=LOOKBACK, horizon=HORIZON, batch_size=8, seed=0, device="cpu"
    )
    test = batchers["test"]
    test.origins = test.origins[:24]     # a couple of batches is plenty
    return model, test


def test_controls_stay_row_aligned_so_the_pairing_holds(dataset, trained):
    """Every control must score the identical windows and target mask.

    paired_interval refuses mismatched rows, so this is the gate that keeps the
    bootstrap intervals meaningful rather than merely computable.
    """
    model, test = trained
    device = torch.device("cpu")
    baseline = controls.evaluate_control(
        model, test, dataset, None, device=device, model_name="d1", seed=0
    )
    for control in controls.build_controls(dataset, dataset.n_stations, seed=0):
        if not control.applies_to(model):
            continue
        perturbed = controls.evaluate_control(
            model, test, dataset, control, device=device, model_name="d1", seed=0
        )
        assert np.array_equal(perturbed.origins, baseline.origins), control.name
        assert np.array_equal(perturbed.mask, baseline.mask), control.name
        np.testing.assert_allclose(perturbed.truth, baseline.truth, rtol=0, atol=0)
        # The bootstrap must accept the pair without complaint.
        bootstrap.paired_interval(baseline, perturbed, horizon=1, n_boot=20)


def test_zero_correction_reproduces_persistence_exactly(dataset, trained):
    """With the anchor on, a zeroed head *is* persistence. If this drifts, the
    reference denominator in the report is measuring something else."""
    model, test = trained
    device = torch.device("cpu")
    control = next(
        c for c in controls.build_controls(dataset, dataset.n_stations)
        if c.name == "zero_correction"
    )
    result = controls.evaluate_control(
        model, test, dataset, control, device=device, model_name="d1", seed=0
    )
    # The anchor is the standardised target at the origin hour; inverted it is
    # y_t, repeated across every horizon.
    expected = dataset.target_scaler.inverse(
        dataset.target_scaled[test.origins.numpy()]
    )
    for h in range(result.pred.shape[2]):
        np.testing.assert_allclose(result.pred[:, :, h], expected, rtol=1e-4, atol=1e-3)


def test_controls_do_not_mutate_the_trained_weights(dataset, trained):
    """Controls perturb inputs and module wiring, never parameters."""
    model, test = trained
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    device = torch.device("cpu")
    for control in controls.build_controls(dataset, dataset.n_stations, seed=0):
        if control.applies_to(model):
            controls.evaluate_control(
                model, test, dataset, control, device=device, model_name="d1", seed=0
            )
    after = model.state_dict()
    for key, value in before.items():
        torch.testing.assert_close(after[key], value, msg=f"{key} changed")


def test_a_graph_perturbation_actually_changes_the_forecast(dataset, trained):
    """The machinery must be capable of moving the output at all.

    This is deliberately asserted on an *untrained* model, where the spatial
    branch is guaranteed to be wired in. If it fails, a null result from the
    real controls would be a plumbing bug rather than a finding.
    """
    model, test = trained
    device = torch.device("cpu")
    baseline = controls.evaluate_control(
        model, test, dataset, None, device=device, model_name="d1", seed=0
    )
    identity = next(
        c for c in controls.build_controls(dataset, dataset.n_stations)
        if c.name == "identity_adjacency"
    )
    perturbed = controls.evaluate_control(
        model, test, dataset, identity, device=device, model_name="d1", seed=0
    )
    assert not np.allclose(perturbed.pred, baseline.pred), (
        "cutting every cross-station edge did not change the forecast -- the "
        "spatial branch is not reaching the output"
    )


def test_station_occlusion_reports_one_row_per_station(dataset, trained):
    model, test = trained
    rows = controls.station_influence(
        model, test, dataset, device=torch.device("cpu"), horizons=(1, 24)
    )
    assert len(rows) == dataset.n_stations
    assert {row["station"] for row in rows} == set(dataset.stations)
    for row in rows:
        assert "delta_h1" in row and "delta_h24" in row


def test_wind_reversal_is_exactly_inert_on_a_static_graph(dataset):
    """A distance graph ignores wind, so reversing it must change nothing at all.

    This is the control on the control. It pins the isolation claim that makes
    ``wind_reversal`` a test of the *graph*: wind_uv reaches the graph and
    nothing else, so a graph that does not read wind must produce a bitwise
    identical forecast. If someone later routes wind_uv into the temporal branch,
    this fails immediately rather than quietly turning every D1 control into a
    whole-model measurement.
    """
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(
            n_features=dataset.n_features,
            n_stations=dataset.n_stations,
            coords=dataset.coords,
            lookback=LOOKBACK,
            horizon=HORIZON,
            d_model=16,
            n_heads=2,
            temporal_layers=1,
            spatial_layers=1,
            graph="static",
            spatial="gcn",
            fusion="concat",
        )
    )
    model.eval()
    batchers = build_batchers(
        dataset, lookback=LOOKBACK, horizon=HORIZON, batch_size=8, seed=0, device="cpu"
    )
    test = batchers["test"]
    test.origins = test.origins[:16]

    device = torch.device("cpu")
    baseline = controls.evaluate_control(
        model, test, dataset, None, device=device, model_name="s0", seed=0
    )
    control = next(
        c for c in controls.build_controls(dataset, dataset.n_stations)
        if c.name == "wind_reversal"
    )
    perturbed = controls.evaluate_control(
        model, test, dataset, control, device=device, model_name="s0", seed=0
    )
    np.testing.assert_array_equal(
        perturbed.pred, baseline.pred,
        err_msg="wind reversal moved a static-graph model, so wind_uv is "
                "reaching something other than the graph",
    )


# ------------------------------------------------- the report refuses nonsense
def _load_script():
    """Import scripts/run_controls.py as a module."""
    import importlib.util

    path = ROOT / "scripts" / "run_controls.py"
    spec = importlib.util.spec_from_file_location("run_controls_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(name, damage, detected, is_reference=False):
    return {
        "control": name,
        "reads": "",
        "is_reference": is_reference,
        "horizons": {
            24: {
                "damage": damage, "ci_low": damage - 1.0, "ci_high": damage + 1.0,
                "detected": detected, "mae_perturbed": 0.0, "mae_unperturbed": 0.0,
            }
        },
    }


def test_report_refuses_to_interpret_an_unsound_denominator(capsys):
    """If the learned correction is worth nothing, every ratio against it is
    noise over noise. Printing "0.8 % of it" would dress that up as a finding."""
    script = _load_script()
    script._interpret(
        [
            _record("wind_reversal", 0.03, False),
            _record("zero_correction", -0.17, False, is_reference=True),
        ]
    )
    out = capsys.readouterr().out
    assert "CANNOT INTERPRET" in out
    assert "% of it" not in out, "percentages printed against an unsound denominator"


def test_report_gives_percentages_when_the_denominator_is_sound(capsys):
    script = _load_script()
    script._interpret(
        [
            _record("wind_reversal", 0.02, False),
            _record("identity_adjacency", 1.00, True),
            _record("zero_correction", 4.00, True, is_reference=True),
        ]
    )
    out = capsys.readouterr().out
    assert "CANNOT INTERPRET" not in out
    assert "25.0 % of it" in out, out
    assert "no effect" in out and "degrades" in out


def test_report_flags_a_control_that_improves_the_forecast(capsys):
    """A perturbation that significantly *helps* is a red flag, not a null."""
    script = _load_script()
    script._interpret(
        [
            _record("wind_reversal", -0.80, True),
            _record("zero_correction", 4.00, True, is_reference=True),
        ]
    )
    assert "IMPROVES" in capsys.readouterr().out
