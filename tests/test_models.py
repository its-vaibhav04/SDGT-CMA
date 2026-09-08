"""Model gates: wiring, capacity matching, and the checks that catch dead branches.

The failures these guard against are all quiet ones. A spatial branch that
ignores the graph still trains. A fusion layer that drops one input still
converges. A model that cannot overfit twenty samples looks like a
generalisation problem rather than the wiring bug it is.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.fusion.cross_view import ConcatFusion, CrossViewFusion
from src.models.head import masked_huber_loss, masked_l1_loss
from src.models.sdgt import ModelConfig, build_model
from src.models.temporal.patch_transformer import (
    MultivariatePatchTransformer,
    n_patches,
    patch_indices,
    pool_to_patches,
)

N_STATIONS, N_FEATURES, LOOKBACK, HORIZON = 12, 27, 48, 24
COORDS = np.array([[39.85 + 0.04 * i, 116.05 + 0.05 * i] for i in range(N_STATIONS)])

CONFIGS = {
    "T0": dict(graph="none", spatial="none", fusion="none"),
    "S0": dict(graph="static", spatial="gcn", fusion="concat"),
    "D0": dict(graph="dynamic", spatial="wind_gat", fusion="concat"),
    "S1": dict(graph="static", spatial="gcn", fusion="cross_view"),
    "D1": dict(graph="dynamic", spatial="wind_gat", fusion="cross_view"),
}


@pytest.fixture(autouse=True)
def deterministic_init():
    """Seed before every test so weight initialisation does not depend on
    execution order.

    Without this, a test that builds a model inherits whatever RNG state the
    previous test happened to leave behind, and assertions about an untrained
    model's behaviour pass or fail depending on which tests ran first.
    """
    torch.manual_seed(0)


def make_config(**overrides) -> ModelConfig:
    settings = dict(
        n_features=N_FEATURES,
        n_stations=N_STATIONS,
        coords=COORDS,
        lookback=LOOKBACK,
        horizon=HORIZON,
        d_model=32,
        n_heads=4,
        temporal_layers=2,
    )
    settings.update(overrides)
    return ModelConfig(**settings)


def make_batch(batch_size: int = 4, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(batch_size, LOOKBACK, N_STATIONS, N_FEATURES, generator=generator)
    wind = torch.randn(batch_size, LOOKBACK, N_STATIONS, 2, generator=generator) * 3.0
    return features, wind


# ------------------------------------------------------------------- patching
def test_patch_count_matches_the_unfold_length():
    for lookback, patch, stride in ((48, 8, 8), (168, 24, 24), (96, 12, 12), (24, 4, 4)):
        indices = patch_indices(lookback, patch, stride)
        assert indices.shape == (n_patches(lookback, patch, stride), patch)
        assert int(indices.max()) < lookback


def test_patch_indices_cover_the_window_without_gaps_when_non_overlapping():
    indices = patch_indices(48, 8, 8)
    flat = indices.flatten().tolist()
    assert flat == list(range(48))


def test_both_branches_pool_on_identical_patch_boundaries():
    """If these ever diverge, cross-view attention aligns unrelated tokens."""
    temporal = MultivariatePatchTransformer(
        d_model=16, lookback=48, patch_length=8, stride=8, n_layers=1, n_heads=2
    )
    hourly = torch.randn(2, 48, N_STATIONS, 16)
    pooled = pool_to_patches(hourly, patch_indices(48, 8, 8))

    assert pooled.shape == (2, N_STATIONS, temporal.n_patches, 16)
    torch.testing.assert_close(temporal.indices, patch_indices(48, 8, 8))


def test_temporal_branch_shares_weights_across_stations():
    """Same input at two stations must give the same output; different inputs must not."""
    branch = MultivariatePatchTransformer(
        d_model=16, lookback=48, patch_length=8, stride=8, n_layers=1, n_heads=2
    )
    branch.eval()
    hidden = torch.randn(1, 48, 3, 16)
    hidden[:, :, 1] = hidden[:, :, 0]              # station 1 duplicates station 0

    with torch.no_grad():
        out = branch(hidden)

    torch.testing.assert_close(out[0, 0], out[0, 1], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(out[0, 0], out[0, 2], atol=1e-3)


# --------------------------------------------------------------------- wiring
@pytest.mark.parametrize("name", list(CONFIGS))
def test_every_configuration_produces_the_right_shape(name):
    model = build_model(make_config(**CONFIGS[name]))
    features, wind = make_batch()
    prediction, _ = model(features, wind)
    assert prediction.shape == (4, N_STATIONS, HORIZON)
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("name", list(CONFIGS))
def test_gradients_reach_every_parameter(name):
    """A disconnected module trains silently and contributes nothing."""
    model = build_model(make_config(**CONFIGS[name]))
    features, wind = make_batch()
    prediction, _ = model(features, wind)
    prediction.sum().backward()

    missing = [
        param_name
        for param_name, param in model.named_parameters()
        if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all())
    ]
    assert not missing, f"{name}: no finite gradient for {missing}"


def test_mismatched_graph_and_spatial_are_rejected():
    with pytest.raises(ValueError, match="both be"):
        make_config(graph="dynamic", spatial="none", fusion="none")
    with pytest.raises(ValueError, match="fusion requires"):
        make_config(graph="none", spatial="none", fusion="concat")


# ----------------------------------------------------------- capacity matching
def test_concat_and_cross_view_have_comparable_capacity():
    """The fusion comparison only isolates mechanism if width is held roughly equal."""
    concat = build_model(make_config(**CONFIGS["D0"])).parameter_counts()
    cross = build_model(make_config(**CONFIGS["D1"])).parameter_counts()
    ratio = cross["total"] / concat["total"]
    assert 0.9 < ratio < 1.1, f"total parameter ratio {ratio:.3f} is outside 10%"


def test_only_the_spatial_and_fusion_blocks_differ_across_the_grid():
    counts = {name: build_model(make_config(**cfg)).parameter_counts()
              for name, cfg in CONFIGS.items()}
    temporal = {c["temporal"] for c in counts.values()}
    embedding = {c["embedding"] for c in counts.values()}
    head = {c["head"] for c in counts.values()}
    assert len(temporal) == 1, "the temporal branch must be identical across the grid"
    assert len(embedding) == 1 and len(head) == 1


# ------------------------------------------------------- the graph is not dead
def test_perturbing_an_upwind_station_propagates_downwind():
    """Proof the spatial branch aggregates rather than passing through.

    Under a strong easterly, station 0 is the most upwind node. Changing its
    history must move its downwind neighbours' forecasts, and the effect must
    fade with distance -- two GAT layers over a top-k sparse graph reach a
    bounded number of hops, so the far end of the line is legitimately outside
    the receptive field.
    """
    model = build_model(make_config(**CONFIGS["D1"]))
    model.eval()
    features, _ = make_batch(batch_size=1)
    wind = torch.zeros(1, LOOKBACK, N_STATIONS, 2)
    wind[..., 0] = 6.0                                  # blowing east

    with torch.no_grad():
        before, _ = model(features, wind)
        perturbed = features.clone()
        perturbed[:, :, 0, :] += 5.0
        after, _ = model(perturbed, wind)

    change = (after - before).abs().amax(dim=-1)[0]     # [N]

    assert float(change[0]) > 1e-4, "the perturbed station itself must react"
    assert float(change[1]) > 1e-4, "the immediate downwind neighbour must react"
    assert float(change[1]) > float(change[4]), "influence must fade with distance"


def test_influence_does_not_travel_upwind():
    """The direction asymmetry, checked end to end through the whole model.

    Perturbing the most *downwind* station must not move the most upwind one.
    This is the model-level counterpart of the graph transpose test: it would
    catch a reversal introduced anywhere between the adjacency and the head.
    """
    model = build_model(make_config(**CONFIGS["D1"]))
    model.eval()
    features, _ = make_batch(batch_size=1)
    wind = torch.zeros(1, LOOKBACK, N_STATIONS, 2)
    wind[..., 0] = 6.0                                  # blowing east

    with torch.no_grad():
        before, _ = model(features, wind)

        downstream = features.clone()
        downstream[:, :, -1, :] += 5.0                  # perturb the far downwind end
        after_downstream, _ = model(downstream, wind)

        upstream = features.clone()
        upstream[:, :, 0, :] += 5.0                     # perturb the far upwind end
        after_upstream, _ = model(upstream, wind)

    upwind_effect = float((after_downstream[0, 0] - before[0, 0]).abs().max())
    downwind_effect = float((after_upstream[0, 1] - before[0, 1]).abs().max())

    assert upwind_effect < 1e-6, (
        f"a downwind source influenced an upwind target by {upwind_effect:.2e}"
    )
    assert downwind_effect > 1e-4, "an upwind source must influence downwind targets"


def test_temporal_only_model_ignores_wind():
    """T0 has no graph, so the wind input must not reach the prediction at all."""
    model = build_model(make_config(**CONFIGS["T0"]))
    model.eval()
    features, wind = make_batch(batch_size=1)
    with torch.no_grad():
        a, _ = model(features, wind)
        b, _ = model(features, wind * -3.0)
    torch.testing.assert_close(a, b)


# ------------------------------------------------------------------- fusion
def test_fusion_accepts_either_spatial_branch_interchangeably():
    """The fusion layer must not care which spatial branch produced its input."""
    fusion = CrossViewFusion(d_model=32, n_heads=4)
    fusion.eval()
    spatial = torch.randn(2, N_STATIONS, 6, 32)
    temporal = torch.randn(2, N_STATIONS, 6, 32)
    fused, diagnostics = fusion(spatial, temporal)
    assert fused.shape == spatial.shape
    assert diagnostics["gate"].shape == spatial.shape


def test_gate_algebra_at_its_limits():
    """Test the arithmetic by forcing the gate, not by hoping an untrained
    sigmoid behaves semantically.

    The original roadmap proposed zeroing one branch and expecting an untrained
    gate to shift toward the other. An untrained gate has no semantics, so that
    test could only ever pass or fail by luck. This checks the algebra instead.
    """
    fusion = CrossViewFusion(d_model=16, n_heads=2, dropout=0.0)
    fusion.eval()
    spatial = torch.randn(1, 3, 4, 16)
    temporal = torch.randn(1, 3, 4, 16)

    with torch.no_grad():
        # Force gate -> 1: output must be LayerNorm of the temporal-query context.
        fusion.gate[0].weight.zero_()
        fusion.gate[0].bias.fill_(20.0)
        fused_one, diagnostics = fusion(spatial, temporal)
        assert float(diagnostics["gate"].min()) > 0.999

        # Force gate -> 0: output must be LayerNorm of the mirrored context.
        fusion.gate[0].bias.fill_(-20.0)
        fused_zero, diagnostics = fusion(spatial, temporal)
        assert float(diagnostics["gate"].max()) < 0.001

    assert not torch.allclose(fused_one, fused_zero, atol=1e-3), (
        "the two gate limits must select genuinely different contexts"
    )


def test_concat_fusion_uses_both_inputs():
    fusion = ConcatFusion(d_model=16, dropout=0.0)
    fusion.eval()
    spatial = torch.randn(1, 3, 4, 16)
    temporal = torch.randn(1, 3, 4, 16)
    with torch.no_grad():
        base = fusion(spatial, temporal)[0]
        changed_spatial = fusion(spatial + 1.0, temporal)[0]
        changed_temporal = fusion(spatial, temporal + 1.0)[0]
    assert not torch.allclose(base, changed_spatial, atol=1e-4)
    assert not torch.allclose(base, changed_temporal, atol=1e-4)


# --------------------------------------------------------------------- losses
def test_huber_ignores_masked_entries():
    prediction = torch.tensor([[[1.0, 100.0]]])
    target = torch.tensor([[[1.0, -100.0]]])
    mask = torch.tensor([[[1.0, 0.0]]])
    assert float(masked_huber_loss(prediction, target, mask)) == pytest.approx(0.0)


def test_huber_gradient_is_bounded_where_squared_error_is_not():
    """Huber's actual purpose: a bounded gradient on extreme residuals.

    The comparison that matters is against squared error, not absolute error.
    Beyond delta, Huber's gradient saturates at delta while MSE's keeps growing
    with the residual -- so a single 600 ug/m3 fireworks spike contributes a
    bounded update instead of dominating the batch. (Against L1, Huber is
    asymptotically identical in the tail and gentler near zero.)
    """
    mask = torch.ones(1, 1, 1)
    target = torch.tensor([[[50.0]]])

    huber_input = torch.zeros(1, 1, 1, requires_grad=True)
    masked_huber_loss(huber_input, target, mask, delta=1.0).backward()

    mse_input = torch.zeros(1, 1, 1, requires_grad=True)
    (((mse_input - target) ** 2) * mask).sum().backward()

    huber_gradient = float(huber_input.grad.abs().max())
    mse_gradient = float(mse_input.grad.abs().max())

    assert huber_gradient == pytest.approx(1.0), "gradient must saturate at delta"
    assert mse_gradient > 50 * huber_gradient, "squared error must explode by comparison"


def test_huber_matches_squared_error_below_delta():
    """Near zero it is quadratic, so small residuals are not over-weighted."""
    mask = torch.ones(1, 1, 1)
    prediction = torch.zeros(1, 1, 1)
    target = torch.tensor([[[0.4]]])
    assert float(masked_huber_loss(prediction, target, mask, delta=1.0)) == pytest.approx(
        0.5 * 0.4**2
    )


def test_huber_becomes_linear_above_delta():
    mask = torch.ones(1, 1, 1)
    prediction = torch.zeros(1, 1, 1)
    a = float(masked_huber_loss(prediction, torch.tensor([[[10.0]]]), mask, delta=1.0))
    b = float(masked_huber_loss(prediction, torch.tensor([[[11.0]]]), mask, delta=1.0))
    assert b - a == pytest.approx(1.0), "unit increases in residual add a constant"


def test_fully_masked_batch_gives_a_finite_zero_loss():
    prediction = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 3, 4)
    loss = masked_huber_loss(prediction, target, mask)
    assert float(loss.detach()) == 0.0
    loss.backward()          # must not raise


# ---------------------------------------------------------------- the overfit
@pytest.mark.parametrize("name", ["T0", "D1"])
def test_model_can_overfit_a_tiny_batch(name):
    """A model that cannot memorise 8 samples has a wiring bug, not a
    generalisation problem -- and this finds it in seconds rather than after a
    full training run."""
    torch.manual_seed(0)
    model = build_model(make_config(**CONFIGS[name], dropout=0.0, edge_dropout=0.0))
    features, wind = make_batch(batch_size=8, seed=1)
    target = torch.randn(8, N_STATIONS, HORIZON)
    mask = torch.ones_like(target)

    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for step in range(200):
        prediction, _ = model(features, wind)
        loss = masked_huber_loss(prediction, target, mask)
        if first is None:
            first = float(loss.detach())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert float(loss) < first * 0.2, (
        f"{name}: loss only fell from {first:.4f} to {float(loss):.4f}"
    )
