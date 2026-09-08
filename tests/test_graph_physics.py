"""Physics gates for the graph modules.

The failure this file exists to prevent: the adjacency is built with one index
convention and aggregated with another. The two are transposes, nothing crashes,
and the model quietly learns that pollution travels upwind. No loss curve or
attention heatmap will reveal it. A synthetic westerly will.

Stations are laid out on a clean east-west line so the expected answer is
obvious by inspection rather than by trusting the same trigonometry twice.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.graphs.dynamic import StaticDistanceGraph, WindGraph, lag_kernel, shift_back
from src.graphs.geometry import bearing_matrix, haversine_matrix, transport_bearing

# West -> East at a constant latitude. Roughly 8.5 km apart at 39.9 N.
LINE_COORDS = np.array(
    [[39.9, 116.0], [39.9, 116.1], [39.9, 116.2], [39.9, 116.3]], dtype=np.float64
)
WEST, MID_WEST, MID_EAST, EAST = 0, 1, 2, 3


def wind_from(direction: str, speed: float, shape=(1, 1, 4)) -> torch.Tensor:
    """Wind blowing *toward* the named compass direction, as (u, v) components."""
    vectors = {
        "east": (speed, 0.0),
        "west": (-speed, 0.0),
        "north": (0.0, speed),
        "south": (0.0, -speed),
    }
    u, v = vectors[direction]
    uv = torch.zeros(*shape, 2)
    uv[..., 0] = u
    uv[..., 1] = v
    return uv


# ------------------------------------------------------------------ geometry
def test_bearing_matrix_is_east_and_west_on_a_line():
    bearings = bearing_matrix(LINE_COORDS)
    assert bearings[WEST, EAST] == pytest.approx(90.0, abs=1.0), "west->east should bear ~090"
    assert bearings[EAST, WEST] == pytest.approx(270.0, abs=1.0), "east->west should bear ~270"


def test_distances_increase_along_the_line():
    distance = haversine_matrix(LINE_COORDS)
    assert distance[WEST, MID_WEST] < distance[WEST, MID_EAST] < distance[WEST, EAST]
    assert distance[WEST, WEST] == pytest.approx(0.0)
    np.testing.assert_allclose(distance, distance.T, atol=1e-9)


def test_transport_bearing_matches_the_component_convention():
    # u eastward -> air moving east -> bearing 090
    assert transport_bearing(np.array(3.0), np.array(0.0)) == pytest.approx(90.0)
    # v northward -> bearing 000
    assert transport_bearing(np.array(0.0), np.array(3.0)) == pytest.approx(0.0)
    assert transport_bearing(np.array(-3.0), np.array(0.0)) == pytest.approx(270.0)


# -------------------------------------------------------------- static graph
def test_static_graph_is_symmetric_before_normalisation():
    graph = StaticDistanceGraph(LINE_COORDS)
    adjacency = graph.adjacency.numpy()
    assert (adjacency >= 0).all()
    np.testing.assert_allclose(adjacency.sum(axis=1), 1.0, atol=1e-6)


def test_static_graph_prefers_near_stations():
    graph = StaticDistanceGraph(LINE_COORDS)
    row = graph.adjacency[WEST]
    assert row[MID_WEST] > row[MID_EAST] >= row[EAST]


# ------------------------------------------------------- THE DIRECTION GATE
def test_westerly_wind_makes_western_sources_influence_eastern_targets():
    """The gate. Under a wind blowing toward the east, the eastern station must
    draw from the west, and the western station must not draw from the east."""
    graph = WindGraph(LINE_COORDS, top_k=3)
    adjacency, _, _ = graph(wind_from("east", 5.0))
    a = adjacency[0, 0].detach()             # [target, source]

    assert float(a[EAST, WEST]) > 0.0, "eastern target should receive from the west"
    assert float(a[EAST, WEST]) > float(a[WEST, EAST]), (
        "influence must run west->east under a westerly; "
        f"got a[E,W]={float(a[EAST, WEST]):.4f} vs a[W,E]={float(a[WEST, EAST]):.4f}"
    )
    assert float(a[WEST, EAST]) == pytest.approx(0.0, abs=1e-6), (
        "an eastern source is downwind of a western target and must not influence it"
    )


def test_reversing_the_wind_transposes_the_prior():
    """Physics is antisymmetric under wind reversal.

    Checked on the *unnormalised* prior. Row normalisation is deliberately not
    symmetric -- the most upwind station has no incoming sources and ends up as a
    pure self-edge -- so the identity holds on the physical weights, not on the
    normalised adjacency. If this fails, the module builds with one index
    convention and aggregates with another.
    """
    graph = WindGraph(LINE_COORDS, top_k=3)
    _, _, east = graph(wind_from("east", 5.0))
    _, _, west = graph(wind_from("west", 5.0))

    torch.testing.assert_close(
        east["prior"][0, 0], west["prior"][0, 0].T, atol=1e-5, rtol=1e-4
    )


def test_northerly_wind_creates_no_east_west_edges():
    """Air moving due south should not connect stations lying due east-west.

    The prior is empty, so every target falls back to the static graph -- which
    is the intended behaviour: with no advective connection available, geography
    is a better prior than an empty row.
    """
    graph = WindGraph(LINE_COORDS, top_k=3)
    adjacency, _, diagnostics = graph(wind_from("south", 5.0))

    assert float(diagnostics["prior"].abs().max()) < 1e-5, (
        "perpendicular wind must not produce advective edges"
    )
    assert float(diagnostics["fallback_rate"]) == pytest.approx(1.0)
    torch.testing.assert_close(
        adjacency[0, 0].detach(), graph.static.adjacency, atol=1e-5, rtol=1e-4
    )


# ------------------------------------------------------------------- shape
def test_rows_are_normalised_and_non_negative():
    graph = WindGraph(LINE_COORDS)
    adjacency, _, _ = graph(wind_from("east", 4.0, shape=(2, 5, 4)))
    assert adjacency.shape == (2, 5, 4, 4)
    assert (adjacency >= 0).all()
    torch.testing.assert_close(
        adjacency.sum(-1), torch.ones(2, 5, 4), atol=1e-5, rtol=1e-4
    )


def test_top_k_limits_the_number_of_sources():
    """Sparsity is enforced on the prior; fallback rows follow the static graph."""
    coords = np.array([[39.9, 116.0 + 0.05 * i] for i in range(10)])
    graph = WindGraph(coords, top_k=3)
    _, _, diagnostics = graph(wind_from("east", 6.0, shape=(1, 1, 10)))
    active = (diagnostics["prior"][0, 0] > 1e-9).sum(dim=-1)
    assert int(active.max()) <= 3, "at most top_k advective sources per target"


def test_calm_rows_use_the_static_graph_and_windy_rows_do_not():
    """Mixed conditions: only the targets with no aligned source should fall back."""
    coords = np.array([[39.9, 116.0 + 0.05 * i] for i in range(10)])
    graph = WindGraph(coords, top_k=3)
    adjacency, _, diagnostics = graph(wind_from("east", 6.0, shape=(1, 1, 10)))

    calm = diagnostics["calm"][0, 0, :, 0]
    # The most upwind station has nothing east of it feeding in, so it falls back.
    assert bool(calm[0]), "the most upwind target should have no advective source"
    assert not bool(calm[-1]), "the most downwind target should have sources"
    torch.testing.assert_close(
        adjacency[0, 0, 0].detach(), graph.static.adjacency[0], atol=1e-5, rtol=1e-4
    )


# ---------------------------------------------------------- calm and missing
def test_calm_wind_falls_back_to_the_static_graph():
    """52 % of real hours are near-calm; the graph must not vanish for half the data."""
    graph = WindGraph(LINE_COORDS)
    adjacency, _, diagnostics = graph(wind_from("east", 0.0))
    torch.testing.assert_close(adjacency[0, 0], graph.static.adjacency, atol=1e-5, rtol=1e-4)
    assert float(diagnostics["fallback_rate"]) == pytest.approx(1.0)


def test_strong_wind_does_not_trigger_the_fallback():
    graph = WindGraph(LINE_COORDS)
    _, _, diagnostics = graph(wind_from("east", 6.0))
    assert float(diagnostics["fallback_rate"]) < 1.0


def test_no_nan_under_extreme_inputs():
    graph = WindGraph(LINE_COORDS)
    for uv in (
        torch.zeros(1, 1, 4, 2),                       # dead calm
        torch.full((1, 1, 4, 2), 1e-8),                # near-zero
        torch.full((1, 1, 4, 2), 50.0),                # implausible gale
        torch.tensor([[[[1e3, -1e3]] * 4]]),           # extreme components
    ):
        adjacency, lag, _ = graph(uv)
        assert torch.isfinite(adjacency).all(), f"non-finite adjacency for {uv.flatten()[:2]}"
        assert torch.isfinite(lag).all()


# ------------------------------------------------------------------- lags
def test_lag_is_bounded_and_falls_with_wind_speed():
    graph = WindGraph(LINE_COORDS, lag_min=1, lag_max=6)
    _, slow, _ = graph(wind_from("east", 0.6))
    _, fast, _ = graph(wind_from("east", 20.0))

    assert (slow >= 1).all() and (slow <= 6).all()
    assert (fast >= 1).all() and (fast <= 6).all()
    far = (EAST, WEST)
    assert float(slow[0, 0][far]) > float(fast[0, 0][far]), (
        "slower wind implies a longer travel time"
    )


def test_lag_kernel_is_a_distribution_peaked_at_the_estimate():
    lag = torch.tensor([[2.0, 4.0]])
    kernel = lag_kernel(lag, lag_max=6)
    torch.testing.assert_close(kernel.sum(-1), torch.ones(1, 2), atol=1e-6, rtol=1e-5)
    assert int(kernel[0, 0].argmax()) == 2
    assert int(kernel[0, 1].argmax()) == 4


def test_shift_back_is_causal():
    hidden = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1)
    shifted = shift_back(hidden, 2)
    # values move later in the window; the head repeats the earliest hour
    assert shifted[0, 0, 0, 0] == 0.0 and shifted[0, 1, 0, 0] == 0.0
    assert shifted[0, 2, 0, 0] == 0.0 and shifted[0, 4, 0, 0] == 2.0


def test_shift_back_by_zero_is_identity():
    hidden = torch.randn(2, 4, 3, 5)
    torch.testing.assert_close(shift_back(hidden, 0), hidden)


# ------------------------------------------------------------- learnability
def test_decay_rate_stays_positive_under_gradient_descent():
    """softplus keeps the decay positive, so weight can never grow with distance."""
    graph = WindGraph(LINE_COORDS)
    optimiser = torch.optim.SGD(graph.parameters(), lr=10.0)
    for _ in range(50):
        adjacency, _, _ = graph(wind_from("east", 5.0))
        loss = -adjacency.sum()          # push hard against the parameterisation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert float(graph.decay_rate.detach()) > 0.0


def test_gradients_reach_the_graph_parameters():
    graph = WindGraph(LINE_COORDS)
    adjacency, _, _ = graph(wind_from("east", 5.0))
    adjacency.sum().backward()
    assert graph.decay_logit.grad is not None
    assert torch.isfinite(graph.decay_logit.grad).all()
