"""Lag-aware, wind-directed graph prior.

Every design choice here traces to a measurement in ``SDGT-CMA_Evidence_Review.md``.

**Direction.** ``wd`` in the raw data is the direction wind blows *from*, so
transport runs the opposite way. Verified from the data rather than assumed: air
labelled NNW averages 22.6 ug/m3, air labelled ESE averages 89.7. The adjacency
is indexed ``[target, source]`` so that aggregating for a target sums messages
from its sources.

**Lag.** The directional signal between stations peaks at a **2-hour** lag and is
gone by 12. A same-hour edge is not a transport model, it is an association. Each
edge therefore carries a travel time estimated from distance and source wind
speed.

**Calm fallback.** 52 % of hours are below 1.5 m/s. The originally proposed
``ReLU(align * speed)`` weight collapses the graph to self-loops for half the
dataset. Below a threshold this falls back to the static distance graph, and the
fallback rate is recorded -- it is a reportable diagnostic, not an implementation
detail.

**Bounded terms.** Distance decay is parameterised through ``softplus`` so it
cannot turn negative and make weight grow with distance; the speed gate is a
``tanh``; alignment is a clamped cosine. Every factor lives in ``[0, 1]`` and
rows are normalised, so the prior is a distribution over sources by construction.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graphs.geometry import bearing_matrix, haversine_matrix


class StaticDistanceGraph(nn.Module):
    """Thresholded-Gaussian distance graph. Symmetric, fixed, row-normalised.

    The conventional baseline, and the fallback the dynamic graph reverts to in
    calm air. Being symmetric, it is immune to the transpose bug -- which is
    exactly why it is built and tested first.
    """

    def __init__(self, coords: np.ndarray, sigma: float | None = None, percentile: float = 75.0):
        super().__init__()
        distance = haversine_matrix(coords)
        off_diagonal = distance[~np.eye(len(coords), dtype=bool)]

        self.sigma = float(sigma if sigma is not None else off_diagonal.std())
        self.threshold = float(np.percentile(off_diagonal, percentile))

        weights = np.exp(-(distance**2) / (self.sigma**2))
        weights[distance > self.threshold] = 0.0
        np.fill_diagonal(weights, 1.0)
        weights = weights / weights.sum(axis=1, keepdims=True)

        self.register_buffer("adjacency", torch.tensor(weights, dtype=torch.float32))

    def forward(self, wind_uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Broadcast the fixed graph to ``[B, L, N, N]`` with zero lag."""
        batch, lookback = wind_uv.shape[0], wind_uv.shape[1]
        adjacency = self.adjacency.expand(batch, lookback, -1, -1)
        lag = torch.zeros_like(adjacency)
        return adjacency, lag, {"fallback_rate": torch.tensor(1.0)}


class WindGraph(nn.Module):
    """Dynamic, directed, lag-aware prior recomputed from wind at every hour."""

    def __init__(
        self,
        coords: np.ndarray,
        *,
        lag_max: int = 6,
        lag_min: int = 1,
        top_k: int = 4,
        distance_scale_km: float = 20.0,
        speed_scale: float = 2.0,
        min_speed: float = 0.5,
        calm_threshold: float = 1e-3,
        static_sigma: float | None = None,
    ):
        super().__init__()
        self.lag_max = lag_max
        self.lag_min = lag_min
        self.top_k = top_k
        self.distance_scale_km = distance_scale_km
        self.speed_scale = speed_scale
        self.min_speed = min_speed
        self.calm_threshold = calm_threshold
        self.n_stations = len(coords)

        self.register_buffer(
            "distance", torch.tensor(haversine_matrix(coords), dtype=torch.float32)
        )
        # [source, target]: alignment is evaluated at the source, so this is the
        # index order the physics needs. See src/graphs/geometry.py.
        self.register_buffer(
            "bearing", torch.tensor(bearing_matrix(coords), dtype=torch.float32)
        )
        self.register_buffer("eye", torch.eye(self.n_stations))

        self.static = StaticDistanceGraph(coords, sigma=static_sigma)

        # softplus(0.5413) == 1.0, so decay starts at exp(-dist/20km).
        self.decay_logit = nn.Parameter(torch.tensor(0.5413))
        # Self-weight, also positive by construction.
        self.self_logit = nn.Parameter(torch.tensor(0.5413))

    @property
    def decay_rate(self) -> torch.Tensor:
        return F.softplus(self.decay_logit)

    def forward(self, wind_uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            wind_uv: ``[B, L, N, 2]`` raw m/s, direction of motion.

        Returns:
            ``(adjacency [B, L, N, N], lag [B, L, N, N], diagnostics)``.
            Adjacency rows are targets and sum to 1; lag is in hours.
        """
        u = wind_uv[..., 0]
        v = wind_uv[..., 1]
        speed = torch.sqrt(u * u + v * v + 1e-12)                    # [B, L, N]

        # Transport bearing at each source, degrees clockwise from north.
        transport = torch.rad2deg(torch.atan2(u, v)) % 360.0         # [B, L, N]

        # delta[..., j, i] compares the source's transport direction with the
        # bearing from that source to the target.
        delta = transport.unsqueeze(-1) - self.bearing               # [B, L, src, tgt]
        aligned = torch.cos(torch.deg2rad(delta)).transpose(-1, -2)  # -> [B, L, tgt, src]

        decay = torch.exp(-self.decay_rate * self.distance / self.distance_scale_km)
        speed_gate = torch.tanh(speed / self.speed_scale).unsqueeze(-2)   # [B, L, 1, src]

        prior = F.relu(aligned) * decay * speed_gate                 # [B, L, tgt, src]
        prior = prior * (1.0 - self.eye)                             # self handled explicitly

        prior = self._sparsify(prior)

        # Travel time from the source's own wind speed.
        effective = speed.clamp(min=self.min_speed).unsqueeze(-2)    # [B, L, 1, src]
        hours = self.distance * 1000.0 / (effective * 3600.0)
        lag = hours.clamp(self.lag_min, self.lag_max)

        # A target with no aligned, non-negligible source gets nothing from the
        # physical prior -- whether because the air is calm or because the wind
        # runs perpendicular to every neighbour. Either way, geography is the
        # better prior than an empty row.
        incoming = prior.sum(dim=-1, keepdim=True)                   # [B, L, tgt, 1]
        calm = incoming < self.calm_threshold

        # Normalise first, then substitute, so the fallback rows are exactly the
        # static graph rather than the static graph put through a second
        # normalisation with a different self-edge.
        adjacency = self._finalise(prior)
        static = self.static.adjacency.expand_as(adjacency)
        adjacency = torch.where(calm, static, adjacency)
        lag = torch.where(calm, torch.full_like(lag, float(self.lag_min)), lag)

        active = adjacency > 1e-4
        diagnostics = {
            "prior": prior.detach(),          # pre-fallback, pre-normalisation
            "calm": calm.detach(),
            "fallback_rate": calm.float().mean().detach(),
            "mean_degree": active.float().sum(-1).mean().detach(),
            "decay_rate": self.decay_rate.detach(),
            "mean_lag": (lag * active).sum().detach()
            / active.float().sum().clamp(min=1).detach(),
        }
        return adjacency, lag, diagnostics

    def _sparsify(self, prior: torch.Tensor) -> torch.Tensor:
        """Keep the strongest ``top_k`` sources per target.

        A positive-cosine rule alone leaves roughly half of all directed pairs
        connected, which is not a transport structure -- it is a hemisphere.
        """
        k = min(self.top_k, prior.shape[-1] - 1)
        if k <= 0:
            return prior
        # Scatter the top-k values rather than thresholding: a threshold keeps
        # every tied entry, and ties are the common case (a row of zeros when no
        # source is aligned would survive intact).
        values, indices = prior.topk(k, dim=-1)
        return torch.zeros_like(prior).scatter(-1, indices, values)

    def _finalise(self, prior: torch.Tensor) -> torch.Tensor:
        """Add the explicit self-edge, then row-normalise."""
        self_weight = F.softplus(self.self_logit)
        prior = prior * (1.0 - self.eye) + self_weight * self.eye
        return prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def lag_kernel(lag: torch.Tensor, lag_max: int, temperature: float = 0.5) -> torch.Tensor:
    """Soft one-hot over ``lag_max + 1`` discrete hour taps.

    A hard round would make the travel time non-differentiable and would jump
    discontinuously as wind speed drifts. This spreads each edge over
    neighbouring integer lags, so the estimate stays smooth.

    Returns:
        ``[..., lag_max + 1]`` summing to 1 along the last axis.
    """
    taps = torch.arange(lag_max + 1, device=lag.device, dtype=lag.dtype)
    distance = (taps - lag.unsqueeze(-1)).abs()
    return torch.softmax(-distance / temperature, dim=-1)


def shift_back(hidden: torch.Tensor, steps: int) -> torch.Tensor:
    """Shift a ``[B, L, N, d]`` sequence ``steps`` hours into the past.

    The first ``steps`` positions repeat the earliest available hour, which is
    the causal choice: at the start of a window there is no older state to read.
    """
    if steps == 0:
        return hidden
    head = hidden[:, :1].expand(-1, steps, -1, -1)
    return torch.cat([head, hidden[:, :-steps]], dim=1)


def build_graph(kind: str, coords: np.ndarray, **kwargs) -> nn.Module | None:
    if kind in ("none", None):
        return None
    if kind == "static":
        return StaticDistanceGraph(coords, sigma=kwargs.get("static_sigma"))
    if kind == "dynamic":
        allowed = {
            "lag_max", "lag_min", "top_k", "distance_scale_km",
            "speed_scale", "min_speed", "calm_threshold", "static_sigma",
        }
        return WindGraph(coords, **{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"unknown graph {kind!r}")
