"""Model assembly.

One class builds all five configurations in the experiment grid. They differ in
exactly three fields -- ``graph.type``, ``spatial.type`` and ``fusion.type`` --
and nothing else, which is what makes every pairwise difference attributable to
one component:

===  ===============  ==============  =========================================
ID   spatial          fusion          isolates
===  ===============  ==============  =========================================
T0   none             none            is a graph needed at all?
S0   static distance  concatenation   conventional reference
D0   lag-aware wind   concatenation   S0 -> D0: the graph, alone
S1   static distance  cross-view      S0 -> S1: the fusion, alone
D1   lag-aware wind   cross-view      full model; D0 -> D1 re-tests fusion
===  ===============  ==============  =========================================

``return_diagnostics`` is the one architectural decision that could not be
deferred. Training runs with it off so nothing extra is retained; evaluation runs
with it on and dumps the adjacency, attention and gate that every interpretability
figure is built from. Adding it after the grid had run would mean running the
grid again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.graphs.dynamic import build_graph
from src.models.embedding import FeatureEmbedding
from src.models.fusion.cross_view import build_fusion
from src.models.head import ForecastHead
from src.models.spatial.wind_gat import build_spatial
from src.models.temporal.patch_transformer import (
    build_temporal,
    n_patches,
    patch_indices,
    pool_to_patches,
)


@dataclass
class ModelConfig:
    """Everything needed to build a model. Mirrors the YAML config layout."""

    n_features: int
    n_stations: int
    coords: np.ndarray

    lookback: int = 48
    horizon: int = 24
    patch_length: int = 8
    patch_stride: int = 8

    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.1

    temporal: str = "patch_transformer"
    temporal_layers: int = 3
    ffn_mult: int = 4

    graph: str = "dynamic"          # none | static | dynamic
    spatial: str = "wind_gat"       # none | gcn | wind_gat
    spatial_layers: int = 2
    edge_dropout: float = 0.1

    fusion: str = "cross_view"      # none | concat | cross_view

    head_hidden: int = 256
    # Predict the change from the last observed value rather than the level.
    # PM2.5 autocorrelation at one hour is 0.969, so persistence is an extremely
    # strong prior at short lead times; asking the head to rediscover it from a
    # patch-pooled summary wastes capacity and measurably fails -- the
    # level-predicting grid was 2x worse than persistence at h=1. With the anchor
    # the model only has to learn the deviation, which is the part that is
    # actually hard. Metrics are unaffected: the level is reconstructed before
    # anything is scored.
    persistence_anchor: bool = True
    graph_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.spatial in ("none", None)) != (self.graph in ("none", None)):
            raise ValueError(
                f"spatial={self.spatial!r} and graph={self.graph!r} must both be "
                "'none' or both be set"
            )
        if self.spatial in ("none", None) and self.fusion not in ("none", None):
            raise ValueError("fusion requires a spatial branch")


class SDGT(nn.Module):
    """Dual-branch spatio-temporal forecaster."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_patches = n_patches(config.lookback, config.patch_length, config.patch_stride)

        self.embedding = FeatureEmbedding(
            n_features=config.n_features,
            n_stations=config.n_stations,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        self.temporal = build_temporal(
            config.temporal,
            d_model=config.d_model,
            lookback=config.lookback,
            patch_length=config.patch_length,
            stride=config.patch_stride,
            n_layers=config.temporal_layers,
            n_heads=config.n_heads,
            ffn_mult=config.ffn_mult,
            dropout=config.dropout,
        )

        self.graph = build_graph(config.graph, config.coords, **config.graph_options)
        self.spatial = build_spatial(
            config.spatial,
            d_model=config.d_model,
            n_layers=config.spatial_layers,
            n_heads=config.n_heads,
            lag_max=config.graph_options.get("lag_max", 6),
            dropout=config.dropout,
            edge_dropout=config.edge_dropout,
        )
        self.fusion = build_fusion(
            config.fusion, config.d_model, n_heads=config.n_heads, dropout=config.dropout
        )

        # The spatial branch pools onto the temporal branch's patch boundaries.
        # Importing the same helper is what guarantees the two token sets line up.
        self.register_buffer(
            "patch_index",
            patch_indices(config.lookback, config.patch_length, config.patch_stride),
            persistent=False,
        )

        self.head = ForecastHead(
            d_model=config.d_model,
            n_patches=self.n_patches,
            horizon=config.horizon,
            hidden=config.head_hidden,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        features: torch.Tensor,
        wind_uv: torch.Tensor,
        anchor: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            features: ``[B, L, N, F]`` scaled model inputs.
            wind_uv: ``[B, L, N, 2]`` raw m/s, direction of motion.
            anchor: ``[B, N]`` last observed target, in target units. Required
                when ``persistence_anchor`` is set.

        Returns:
            ``(prediction [B, N, tau], diagnostics)``. Diagnostics are empty
            unless requested.
        """
        if self.config.persistence_anchor and anchor is None:
            raise ValueError("persistence_anchor is enabled but no anchor was supplied")
        hidden = self.embedding(features)                      # [B, L, N, d]
        temporal = self.temporal(hidden)                       # [B, N, Np, d]

        diagnostics: dict[str, torch.Tensor] = {}

        if self.spatial is None:
            fused, fusion_diagnostics = self.fusion(None, temporal)
        else:
            adjacency, lag, graph_diagnostics = self.graph(wind_uv)
            spatial_hourly, spatial_diagnostics = self.spatial(
                hidden, adjacency, lag, return_weights=return_diagnostics
            )
            spatial = pool_to_patches(spatial_hourly, self.patch_index)   # [B, N, Np, d]
            fused, fusion_diagnostics = self.fusion(
                spatial, temporal, return_weights=return_diagnostics
            )

            if return_diagnostics:
                diagnostics["adjacency"] = adjacency.detach()
                diagnostics["lag"] = lag.detach()
                diagnostics.update(
                    {f"graph_{k}": v for k, v in graph_diagnostics.items() if k != "prior"}
                )
                diagnostics.update(spatial_diagnostics)
            else:
                # Scalars worth logging every step, at negligible cost.
                for key in ("fallback_rate", "mean_degree", "decay_rate"):
                    if key in graph_diagnostics:
                        diagnostics[f"graph_{key}"] = graph_diagnostics[key]
                if "prior_strength" in spatial_diagnostics:
                    diagnostics["prior_strength"] = spatial_diagnostics["prior_strength"]

        if return_diagnostics:
            diagnostics.update(fusion_diagnostics)

        prediction = self.head(fused)
        if self.config.persistence_anchor:
            # The head now carries the change from the last observation, so the
            # forecast is persistence plus a learned correction.
            prediction = prediction + anchor.unsqueeze(-1)

        return prediction, diagnostics

    # -------------------------------------------------------------- reporting
    def parameter_counts(self) -> dict[str, int]:
        """Per-component parameter counts, reported alongside every result.

        The cross-view versus concatenation comparison is only about the fusion
        *mechanism* if capacity is roughly matched. Publishing the numbers makes
        that auditable instead of asserted.
        """
        def count(module: nn.Module | None) -> int:
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "embedding": count(self.embedding),
            "temporal": count(self.temporal),
            "graph": count(self.graph),
            "spatial": count(self.spatial),
            "fusion": count(self.fusion),
            "head": count(self.head),
        }
        counts["total"] = sum(counts.values())
        return counts


def build_model(config: ModelConfig) -> SDGT:
    return SDGT(config)
