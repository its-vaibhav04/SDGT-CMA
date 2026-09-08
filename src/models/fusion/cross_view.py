"""Fusion of the spatial and temporal views.

**The correction this module exists for.** The original design pooled both
branches to a single token per station *before* fusing, which left the
cross-attention nothing to attend over except the 12 stations. It could not make
the spatial branch attend to a particular historical event, because no time axis
remained. Here both branches arrive as ``[B, N, Np, d]`` -- station *and* patch
axes intact -- and attention runs over the patch axis within each station. Only
then is the synopsis's claim, that "the spatial branch attends to temporally
significant patterns", actually true of the tensors.

**On "cross-modal".** The two branches are different *views* of the same tabular
monitoring record, not different modalities like imagery or text. The accurate
term is cross-view attention; SDGT-CMA remains the project label.

**Parameter matching.** Comparing cross-view against concatenation only isolates
the fusion *mechanism* if capacity is held roughly constant -- otherwise the
comparison measures width. Both variants are sized to about ``10 d^2`` and
:func:`count_parameters` is reported alongside every result so the match is
auditable rather than asserted.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossViewFusion(nn.Module):
    """Bidirectional cross-attention over patch tokens, combined by a learned gate."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.spatial_to_temporal = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.temporal_to_spatial = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            spatial, temporal: both ``[B, N, Np, d]``.

        Returns:
            ``(fused [B, N, Np, d], diagnostics)``. The gate is kept because it
            answers a question worth a figure: does the model lean on spatial or
            temporal information, and does that shift with lead time?
        """
        batch, n_stations, n_patches, d_model = temporal.shape
        flat = (batch * n_stations, n_patches, d_model)
        s = spatial.reshape(*flat)
        t = temporal.reshape(*flat)

        # Temporal patches query the spatial context: "given what happened around
        # this station in these hours, which spatial state matters?"
        ctx_t, w_ts = self.spatial_to_temporal(t, s, s, need_weights=return_weights)
        # The mirrored pass.
        ctx_s, w_st = self.temporal_to_spatial(s, t, t, need_weights=return_weights)

        gate = self.gate(torch.cat([ctx_t, ctx_s], dim=-1))
        fused = self.norm(gate * ctx_t + (1.0 - gate) * ctx_s)

        diagnostics = {"gate": gate.reshape(batch, n_stations, n_patches, d_model)}
        if return_weights and w_ts is not None:
            # [B*N, Np, Np] -> [B, N, Np, Np] so the axes stay interpretable.
            shape = (batch, n_stations, n_patches, n_patches)
            diagnostics["attn_temporal_query"] = w_ts.reshape(shape).detach()
            diagnostics["attn_spatial_query"] = w_st.reshape(shape).detach()

        return fused.reshape(batch, n_stations, n_patches, d_model), diagnostics


class ConcatFusion(nn.Module):
    """Parameter-matched concatenation baseline.

    Receives exactly the same tokens as :class:`CrossViewFusion` -- the only
    difference is that it has no mechanism for one view to query the other.
    """

    def __init__(self, d_model: int, hidden_mult: int = 3, dropout: float = 0.1):
        super().__init__()
        hidden = hidden_mult * d_model
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused = self.norm(self.net(torch.cat([spatial, temporal], dim=-1)))
        return fused, {}


class NoFusion(nn.Module):
    """Temporal-only pass-through, for the T0 configuration."""

    def forward(
        self,
        spatial: torch.Tensor | None,
        temporal: torch.Tensor,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return temporal, {}


def build_fusion(kind: str, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> nn.Module:
    if kind == "cross_view":
        return CrossViewFusion(d_model, n_heads=n_heads, dropout=dropout)
    if kind == "concat":
        return ConcatFusion(d_model, dropout=dropout)
    if kind in ("none", None):
        return NoFusion()
    raise ValueError(f"unknown fusion {kind!r}")


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
