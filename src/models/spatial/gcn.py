"""Graph convolution over a fixed adjacency -- the conventional spatial branch.

Every neighbour is weighted by the (normalised) adjacency and nothing else: the
aggregation is the same at 3 a.m. in still air as it is during a 6 m/s
north-westerly. That is precisely the limitation the wind-aware branch is meant
to improve on, so implementing it correctly matters for the comparison, not just
as a throwaway.

Applied per hour, so the output keeps the ``[B, L, N, d]`` shape the patch
pooling expects.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """One propagation step: ``H' = act(A H W)``, with a residual connection."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.transform = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: ``[B, L, N, d]``
            adjacency: ``[B, L, N, N]``, row-normalised, ``[target, source]``.
        """
        messages = torch.einsum("blij,bljd->blid", adjacency, self.transform(hidden))
        return self.norm(hidden + self.dropout(self.activation(messages)))


class GCNSpatial(nn.Module):
    """Stacked graph convolutions. ``[B, L, N, d] -> [B, L, N, d]``."""

    def __init__(self, d_model: int, n_layers: int = 2, dropout: float = 0.1, **_ignored):
        super().__init__()
        self.layers = nn.ModuleList(GCNLayer(d_model, dropout) for _ in range(n_layers))

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        lag: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = hidden
        for layer in self.layers:
            out = layer(out, adjacency)

        diagnostics: dict[str, torch.Tensor] = {}
        if return_weights:
            # A GCN has no learned attention; the adjacency *is* the weighting.
            diagnostics["attention"] = adjacency.detach()
        return out, diagnostics
