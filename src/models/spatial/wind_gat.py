"""Wind-aware graph attention over (source station, transport lag) pairs.

**The design decision.** A lag-aware graph needs, for target *i*, source *j* and
hour *t*, the source's state at ``t - lag_ij``. Gathering that directly builds a
``[B, L, N, N, d]`` tensor -- around 400 MB at realistic batch sizes, and awkward
to make differentiable in the lag. Instead, station *j* at lag *k* is treated as
a **distinct attention key**. With ``N = 12`` and ``K = 6`` that is 84 keys per
target: trivially cheap, differentiable through the soft lag kernel, and the
resulting ``alpha`` reads directly as *"target i drew this much from station j,
k hours ago"* -- exactly what the attention figure needs.

**The prior enters as a log-space bias.** Adding a raw edge weight to an
unnormalised attention logit compares two quantities on unrelated scales. Adding
``log(prior)`` instead makes the physical prior a *multiplicative* factor on
attention probability, which is what a prior should be. Its strength is a
learnable, non-negative scalar, so the model can lean on the physics or ignore it
-- and the learned value is worth reporting either way.

Setting ``lag_max = 0`` recovers a conventional same-hour GAT, which is a free
ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graphs.dynamic import lag_kernel, shift_back

NEG_INF = -1e9
EPSILON = 1e-8


class WindGATLayer(nn.Module):
    """One multi-head attention step over ``(source, lag)`` keys."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        lag_max: int = 6,
        dropout: float = 0.1,
        edge_dropout: float = 0.1,
        lag_temperature: float = 0.5,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must divide by n_heads {n_heads}")

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.lag_max = lag_max
        self.lag_temperature = lag_temperature
        self.edge_dropout = edge_dropout

        self.transform = nn.Linear(d_model, d_model, bias=False)
        self.attn_target = nn.Parameter(torch.empty(n_heads, self.d_head))
        self.attn_source = nn.Parameter(torch.empty(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.attn_target)
        nn.init.xavier_uniform_(self.attn_source)

        # softplus(0.5413) == 1.0: the prior starts at full strength.
        self.prior_logit = nn.Parameter(torch.tensor(0.5413))

        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @property
    def prior_strength(self) -> torch.Tensor:
        return F.softplus(self.prior_logit)

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        log_prior: torch.Tensor,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            hidden: ``[B, L, N, d]``
            adjacency: ``[B, L, N, N]`` row-normalised prior, ``[target, source]``.
            log_prior: ``[B, L, N, N, K+1]`` log of prior x lag kernel, computed
                once per forward pass by the parent module -- the graph does not
                change between layers, so rebuilding it per layer is pure waste.

        Returns:
            ``(output [B, L, N, d], attention [B, L, N, N, K+1] or None)``
        """
        batch, lookback, n_stations, d_model = hidden.shape
        taps = self.lag_max + 1

        projected = self.transform(hidden)
        heads = projected.view(batch, lookback, n_stations, self.n_heads, self.d_head)

        # Every source station's state at each candidate lag.
        lagged = torch.stack(
            [shift_back(projected, k) for k in range(taps)], dim=-2
        )                                                       # [B, L, N, K+1, d]
        lagged_heads = lagged.view(
            batch, lookback, n_stations, taps, self.n_heads, self.d_head
        )

        # Additive attention, GAT style, split into target and source halves.
        target_score = torch.einsum("blnhd,hd->blhn", heads, self.attn_target)
        source_score = torch.einsum("blnkhd,hd->blhnk", lagged_heads, self.attn_source)

        logits = self.leaky(
            target_score[..., :, None, None] + source_score[..., None, :, :]
        )                                                       # [B, L, H, tgt, src, K+1]

        # Physical prior as a multiplicative factor on attention probability.
        logits = logits + self.prior_strength * log_prior.unsqueeze(2)

        # An absent edge receives no attention mass at all.
        absent = (adjacency <= 0).unsqueeze(2).unsqueeze(-1)
        logits = logits.masked_fill(absent, NEG_INF)

        if self.training and self.edge_dropout > 0:
            drop = torch.rand_like(adjacency) < self.edge_dropout
            drop = drop & (adjacency > 0)
            # Never drop the self-edge: a node must always be able to see itself.
            eye = torch.eye(n_stations, device=hidden.device, dtype=torch.bool)
            drop = drop & ~eye
            logits = logits.masked_fill(drop.unsqueeze(2).unsqueeze(-1), NEG_INF)

        flat = logits.reshape(batch, lookback, self.n_heads, n_stations, n_stations * taps)
        weights = torch.softmax(flat, dim=-1)
        weights = torch.nan_to_num(weights)          # a fully masked row yields zeros
        weights = weights.view(batch, lookback, self.n_heads, n_stations, n_stations, taps)

        messages = torch.einsum("blhijk,bljkhd->blihd", self.dropout(weights), lagged_heads)
        messages = messages.reshape(batch, lookback, n_stations, d_model)

        output = self.norm(hidden + self.dropout(messages))
        attention = weights.mean(dim=2).detach() if return_weights else None
        return output, attention


class WindGATSpatial(nn.Module):
    """Stacked wind-aware attention layers. ``[B, L, N, d] -> [B, L, N, d]``."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        lag_max: int = 6,
        dropout: float = 0.1,
        edge_dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            WindGATLayer(
                d_model,
                n_heads=n_heads,
                lag_max=lag_max,
                dropout=dropout,
                edge_dropout=edge_dropout,
            )
            for _ in range(n_layers)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        lag: torch.Tensor,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Built once: the adjacency and lag are fixed for this forward pass, so
        # every layer shares the same physical bias term.
        lag_max = self.layers[0].lag_max
        kernel = lag_kernel(lag, lag_max, self.layers[0].lag_temperature)
        log_prior = torch.log(adjacency.unsqueeze(-1) * kernel + EPSILON)

        out = hidden
        attention = None
        for layer in self.layers:
            out, attention = layer(out, adjacency, log_prior, return_weights=return_weights)

        diagnostics: dict[str, torch.Tensor] = {}
        if return_weights and attention is not None:
            # Full [B, L, N, N, K+1] weights, plus the lag-marginalised view the
            # station-to-station attention figure actually plots.
            diagnostics["attention_by_lag"] = attention
            diagnostics["attention"] = attention.sum(dim=-1)
        diagnostics["prior_strength"] = torch.stack(
            [layer.prior_strength.detach() for layer in self.layers]
        ).mean()
        return out, diagnostics


def build_spatial(kind: str, **kwargs) -> nn.Module | None:
    from src.models.spatial.gcn import GCNSpatial

    if kind in ("none", None):
        return None
    if kind == "gcn":
        return GCNSpatial(**kwargs)
    if kind == "wind_gat":
        return WindGATSpatial(**kwargs)
    raise ValueError(f"unknown spatial branch {kind!r}")
