"""Multivariate patch transformer -- the temporal branch.

**On the name.** PatchTST contributes two ideas: subseries patches as tokens, and
*channel independence*, where each channel is one univariate series processed
with shared weights. This module takes the first and not the second: within a
patch it keeps all of a station's variables together (pollutants, meteorology,
wind, calendar, masks), and shares weights across *stations*. That is a
legitimate and deliberate design -- pollutant/meteorology interaction inside a
station is exactly what we want the encoder to see -- but it is not PatchTST
channel independence, and calling it that would be wrong in a way a reviewer who
knows the paper would notice immediately.

**Why patching.** Feeding 48 individual hourly steps to self-attention is both
expensive and low-signal: one hour on its own says very little. Chopping the
window into short patches and attending between patches captures diurnal
structure far more efficiently.

``patch_indices`` is exported and **imported by the spatial branch**, so both
branches emit tokens on identical patch boundaries. If they ever pooled on
different boundaries the cross-view attention in the fusion layer would be
aligning tokens that do not correspond -- and nothing would crash.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def n_patches(lookback: int, patch_length: int, stride: int) -> int:
    if patch_length > lookback:
        raise ValueError(f"patch_length {patch_length} exceeds lookback {lookback}")
    return (lookback - patch_length) // stride + 1


def patch_indices(lookback: int, patch_length: int, stride: int) -> torch.Tensor:
    """``[Np, patch_length]`` of time offsets into the lookback window.

    Row ``p`` lists the hours belonging to patch ``p``. Shared by both branches.
    """
    count = n_patches(lookback, patch_length, stride)
    starts = torch.arange(count) * stride
    return starts[:, None] + torch.arange(patch_length)[None, :]


def pool_to_patches(hourly: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Mean-pool an hourly sequence within each patch.

    Args:
        hourly: ``[B, L, N, d]``
        indices: ``[Np, P]`` from :func:`patch_indices`

    Returns:
        ``[B, N, Np, d]`` -- note the axis order matches the temporal branch's
        output, ready for fusion.
    """
    gathered = hourly[:, indices]                 # [B, Np, P, N, d]
    pooled = gathered.mean(dim=2)                 # [B, Np, N, d]
    return pooled.permute(0, 2, 1, 3).contiguous()


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding over patch positions.

    Fixed rather than learned because there are only 6-8 patch positions and a
    few hundred independent weather episodes to learn from; spending parameters
    here is a poor trade.
    """

    def __init__(self, d_model: int, max_positions: int = 512):
        super().__init__()
        position = torch.arange(max_positions).unsqueeze(1).float()
        divisor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_positions, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[: x.shape[1]].unsqueeze(0)


class MultivariatePatchTransformer(nn.Module):
    """``[B, L, N, d] -> [B, N, Np, d]``, weights shared across stations."""

    def __init__(
        self,
        d_model: int,
        lookback: int,
        patch_length: int,
        stride: int,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = lookback
        self.patch_length = patch_length
        self.stride = stride
        self.n_patches = n_patches(lookback, patch_length, stride)
        self.register_buffer(
            "indices", patch_indices(lookback, patch_length, stride), persistent=False
        )

        self.patch_projection = nn.Linear(patch_length * d_model, d_model)
        self.positional = SinusoidalPositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-norm: stabler on small data
        )
        # enable_nested_tensor is incompatible with norm_first and only warns.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: ``[B, L, N, d]`` embedded history.

        Returns:
            ``[B, N, Np, d]`` one token per station per patch.
        """
        batch, lookback, n_stations, d_model = hidden.shape
        if lookback != self.lookback:
            raise ValueError(f"expected lookback {self.lookback}, got {lookback}")

        # [B, L, N, d] -> [B, N, L, d] -> fold stations into the batch so one set
        # of weights processes every station's window independently.
        sequence = hidden.permute(0, 2, 1, 3).reshape(batch * n_stations, lookback, d_model)

        patches = sequence[:, self.indices]                            # [B*N, Np, P, d]
        patches = patches.reshape(batch * n_stations, self.n_patches, -1)

        tokens = self.patch_projection(patches)                        # [B*N, Np, d]
        tokens = self.input_dropout(self.positional(tokens))
        encoded = self.norm(self.encoder(tokens))

        return encoded.reshape(batch, n_stations, self.n_patches, d_model)


class TemporalConvNet(nn.Module):
    """Dilated causal TCN -- the non-Transformer control.

    Answers the question the experiment grid must not leave open: is the
    Transformer actually necessary, or would any sequence model do? Emits the
    same ``[B, N, Np, d]`` shape so it drops into the grid unchanged.
    """

    def __init__(
        self,
        d_model: int,
        lookback: int,
        patch_length: int,
        stride: int,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = lookback
        self.n_patches = n_patches(lookback, patch_length, stride)
        self.register_buffer(
            "indices", patch_indices(lookback, patch_length, stride), persistent=False
        )

        blocks = []
        for layer in range(n_layers):
            dilation = 2**layer
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size,
                        padding=(kernel_size - 1) * dilation,
                        dilation=dilation,
                    ),
                    _Chomp((kernel_size - 1) * dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, lookback, n_stations, d_model = hidden.shape
        sequence = hidden.permute(0, 2, 3, 1).reshape(batch * n_stations, d_model, lookback)

        out = sequence
        for block in self.blocks:
            out = out + block(out)                                     # residual

        out = out.permute(0, 2, 1)                                     # [B*N, L, d]
        pooled = out[:, self.indices].mean(dim=2)                      # [B*N, Np, d]
        return self.norm(pooled).reshape(batch, n_stations, self.n_patches, d_model)


class _Chomp(nn.Module):
    """Trim the right padding so a Conv1d stays causal."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.size] if self.size > 0 else x


def build_temporal(kind: str, **kwargs) -> nn.Module:
    if kind == "patch_transformer":
        return MultivariatePatchTransformer(**kwargs)
    if kind == "tcn":
        kwargs.pop("n_heads", None)
        kwargs.pop("ffn_mult", None)
        return TemporalConvNet(**kwargs)
    raise ValueError(f"unknown temporal encoder {kind!r}")
