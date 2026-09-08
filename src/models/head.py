"""Forecast head: fused patch tokens to a multi-horizon prediction.

Direct multi-horizon output -- all ``tau`` hours predicted at once rather than
rolled out recursively. Recursive rollout compounds its own errors and, at 24
steps on data this noisy, that compounding dominates everything else.

Flatten-and-project follows PatchTST's own forecasting head. It lets the head
weight patches differently by position (the most recent patch matters far more
at h=1 than at h=24), which mean-pooling would throw away.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """``[B, N, Np, d] -> [B, N, tau]``, applied per station with shared weights."""

    def __init__(
        self,
        d_model: int,
        n_patches: int,
        horizon: int,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Flatten(start_dim=-2),                 # [B, N, Np * d]
            nn.Linear(n_patches * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss over observed targets only.

    Huber behaves like squared error near zero and absolute error far from it, so
    a 600 ug/m3 fireworks spike contributes a bounded gradient instead of
    dominating the batch. The mask matters as much as the loss shape: 2.08 % of
    PM2.5 is missing, and scoring imputed values as truth would quietly reward
    the imputation rather than the forecast.
    """
    if not mask.any():
        return prediction.sum() * 0.0
    per_element = nn.functional.huber_loss(
        prediction, target, reduction="none", delta=delta
    )
    return (per_element * mask).sum() / mask.sum()


def masked_l1_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Plain MAE, for the loss ablation."""
    if not mask.any():
        return prediction.sum() * 0.0
    per_element = (prediction - target).abs()
    return (per_element * mask).sum() / mask.sum()


def build_loss(name: str, delta: float = 1.0):
    if name == "huber":
        return lambda p, t, m: masked_huber_loss(p, t, m, delta)
    if name in ("l1", "mae"):
        return masked_l1_loss
    raise ValueError(f"unknown loss {name!r}")
