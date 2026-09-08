"""Forecast-origin bookkeeping and batching.

A *forecast origin* ``t`` is the last hour the model may observe. It reads
``features[t-L+1 : t+1]`` and predicts PM2.5 for hours ``t+1 … t+tau``. Getting
this off by one is the single most common silent bug in spatio-temporal
forecasting: it does not crash, it just quietly inflates validation accuracy.
``tests/test_data_contract.py`` pins it down explicitly.

Splits are defined on **target** time. An origin belongs to a split when every
one of its ``tau`` target hours falls inside that split's range. The lookback of
the earliest validation and test origins therefore reaches back into the
preceding split. That is correct and is not leakage -- the model is reading
history, exactly as it would in deployment. Only targets must respect the
boundary.

**Why there is no DataLoader here.** The whole Beijing dataset is about 45 MB, so
it fits on the GPU with room to spare. Batching is then pure fancy-indexing into
resident tensors, which is faster than any worker-based pipeline and removes a
whole class of nondeterminism from forked workers. :class:`WindowBatcher` does
that; :class:`WindowDataset` is provided for the cases where a standard
``DataLoader`` is genuinely wanted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.contract import ProcessedDataset, Split


def valid_origins(
    *,
    split: Split,
    n_hours: int,
    lookback: int,
    horizon: int,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Forecast origins whose full target window lies inside ``split``.

    Args:
        split: The target-time block this set belongs to.
        n_hours: Total hours in the record.
        lookback: ``L``; the origin needs ``L`` hours of history behind it.
        horizon: ``tau``; hours predicted ahead.
        target_mask: Optional ``[T, N]`` observation mask. When given, origins
            with no observed target anywhere in the window are dropped -- they
            would contribute nothing to a masked loss but would still dilute the
            mean.

    Returns:
        Sorted int64 array of origin indices.
    """
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must both be >= 1")

    # targets t+1 .. t+horizon must satisfy split.start <= t+1 and t+horizon <= split.end-1
    low = max(split.start - 1, lookback - 1)
    high = min(split.end - horizon, n_hours - horizon) - 1
    if high < low:
        return np.empty(0, dtype=np.int64)

    origins = np.arange(low, high + 1, dtype=np.int64)

    if target_mask is not None:
        offsets = np.arange(1, horizon + 1, dtype=np.int64)
        windows = origins[:, None] + offsets[None, :]        # [n_origins, horizon]
        keep = target_mask[windows].any(axis=(1, 2))
        origins = origins[keep]

    return origins


@dataclass
class Batch:
    """One training batch. Shapes are stated because they are load-bearing."""

    features: torch.Tensor     # [B, L, N, F]
    wind_uv: torch.Tensor      # [B, L, N, 2] raw m/s
    target: torch.Tensor       # [B, N, tau]  standardised
    target_mask: torch.Tensor  # [B, N, tau]  bool
    origins: torch.Tensor      # [B]          int64, index into the hour axis

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            features=self.features.to(device),
            wind_uv=self.wind_uv.to(device),
            target=self.target.to(device),
            target_mask=self.target_mask.to(device),
            origins=self.origins.to(device),
        )

    def __len__(self) -> int:
        return self.features.shape[0]


class WindowTensors:
    """The city record as resident tensors, plus the window-gathering logic."""

    def __init__(self, dataset: ProcessedDataset, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.features = torch.from_numpy(dataset.features).to(self.device)
        self.wind_uv = torch.from_numpy(dataset.wind_uv).to(self.device)
        self.target_scaled = torch.from_numpy(dataset.target_scaled).to(self.device)
        self.target_mask = torch.from_numpy(dataset.target_mask).to(self.device)
        self.n_hours = dataset.n_hours
        self.n_stations = dataset.n_stations
        self.n_features = dataset.n_features

    def gather(self, origins: torch.Tensor, lookback: int, horizon: int) -> Batch:
        """Build a batch for the given origins by indexing, without copying slices.

        ``history`` indexes hours ``t-L+1 … t`` and ``future`` indexes ``t+1 …
        t+tau``; the two never overlap, which is what makes the off-by-one
        impossible to introduce silently here.
        """
        origins = origins.to(self.device)
        back = torch.arange(-lookback + 1, 1, device=self.device)      # [L]
        ahead = torch.arange(1, horizon + 1, device=self.device)       # [tau]

        history = origins[:, None] + back[None, :]                     # [B, L]
        future = origins[:, None] + ahead[None, :]                     # [B, tau]

        target = self.target_scaled[future]                            # [B, tau, N]
        mask = self.target_mask[future]                                # [B, tau, N]

        return Batch(
            features=self.features[history],                           # [B, L, N, F]
            wind_uv=self.wind_uv[history],                             # [B, L, N, 2]
            target=target.permute(0, 2, 1).contiguous(),               # [B, N, tau]
            target_mask=mask.permute(0, 2, 1).contiguous(),            # [B, N, tau]
            origins=origins,
        )


class WindowBatcher:
    """Iterate a split in batches. Shuffling is seeded and reproducible."""

    def __init__(
        self,
        tensors: WindowTensors,
        origins: np.ndarray,
        *,
        lookback: int,
        horizon: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if origins.size == 0:
            raise ValueError("no valid origins for this split; check lookback and horizon")
        self.tensors = tensors
        self.origins = torch.from_numpy(np.asarray(origins, dtype=np.int64))
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        n = len(self.origins)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def n_origins(self) -> int:
        return len(self.origins)

    def __iter__(self) -> Iterator[Batch]:
        order = (
            torch.randperm(len(self.origins), generator=self._generator)
            if self.shuffle
            else torch.arange(len(self.origins))
        )
        for start in range(0, len(order), self.batch_size):
            chunk = order[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                break
            yield self.tensors.gather(self.origins[chunk], self.lookback, self.horizon)


class WindowDataset(Dataset):
    """Standard ``Dataset`` view, for the cases where a ``DataLoader`` is wanted."""

    def __init__(
        self,
        dataset: ProcessedDataset,
        split: Split,
        *,
        lookback: int,
        horizon: int,
    ):
        self.tensors = WindowTensors(dataset, device="cpu")
        self.origins = valid_origins(
            split=split,
            n_hours=dataset.n_hours,
            lookback=lookback,
            horizon=horizon,
            target_mask=dataset.target_mask,
        )
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = self.tensors.gather(
            torch.tensor([self.origins[index]]), self.lookback, self.horizon
        )
        return {
            "features": batch.features[0],
            "wind_uv": batch.wind_uv[0],
            "target": batch.target[0],
            "target_mask": batch.target_mask[0],
            "origin": batch.origins[0],
        }


def build_batchers(
    dataset: ProcessedDataset,
    *,
    lookback: int,
    horizon: int,
    batch_size: int,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, WindowBatcher]:
    """One batcher per split, all sharing a single resident copy of the record."""
    tensors = WindowTensors(dataset, device=device)
    batchers: dict[str, WindowBatcher] = {}
    for name, split in dataset.splits.items():
        origins = valid_origins(
            split=split,
            n_hours=dataset.n_hours,
            lookback=lookback,
            horizon=horizon,
            target_mask=dataset.target_mask,
        )
        batchers[name] = WindowBatcher(
            tensors,
            origins,
            lookback=lookback,
            horizon=horizon,
            batch_size=batch_size,
            shuffle=(name == "train"),
            seed=seed,
        )
    return batchers
