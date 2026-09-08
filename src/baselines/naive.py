"""Naive baselines: persistence, seasonal naive, and station-hour climatology.

These are not filler. On highly autocorrelated air-quality data they are often
more informative than another complex neural comparator, and a published study
on this exact dataset finds that careful classical methods beat persistence by
only 6-18 % RMSE. If a proposed architecture cannot clear these, no amount of
tuning will make it interesting.

Expected persistence MAE on Beijing (from ``analysis/profile_dataset.py``):
10.54 / 32.67 / 45.44 / 58.53 ug/m3 at h = 1 / 6 / 12 / 24, computed over all
rows. Restricted to the test split the values differ slightly; the point is that
an implementation landing far from these has a pipeline bug, not a modelling one.
"""

from __future__ import annotations

import numpy as np

from src.data.contract import ProcessedDataset
from src.metrics import Predictions


def _windows(dataset: ProcessedDataset, origins: np.ndarray, horizon: int):
    """Target values and mask for each origin, shaped ``[n, N, tau]``."""
    offsets = np.arange(1, horizon + 1)
    future = origins[:, None] + offsets[None, :]          # [n, tau]
    truth = dataset.target_raw[future].transpose(0, 2, 1)  # [n, N, tau]
    mask = dataset.target_mask[future].transpose(0, 2, 1)
    return truth, mask


def persistence(
    dataset: ProcessedDataset, origins: np.ndarray, horizon: int, seed: int = 0
) -> Predictions:
    """Carry the last observed value forward: ``yhat[t+h] = y[t]``.

    Where the origin hour itself is unobserved, fall back to the most recent
    observed value at that station -- otherwise the baseline would be penalised
    for missing data rather than for being naive.
    """
    truth, mask = _windows(dataset, origins, horizon)

    last = np.where(dataset.target_mask, dataset.target_raw, np.nan)
    filled = _carry_forward(last)
    anchor = filled[origins]                               # [n, N]
    pred = np.repeat(anchor[:, :, None], horizon, axis=2)

    return Predictions(
        origins=origins,
        pred=pred.astype(np.float32),
        truth=truth.astype(np.float32),
        mask=mask,
        model="persistence",
        seed=seed,
    )


def seasonal_naive(
    dataset: ProcessedDataset,
    origins: np.ndarray,
    horizon: int,
    period: int = 24,
    seed: int = 0,
) -> Predictions:
    """``yhat[t+h] = y[t+h-period]``, the same hour one period earlier.

    With ``period=24`` this is "tomorrow looks like today"; with ``period=168``,
    "next Tuesday looks like last Tuesday". Both are worth reporting: the
    weekday effect in Beijing is real (18 % spread) even though the lag-168
    autocorrelation is ~0.02.
    """
    truth, mask = _windows(dataset, origins, horizon)
    filled = _carry_forward(np.where(dataset.target_mask, dataset.target_raw, np.nan))

    offsets = np.arange(1, horizon + 1)
    source = origins[:, None] + offsets[None, :] - period          # [n, tau]
    source = np.clip(source, 0, dataset.n_hours - 1)
    pred = filled[source].transpose(0, 2, 1)                       # [n, N, tau]

    return Predictions(
        origins=origins,
        pred=pred.astype(np.float32),
        truth=truth.astype(np.float32),
        mask=mask,
        model=f"seasonal_naive_{period}h",
        seed=seed,
    )


def climatology(
    dataset: ProcessedDataset, origins: np.ndarray, horizon: int, seed: int = 0
) -> Predictions:
    """Training-split mean by (station, month, hour of day).

    Uses no recent history at all, so it isolates how much of the signal is pure
    seasonality. Fitted on the training split only.
    """
    truth, mask = _windows(dataset, origins, horizon)
    table = _fit_climatology(dataset)

    months = dataset.timestamps.astype("datetime64[M]").astype(int) % 12
    hours = dataset.timestamps.astype("datetime64[h]").astype(np.int64) % 24

    offsets = np.arange(1, horizon + 1)
    future = origins[:, None] + offsets[None, :]
    pred = table[months[future], hours[future]].transpose(0, 2, 1)   # [n, N, tau]

    return Predictions(
        origins=origins,
        pred=pred.astype(np.float32),
        truth=truth.astype(np.float32),
        mask=mask,
        model="climatology",
        seed=seed,
    )


def _carry_forward(values: np.ndarray) -> np.ndarray:
    """Unlimited backwards fill along the time axis of a ``[T, N]`` array.

    Unlike the pipeline's 3-hour limited fill, a naive baseline may use the last
    observation however old it is -- that *is* the baseline. The global mean
    covers a leading gap with nothing behind it.
    """
    out = values.copy()
    for station in range(out.shape[1]):
        column = out[:, station]
        last = np.nan
        for t in range(out.shape[0]):
            if np.isfinite(column[t]):
                last = column[t]
            else:
                column[t] = last
    return np.nan_to_num(out, nan=float(np.nanmean(values)))


def _fit_climatology(dataset: ProcessedDataset) -> np.ndarray:
    """``[12, 24, N]`` table of training-split means, with sensible fallbacks."""
    train = dataset.splits["train"].as_slice()
    target = dataset.target_raw[train]
    observed = dataset.target_mask[train]
    months = (dataset.timestamps.astype("datetime64[M]").astype(int) % 12)[train]
    hours = (dataset.timestamps.astype("datetime64[h]").astype(np.int64) % 24)[train]

    n_stations = dataset.n_stations
    table = np.zeros((12, 24, n_stations), dtype=np.float64)

    for station in range(n_stations):
        column = target[:, station]
        valid = observed[:, station]
        overall = float(column[valid].mean()) if valid.any() else 0.0
        for month in range(12):
            in_month = valid & (months == month)
            month_mean = float(column[in_month].mean()) if in_month.any() else overall
            for hour in range(24):
                cell = in_month & (hours == hour)
                table[month, hour, station] = (
                    float(column[cell].mean()) if cell.any() else month_mean
                )
    return table
