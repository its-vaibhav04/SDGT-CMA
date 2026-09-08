"""Linear baselines: ridge and DLinear.

Both predict all ``tau`` hours directly from the lookback window, with weights
shared across stations plus a station one-hot, mirroring how the neural models
treat station identity.

**Why these invert the feature scaling first.** The processed features apply
``log1p`` to pollutants for conditioning, while the target stays standardised
raw so that metrics land in ug/m3. A neural network bridges those two spaces
without difficulty; a linear model cannot. Measured on this dataset, the
correlation between the last observed PM2.5 and the next hour's target is 0.96
in physical units but only 0.83 in log space -- so feeding a linear baseline the
log-scaled column cripples it for a reason that has nothing to do with being
linear.

The comparison that matters is *equal information*, not identical array
contents: same rows, same split, same target, same masks. Handicapping the
baselines by denying them a sensible representation would stack the deck for the
proposed model, which is precisely the failure this project is trying to avoid.

DLinear is included because *Are Transformers Effective for Time Series
Forecasting?* (AAAI 2023) showed a one-layer linear model on a trend/seasonal
decomposition matching or beating Transformers on standard long-horizon
benchmarks. If it wins here, that is a finding.
"""

from __future__ import annotations

import numpy as np

from src.data.contract import ProcessedDataset
from src.metrics import Predictions

RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)

# Channels a linear model can actually use. The PM2.5 observation mask and
# staleness counter are included deliberately: without them the baseline cannot
# distinguish a real reading from a climatology fill, which costs it most at the
# 1-hour horizon where persistence is otherwise unbeatable. The neural models see
# these channels, so the baselines must too. The ceiling flag is genuinely
# useless here -- it fires on 60 station-hours out of 420,768.
LINEAR_FEATURES = (
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "WSPM", "RAIN",
    "wind_u", "wind_v",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "obs_PM2.5", "since_obs_PM2.5",
)

# Lag set: dense over the recent history where autocorrelation is high (0.97 at
# 1 h, 0.77 at 6 h), sparse further back where it is not (0.15 at 48 h).
LAG_OFFSETS = (1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 36, 48)


def physical_features(dataset: ProcessedDataset, names: tuple[str, ...]) -> np.ndarray:
    """Selected feature columns inverted back to physical units, ``[T, N, len(names)]``."""
    columns = []
    for name in names:
        index = dataset.feature_index(name)
        spec = dataset.scaler.specs[index]
        columns.append(spec.invert(dataset.features[..., index]))
    return np.stack(columns, axis=-1)


def _design(
    values: np.ndarray,
    origins: np.ndarray,
    lookback: int,
    n_stations: int,
) -> np.ndarray:
    """Lagged design matrix, ``[n * N, n_lags * f + N]``, station-major rows.

    Row ``i * N + j`` holds station ``j``'s history at origin ``origins[i]``,
    matching the target layout in :func:`_targets`.
    """
    lags = [lag for lag in LAG_OFFSETS if lag <= lookback]
    blocks = []
    for lag in lags:
        at = origins - lag + 1                                   # lag 1 == the origin hour
        block = values[at]                                       # [n, N, f]
        blocks.append(block.reshape(len(origins) * n_stations, -1))
    flat = np.concatenate(blocks, axis=1)

    onehot = np.tile(np.eye(n_stations, dtype=np.float64), (len(origins), 1))
    return np.concatenate([flat, onehot], axis=1)


def _targets(dataset: ProcessedDataset, origins: np.ndarray, horizon: int):
    offsets = np.arange(1, horizon + 1)
    future = origins[:, None] + offsets[None, :]
    truth = dataset.target_raw[future].transpose(0, 2, 1)         # [n, N, tau]
    mask = dataset.target_mask[future].transpose(0, 2, 1)
    n, N, tau = truth.shape
    return truth, mask, truth.reshape(n * N, tau), mask.reshape(n * N, tau)


def _standardise(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mean) / std
    return np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1), mean, std


def _select_alpha(Xtr, Ytr, Wtr, Xva, Yva, Wva, alphas=RIDGE_ALPHAS) -> np.ndarray:
    """Fit one ridge per horizon, each with its own penalty chosen on validation MAE.

    Two details that matter more than they look.

    *Per-horizon rows.* Each horizon uses only the rows observed at that horizon.
    Regressing onto a masked-to-zero target would drag predictions toward zero
    wherever data is missing.

    *Per-horizon penalty.* A single global penalty is chosen by whichever horizon
    dominates the mean validation error -- here h=24, whose errors are four times
    larger than h=1's. That over-regularises the short horizons badly enough that
    ridge loses to persistence at h=1. Each horizon picks its own.

    The normal equations are formed once per horizon and reused across the
    penalty grid, since only the diagonal changes.
    """
    n_features, n_horizons = Xtr.shape[1], Ytr.shape[1]
    identity = np.eye(n_features)
    weights = np.zeros((n_features, n_horizons))

    for h in range(n_horizons):
        train_rows, val_rows = Wtr[:, h], Wva[:, h]
        if train_rows.sum() < n_features or not val_rows.any():
            continue

        Xk = Xtr[train_rows]
        gram = Xk.T @ Xk
        rhs = Xk.T @ Ytr[train_rows, h]

        best_column, best_score = None, np.inf
        for alpha in alphas:
            column = np.linalg.solve(gram + alpha * identity, rhs)
            score = float(np.abs(Xva[val_rows] @ column - Yva[val_rows, h]).mean())
            if score < best_score:
                best_column, best_score = column, score
        weights[:, h] = best_column
    return weights


def ridge(
    dataset: ProcessedDataset,
    origins: dict[str, np.ndarray],
    *,
    lookback: int = 48,
    horizon: int = 24,
    seed: int = 0,
    model_name: str = "ridge",
) -> Predictions:
    """Direct multi-output ridge in physical units. Penalty chosen on validation MAE."""
    values = physical_features(dataset, LINEAR_FEATURES)
    n_stations = dataset.n_stations

    Xtr, mean, std = _standardise(_design(values, origins["train"], lookback, n_stations))
    _, _, Ytr, Wtr = _targets(dataset, origins["train"], horizon)

    Xva, _, _ = _standardise(_design(values, origins["val"], lookback, n_stations), mean, std)
    _, _, Yva, Wva = _targets(dataset, origins["val"], horizon)

    weights = _select_alpha(Xtr, Ytr, Wtr, Xva, Yva, Wva)

    Xte, _, _ = _standardise(_design(values, origins["test"], lookback, n_stations), mean, std)
    truth, mask, _, _ = _targets(dataset, origins["test"], horizon)

    pred = (Xte @ weights).reshape(len(origins["test"]), n_stations, horizon)
    return Predictions(
        origins=origins["test"],
        pred=pred.astype(np.float32),
        truth=truth.astype(np.float32),
        mask=mask,
        model=model_name,
        seed=seed,
    )


def dlinear(
    dataset: ProcessedDataset,
    origins: dict[str, np.ndarray],
    *,
    lookback: int = 48,
    horizon: int = 24,
    kernel: int = 25,
    seed: int = 0,
) -> Predictions:
    """DLinear: split the PM2.5 history into trend and residual, fit each linearly.

    The trend is a centred moving average; the seasonal part is what remains. Two
    linear maps from lookback to horizon are summed. Only the target channel is
    used, which is what makes it a genuinely minimal comparator.
    """
    index = dataset.feature_index("PM2.5")
    spec = dataset.scaler.specs[index]
    pm25 = spec.invert(dataset.features[..., index])              # [T, N] ug/m3

    pad = kernel // 2
    window = np.ones(kernel) / kernel

    def decompose(subset: np.ndarray) -> np.ndarray:
        back = np.arange(-lookback + 1, 1)
        history = subset[:, None] + back[None, :]                 # [n, L]
        series = pm25[history]                                    # [n, L, N]
        series = series.transpose(0, 2, 1).reshape(-1, lookback)  # [n*N, L]

        padded = np.pad(series, ((0, 0), (pad, pad)), mode="edge")
        trend = np.apply_along_axis(
            lambda row: np.convolve(row, window, mode="valid"), 1, padded
        )[:, :lookback]
        return np.concatenate([trend, series - trend], axis=1)    # [n*N, 2L]

    Xtr, mean, std = _standardise(decompose(origins["train"]))
    _, _, Ytr, Wtr = _targets(dataset, origins["train"], horizon)

    Xva, _, _ = _standardise(decompose(origins["val"]), mean, std)
    _, _, Yva, Wva = _targets(dataset, origins["val"], horizon)

    weights = _select_alpha(Xtr, Ytr, Wtr, Xva, Yva, Wva)

    Xte, _, _ = _standardise(decompose(origins["test"]), mean, std)
    truth, mask, _, _ = _targets(dataset, origins["test"], horizon)

    pred = (Xte @ weights).reshape(len(origins["test"]), dataset.n_stations, horizon)
    return Predictions(
        origins=origins["test"],
        pred=pred.astype(np.float32),
        truth=truth.astype(np.float32),
        mask=mask,
        model="dlinear",
        seed=seed,
    )
