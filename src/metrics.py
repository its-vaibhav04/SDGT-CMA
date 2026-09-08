"""Masked forecast metrics, always in ug/m3.

Two rules, both load-bearing.

**Everything is masked.** 2.08 % of PM2.5 is missing. Scoring an imputed value as
if it were truth quietly inflates every number in the report, and the inflation
is largest exactly where the data is worst.

**Everything is in ug/m3.** Metrics on standardised data are uninterpretable to
anyone reading the thesis, and comparing them across differently-scaled runs is
meaningless. Predictions are inverted before they reach this module.

MAPE is deliberately absent from the primary set. It is unstable when
concentrations approach zero and a handful of near-zero hours can dominate the
average for the wrong reason. WAPE carries the same "percentage error" intuition
without the instability. If MAPE is needed to match a specific prior paper, add
it explicitly with a stated denominator floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_HORIZONS = (1, 6, 12, 24)


@dataclass
class Predictions:
    """The common output format. Every model writes exactly this.

    Because the origin set is derived from the frozen split and is identical for
    every model, any two saved runs are automatically row-aligned -- which is
    what makes the paired bootstrap in ``src/bootstrap.py`` valid.
    """

    origins: np.ndarray   # [n] int64,  forecast-origin hour indices
    pred: np.ndarray      # [n, N, tau] float32, ug/m3
    truth: np.ndarray     # [n, N, tau] float32, ug/m3
    mask: np.ndarray      # [n, N, tau] bool
    model: str = "unnamed"
    seed: int = 0

    def __post_init__(self) -> None:
        if not (self.pred.shape == self.truth.shape == self.mask.shape):
            raise ValueError(
                f"shape mismatch: pred {self.pred.shape}, truth {self.truth.shape}, "
                f"mask {self.mask.shape}"
            )
        if len(self.origins) != len(self.pred):
            raise ValueError("origins and predictions disagree on the number of windows")

    @property
    def n_windows(self) -> int:
        return self.pred.shape[0]

    @property
    def n_stations(self) -> int:
        return self.pred.shape[1]

    @property
    def horizon(self) -> int:
        return self.pred.shape[2]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            origins=self.origins,
            pred=self.pred.astype(np.float32),
            truth=self.truth.astype(np.float32),
            mask=self.mask,
            model=np.array(self.model),
            seed=np.array(self.seed),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Predictions":
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                origins=payload["origins"],
                pred=payload["pred"],
                truth=payload["truth"],
                mask=payload["mask"],
                model=str(payload["model"]),
                seed=int(payload["seed"]),
            )


# ------------------------------------------------------------------ primitives
def masked_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.abs(pred[mask] - truth[mask]).mean())


def masked_rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.sqrt(((pred[mask] - truth[mask]) ** 2).mean()))


def masked_wape(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|error| / sum|truth|.

    Scale-aware like MAPE but pooled, so a single near-zero observation cannot
    blow up the average.
    """
    if not mask.any():
        return float("nan")
    denominator = np.abs(truth[mask]).sum()
    if denominator < 1e-9:
        return float("nan")
    return float(100.0 * np.abs(pred[mask] - truth[mask]).sum() / denominator)


def masked_bias(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Mean signed error. Reveals systematic under-forecasting of episodes."""
    if not mask.any():
        return float("nan")
    return float((pred[mask] - truth[mask]).mean())


def masked_r2(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    residual = ((pred[mask] - truth[mask]) ** 2).sum()
    total = ((truth[mask] - truth[mask].mean()) ** 2).sum()
    if total < 1e-9:
        return float("nan")
    return float(1.0 - residual / total)


ALL_METRICS = {
    "mae": masked_mae,
    "rmse": masked_rmse,
    "wape": masked_wape,
    "bias": masked_bias,
    "r2": masked_r2,
}


def compute(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    metrics: Iterable[str] = ("mae", "rmse", "wape", "bias"),
) -> dict[str, float]:
    return {name: ALL_METRICS[name](pred, truth, mask) for name in metrics}


# -------------------------------------------------------------- aggregations
def by_horizon(
    predictions: Predictions,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    metrics: Iterable[str] = ("mae", "rmse", "wape", "bias"),
) -> dict[int, dict[str, float]]:
    """Per-horizon metrics. Horizons are 1-indexed: h=1 is the next hour."""
    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        if h < 1 or h > predictions.horizon:
            raise ValueError(f"horizon {h} outside the predicted range 1..{predictions.horizon}")
        index = h - 1
        out[h] = compute(
            predictions.pred[:, :, index],
            predictions.truth[:, :, index],
            predictions.mask[:, :, index],
            metrics,
        )
    return out


def by_station(
    predictions: Predictions,
    stations: Sequence[str],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    metric: str = "mae",
) -> dict[str, dict[int, float]]:
    """Per-station, per-horizon values -- the source for the error map figure."""
    fn = ALL_METRICS[metric]
    out: dict[str, dict[int, float]] = {}
    for index, name in enumerate(stations):
        out[name] = {
            h: fn(
                predictions.pred[:, index, h - 1],
                predictions.truth[:, index, h - 1],
                predictions.mask[:, index, h - 1],
            )
            for h in horizons
        }
    return out


def by_slice(
    predictions: Predictions,
    selector: np.ndarray,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    metrics: Iterable[str] = ("mae", "rmse"),
) -> dict[int, dict[str, float]]:
    """Metrics restricted to a boolean slice of windows, e.g. a wind-speed bin.

    Args:
        selector: ``[n_windows]`` boolean, or a full ``[n, N, tau]`` mask.
    """
    if selector.ndim == 1:
        selector = selector[:, None, None]
    combined = predictions.mask & selector
    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        index = h - 1
        out[h] = compute(
            predictions.pred[:, :, index],
            predictions.truth[:, :, index],
            combined[:, :, index],
            metrics,
        )
    return out


def top_decile_mask(predictions: Predictions, quantile: float = 0.9) -> np.ndarray:
    """Mask selecting the highest-concentration observed targets.

    This slice is where an early-warning system is actually judged, and it is
    where a model that quietly regresses to the mean is exposed.
    """
    observed = predictions.truth[predictions.mask]
    if observed.size == 0:
        return np.zeros_like(predictions.mask)
    threshold = float(np.quantile(observed, quantile))
    return predictions.mask & (predictions.truth >= threshold)


def summary_table(
    predictions: Predictions,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> list[dict[str, float | str | int]]:
    """Tidy rows: one per (model, seed, horizon). Ready for a DataFrame."""
    rows: list[dict[str, float | str | int]] = []
    for h, values in by_horizon(predictions, horizons).items():
        rows.append({"model": predictions.model, "seed": predictions.seed, "horizon": h, **values})
    return rows


def format_table(rows: Sequence[dict], metrics: Sequence[str] = ("mae", "rmse", "wape")) -> str:
    """A plain-text metrics table for logs and terminal output."""
    header = f"{'model':<28}{'seed':>5}{'h':>5}" + "".join(f"{m:>10}" for m in metrics)
    lines = [header, "-" * len(header)]
    for row in rows:
        line = f"{str(row['model']):<28}{int(row['seed']):>5}{int(row['horizon']):>5}"
        line += "".join(f"{float(row[m]):>10.2f}" for m in metrics)
        lines.append(line)
    return "\n".join(lines)
