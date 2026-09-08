"""Weekly block bootstrap for forecast comparisons.

Why blocks. Sliding windows overlap by ``L-1`` hours, so adjacent forecasts are
almost the same forecast. Treating each window as an independent sample would
give absurdly tight intervals. PM2.5 decorrelates over roughly two days, so
resampling whole *weeks* -- keeping all stations and all hours within a week
together -- is the smallest block that safely exceeds the decorrelation time.

Why paired. On this dataset a one-year test split contains about 53 independent
weeks, which puts a +/-13 % confidence interval on any absolute MAE. That is far
wider than the 1-3 % effects this project is looking for, so **absolute MAE
cannot rank models here**. But when two models are scored on identical rows, the
shared week-to-week variance cancels and paired differences resolve below half a
percent. Always compare paired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from src.metrics import ALL_METRICS, Predictions

HOURS_PER_WEEK = 168


@dataclass
class Interval:
    """A bootstrap estimate with a percentile confidence interval."""

    point: float
    low: float
    high: float
    level: float = 0.95
    n_blocks: int = 0

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero (only meaningful for differences)."""
        return self.low > 0.0 or self.high < 0.0

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0

    def __str__(self) -> str:
        return f"{self.point:+.3f} [{self.low:+.3f}, {self.high:+.3f}]"


def week_ids(origins: np.ndarray) -> np.ndarray:
    """Assign each forecast origin to a week block."""
    return origins // HOURS_PER_WEEK


def _blocks(origins: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    weeks = week_ids(origins)
    unique = np.unique(weeks)
    return unique, [np.flatnonzero(weeks == week) for week in unique]


def _errors(predictions: Predictions, horizon: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-window error contributions at one horizon, plus the count that backs each.

    Returned as sums and counts rather than means so that resampling can pool
    correctly: a week with more observed station-hours must carry more weight.
    """
    index = horizon - 1
    pred = predictions.pred[:, :, index]
    truth = predictions.truth[:, :, index]
    mask = predictions.mask[:, :, index]

    if metric == "mae":
        contribution = np.abs(pred - truth)
    elif metric == "rmse":
        contribution = (pred - truth) ** 2
    else:
        raise ValueError(f"blocked resampling supports mae and rmse, not {metric!r}")

    contribution = np.where(mask, contribution, 0.0)
    return contribution.sum(axis=1), mask.sum(axis=1)


def _aggregate(totals: np.ndarray, counts: np.ndarray, metric: str) -> float:
    denominator = counts.sum()
    if denominator == 0:
        return float("nan")
    value = totals.sum() / denominator
    return float(np.sqrt(value)) if metric == "rmse" else float(value)


def absolute_interval(
    predictions: Predictions,
    *,
    horizon: int,
    metric: str = "mae",
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Confidence interval for one model's metric. Expect it to be wide."""
    unique, blocks = _blocks(predictions.origins)
    totals, counts = _errors(predictions, horizon, metric)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        rows = np.concatenate([blocks[j] for j in picked])
        draws[i] = _aggregate(totals[rows], counts[rows], metric)

    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return Interval(
        point=_aggregate(totals, counts, metric),
        low=float(low),
        high=float(high),
        level=level,
        n_blocks=len(unique),
    )


def paired_interval(
    baseline: Predictions,
    candidate: Predictions,
    *,
    horizon: int,
    metric: str = "mae",
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Interval for ``metric(baseline) - metric(candidate)`` on identical rows.

    A positive point estimate means the candidate is better. The two runs must
    share an origin set; that is guaranteed by the frozen split, and checked
    here because a silent misalignment would invalidate the comparison.
    """
    if not np.array_equal(baseline.origins, candidate.origins):
        raise ValueError(
            "paired comparison requires identical origin sets -- "
            f"{baseline.model} has {len(baseline.origins)} windows, "
            f"{candidate.model} has {len(candidate.origins)}"
        )
    if not np.array_equal(baseline.mask, candidate.mask):
        raise ValueError("paired comparison requires identical target masks")

    unique, blocks = _blocks(baseline.origins)
    base_totals, counts = _errors(baseline, horizon, metric)
    cand_totals, _ = _errors(candidate, horizon, metric)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        rows = np.concatenate([blocks[j] for j in picked])
        draws[i] = _aggregate(base_totals[rows], counts[rows], metric) - _aggregate(
            cand_totals[rows], counts[rows], metric
        )

    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    point = _aggregate(base_totals, counts, metric) - _aggregate(cand_totals, counts, metric)
    return Interval(point=point, low=float(low), high=float(high), level=level, n_blocks=len(unique))


def compare(
    baseline: Predictions,
    candidate: Predictions,
    *,
    horizons: Sequence[int] = (1, 6, 12, 24),
    metric: str = "mae",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[int, Interval]:
    return {
        h: paired_interval(
            baseline, candidate, horizon=h, metric=metric, n_boot=n_boot, seed=seed
        )
        for h in horizons
    }


def seed_spread(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard deviation across seeds.

    Report both. When two configurations differ by less than this spread, the
    honest word is "comparable" -- do not bold the smaller number.
    """
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def describe_comparison(
    baseline: Predictions,
    candidate: Predictions,
    intervals: dict[int, Interval],
    metric: str = "mae",
) -> str:
    lines = [
        f"{candidate.model} vs {baseline.model}  ({metric.upper()}, paired weekly block bootstrap)",
        f"{'h':>4}  {'difference':>26}  {'verdict':>12}",
    ]
    for horizon, interval in intervals.items():
        verdict = "significant" if interval.significant else "comparable"
        lines.append(f"{horizon:>4}  {str(interval):>26}  {verdict:>12}")
    lines.append(f"blocks: {next(iter(intervals.values())).n_blocks} independent weeks")
    return "\n".join(lines)
