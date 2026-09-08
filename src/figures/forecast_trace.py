"""Predicted versus actual PM2.5 through a chosen period.

The Phase 2 deliverable and the figure the project always has available: given a
saved ``predictions.npz`` and a date range, plot what each model said would
happen against what did.

Two of the case studies this is built for are chosen deliberately (see
``analysis/find_episodes.py``):

* **2017-01-01** -- the worst episode in the test year, 24 h mean 443 ug/m3,
  peaking at 522. The case for the model.
* **2017-01-28** -- Chinese New Year. A fireworks-driven spike to 607 ug/m3, the
  highest single hour in the test set and unpredictable from any feature the
  model has. Showing the miss and explaining it is worth more than hiding it,
  and it is the concrete justification for choosing Huber loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.contract import ProcessedDataset
from src.figures import style
from src.metrics import Predictions


def _series_at_horizon(
    predictions: Predictions,
    horizon: int,
    station: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid-time series for one horizon: (valid_hour, predicted, observed).

    The *valid time* of a forecast is ``origin + horizon`` -- the hour being
    predicted, not the hour it was issued. Plotting against origin time would
    shift every curve left by the horizon and make a lagging model look prescient.
    """
    index = horizon - 1
    if station is None:
        mask = predictions.mask[:, :, index]
        counts = mask.sum(axis=1)
        safe = np.maximum(counts, 1)
        # Averaging only over reporting stations; hours with none stay NaN so the
        # line breaks rather than plotting a fabricated zero.
        pred = np.where(mask, predictions.pred[:, :, index], 0.0).sum(axis=1) / safe
        truth = np.where(mask, predictions.truth[:, :, index], 0.0).sum(axis=1) / safe
        pred = np.where(counts > 0, pred, np.nan)
        truth = np.where(counts > 0, truth, np.nan)
    else:
        mask = predictions.mask[:, station, index]
        pred = np.where(mask, predictions.pred[:, station, index], np.nan)
        truth = np.where(mask, predictions.truth[:, station, index], np.nan)

    return predictions.origins + horizon, pred, truth


def plot(
    dataset: ProcessedDataset,
    runs: Sequence[Predictions],
    *,
    start: str,
    end: str,
    horizon: int = 24,
    station: int | None = None,
    title: str | None = None,
    name: str | None = None,
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """Plot every run's forecast for one horizon against the observed series.

    Args:
        start, end: ISO datetimes bounding the *valid time* window.
        station: Station index, or ``None`` for the network mean.
    """
    import matplotlib.pyplot as plt

    style.apply_style()

    hours = dataset.timestamps.astype("datetime64[h]")
    lo = int(np.searchsorted(hours, np.datetime64(start, "h")))
    hi = int(np.searchsorted(hours, np.datetime64(end, "h")))
    if hi <= lo:
        raise ValueError(f"empty window: {start} -> {end}")

    fig, ax = plt.subplots(figsize=(11, 4.2))

    observed_drawn = False
    for order, run in enumerate(runs):
        valid, pred, truth = _series_at_horizon(run, horizon, station)
        keep = (valid >= lo) & (valid < hi)
        if not keep.any():
            continue
        times = hours[valid[keep]]

        if not observed_drawn:
            ax.plot(times, truth[keep], color=style.TRUTH, linewidth=2.0, label="observed", zorder=5)
            observed_drawn = True

        colour = style.SERIES[order % len(style.SERIES)]
        ax.plot(times, pred[keep], color=colour, linewidth=1.5, label=run.model, alpha=0.95)

    # Headroom so the legend never sits on top of an episode peak.
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.18)
    style.shade_aqi_bands(ax)

    where = "network mean" if station is None else dataset.stations[station]
    ax.set_title(title or f"PM2.5 forecast at h={horizon} — {where}")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_xlabel("valid time")
    ax.legend(loc="upper left", ncols=min(len(runs) + 1, 4))
    fig.autofmt_xdate()

    slug = name or f"forecast_h{horizon}_{start[:10]}"
    return style.save(fig, slug, root)


# Baselines that establish a floor rather than compete. Drawn as one muted
# reference group so the contended range stays readable and no categorical hue is
# ever reused -- a cycled palette makes two unrelated series look like a pair.
REFERENCE_MODELS = ("seasonal_naive_24h", "seasonal_naive_168h", "climatology")
FLOOR_MODEL = "persistence"


def _group_by_model(runs: Sequence[Predictions]) -> dict[str, list[Predictions]]:
    """Collect runs of the same configuration across seeds, in a stable order."""
    grouped: dict[str, list[Predictions]] = {}
    for run in runs:
        grouped.setdefault(run.model, []).append(run)
    for group in grouped.values():
        group.sort(key=lambda r: r.seed)
    return grouped


def _curves(
    group: Sequence[Predictions], horizons: Sequence[int], metric: str
) -> tuple[list[float], list[float], list[float]]:
    """Mean, min and max across seeds at each horizon.

    Min/max rather than a standard deviation: with three seeds the range is the
    honest summary, and a one-sigma band on n=3 implies a precision that is not
    there.
    """
    from src.metrics import ALL_METRICS

    fn = ALL_METRICS[metric]
    per_seed = [
        [fn(r.pred[:, :, h - 1], r.truth[:, :, h - 1], r.mask[:, :, h - 1]) for h in horizons]
        for r in group
    ]
    columns = list(zip(*per_seed))
    return (
        [float(np.mean(c)) for c in columns],
        [float(np.min(c)) for c in columns],
        [float(np.max(c)) for c in columns],
    )


def error_by_horizon(
    runs: Sequence[Predictions],
    *,
    horizons: Sequence[int] = (1, 6, 12, 24),
    metric: str = "mae",
    name: str = "error_by_horizon",
    root: Path | str = style.FIGURE_ROOT,
    references: Sequence[str] = (),
) -> list[Path]:
    """Error against forecast horizon, one line per *configuration*.

    Runs of the same configuration under different seeds are averaged, with the
    seed range drawn as a band -- fifteen separate lines for five configurations
    is not a figure, and the spread is exactly what a reader needs to judge
    whether two configurations differ at all.

    Persistence is the dashed floor. The naive baselines collapse into one grey
    group. ``references`` nominates further models to draw as neutral reference
    lines rather than coloured series: LightGBM belongs there, because it is the
    bar the grid has to clear, not a member of the factorial.

    Returns the paths written -- more than one if the contenders had to be split
    across figures. It never raises for having too many series; killing the last
    cell of a multi-hour run over a palette limit is the wrong failure mode.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    grouped = _group_by_model(runs)

    neutral = set(REFERENCE_MODELS) | set(references)
    contenders = [m for m in grouped if m not in neutral and m != FLOOR_MODEL]
    contenders.sort()

    limit = len(style.SERIES)
    chunks = [contenders[i : i + limit] for i in range(0, len(contenders), limit)] or [[]]
    written: list[Path] = []

    for chunk_index, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        drew_naive = False

        for model in REFERENCE_MODELS:
            if model not in grouped:
                continue
            mean, _, _ = _curves(grouped[model], horizons, metric)
            ax.plot(horizons, mean, color=style.REFERENCE, linewidth=1.2, linestyle=":",
                    alpha=0.7, label="naive reference" if not drew_naive else None, zorder=1)
            drew_naive = True

        for model in references:
            if model not in grouped:
                continue
            mean, _, _ = _curves(grouped[model], horizons, metric)
            # Neutral, never a categorical hue: a reference line sharing a
            # colour with a series reads as though the two are related.
            ax.plot(horizons, mean, color=style.REFERENCE_STRONG, linewidth=1.8,
                    linestyle="-.", marker="s", markersize=4,
                    label=f"{model} (bar to clear)", zorder=2)

        if FLOOR_MODEL in grouped:
            mean, _, _ = _curves(grouped[FLOOR_MODEL], horizons, metric)
            ax.plot(horizons, mean, color=style.TRUTH, linewidth=1.8, linestyle="--",
                    marker="o", markersize=5, label=FLOOR_MODEL, zorder=3)

        for index, model in enumerate(chunk):
            group = grouped[model]
            mean, low, high = _curves(group, horizons, metric)
            colour = style.SERIES[index]
            if len(group) > 1:
                ax.fill_between(horizons, low, high, color=colour, alpha=0.16,
                                linewidth=0, zorder=3)
            label = model if len(group) == 1 else f"{model}  ({len(group)} seeds)"
            ax.plot(horizons, mean, color=colour, linewidth=2.0, marker="o",
                    markersize=5, label=label, zorder=4)

        ax.set_xticks(list(horizons))
        ax.set_xlabel("forecast horizon (hours ahead)")
        ax.set_ylabel(f"{metric.upper()} (μg/m³)")
        seeded = any(len(grouped[m]) > 1 for m in chunk)
        subtitle = "\nline = mean across seeds, band = seed range" if seeded else ""
        ax.set_title(f"{metric.upper()} by forecast horizon{subtitle}")
        ax.set_ylim(bottom=0)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", ncols=2, fontsize=8)

        suffix = "" if len(chunks) == 1 else f"_{chunk_index + 1}"
        written.append(style.save(fig, f"{name}{suffix}", root))

    return written
