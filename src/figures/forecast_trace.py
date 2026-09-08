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


# Baselines that exist to establish a floor, not to compete. Drawn as one muted
# reference group so the contended range stays readable and no categorical hue is
# ever reused -- a cycled palette makes two unrelated series look like a pair.
REFERENCE_MODELS = ("seasonal_naive_24h", "seasonal_naive_168h", "climatology")


def error_by_horizon(
    runs: Sequence[Predictions],
    *,
    horizons: Sequence[int] = (1, 6, 12, 24),
    metric: str = "mae",
    name: str = "error_by_horizon",
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """Error against forecast horizon, one line per contending model.

    The core quantitative figure. Persistence is dashed because it is the floor
    every other model must clear to be worth reporting; the weaker naive
    baselines collapse into a single grey reference group.
    """
    import matplotlib.pyplot as plt

    from src.metrics import ALL_METRICS

    style.apply_style()
    fn = ALL_METRICS[metric]

    def curve(run):
        return [
            fn(run.pred[:, :, h - 1], run.truth[:, :, h - 1], run.mask[:, :, h - 1])
            for h in horizons
        ]

    contenders = [r for r in runs if r.model not in REFERENCE_MODELS and r.model != "persistence"]
    persistence = [r for r in runs if r.model == "persistence"]
    references = [r for r in runs if r.model in REFERENCE_MODELS]

    if len(contenders) > len(style.SERIES):
        raise ValueError(
            f"{len(contenders)} contending models exceeds the {len(style.SERIES)}-colour "
            "palette; split into small multiples rather than cycling hues"
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    for index, run in enumerate(references):
        ax.plot(horizons, curve(run), color=style.REFERENCE, linewidth=1.2,
                linestyle=":", marker="", alpha=0.7,
                label="naive reference" if index == 0 else None, zorder=1)

    for run in persistence:
        ax.plot(horizons, curve(run), color=style.TRUTH, linewidth=1.8,
                linestyle="--", marker="o", markersize=5, label="persistence", zorder=2)

    for index, run in enumerate(contenders):
        ax.plot(horizons, curve(run), color=style.SERIES[index], linewidth=2.0,
                marker="o", markersize=5, label=run.model, zorder=3)

    ax.set_xticks(list(horizons))
    ax.set_xlabel("forecast horizon (hours ahead)")
    ax.set_ylabel(f"{metric.upper()} (μg/m³)")
    ax.set_title(f"{metric.upper()} by forecast horizon")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", ncols=2)

    return style.save(fig, name, root)
