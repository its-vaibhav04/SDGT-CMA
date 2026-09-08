"""Figures built from raw data and from a trained model's saved diagnostics.

Four of these need nothing but the dataset and can be produced immediately; the
attention and gate plots need a ``diagnostics.npz`` from ``src/evaluate.py``.

A standing caution that belongs in the report as well as here: attention weights
show what a model *weighted*, not what caused its answer. Present them as the
model's learned emphasis. The claim that the graph uses wind is earned by the
wind-reversal control in Phase 9, not by a heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.contract import ProcessedDataset
from src.figures import style
from src.metrics import Predictions, masked_mae

COMPASS_ORDER = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


# ------------------------------------------------------------------ raw data
def pollution_rose(
    dataset: ProcessedDataset,
    *,
    min_speed: float = 2.0,
    name: str = "pollution_rose",
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """Mean PM2.5 by the direction the wind comes *from*.

    The physical justification for the whole wind hypothesis, and it needs no
    model: air arriving from the NNW averages 22.6 ug/m3, from the ESE 89.7 --
    a fourfold swing, and exactly the geography (clean Mongolian plateau to the
    north-west, Hebei industrial corridor to the south-east).

    It doubles as a convention check. If ``wd`` meant the direction wind blows
    *toward*, this plot would be inverted.
    """
    import matplotlib.pyplot as plt

    style.apply_style()

    u = dataset.wind_uv[..., 0]
    v = dataset.wind_uv[..., 1]
    speed = np.hypot(u, v)
    # Direction the wind came FROM is opposite the direction of motion.
    from_bearing = (np.degrees(np.arctan2(u, v)) + 180.0) % 360.0

    usable = (speed >= min_speed) & dataset.target_mask
    sector = np.floor(((from_bearing + 11.25) % 360.0) / 22.5).astype(int)

    means, counts = [], []
    for index in range(16):
        selected = usable & (sector == index)
        counts.append(int(selected.sum()))
        means.append(float(dataset.target_raw[selected].mean()) if selected.any() else np.nan)

    fig, ax = plt.subplots(figsize=(6.4, 6.4), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    angles = np.radians(np.arange(16) * 22.5)
    ramp = style.RAMP
    colours = [
        ramp[min(len(ramp) - 1, max(0, int((value - 20) // 12)))] if np.isfinite(value) else "#DDD"
        for value in means
    ]
    ax.bar(angles, means, width=np.radians(20.0), color=colours,
           edgecolor="white", linewidth=1.2, zorder=3)

    ax.set_xticks(angles)
    ax.set_xticklabels(COMPASS_ORDER, fontsize=9)
    ax.set_ylim(0, max(v for v in means if np.isfinite(v)) * 1.15)
    ax.set_ylabel("")
    ax.grid(color=style.GRID, linewidth=0.7)

    cleanest = int(np.nanargmin(means))
    dirtiest = int(np.nanargmax(means))
    ax.set_title(
        f"Mean PM2.5 by wind origin  (hours with speed ≥ {min_speed:g} m/s)\n"
        f"cleanest {COMPASS_ORDER[cleanest]} {means[cleanest]:.1f}  ·  "
        f"dirtiest {COMPASS_ORDER[dirtiest]} {means[dirtiest]:.1f} μg/m³  "
        f"({means[dirtiest] / means[cleanest]:.1f}× swing)",
        fontsize=11, pad=24,
    )
    return style.save(fig, name, root)


# ---------------------------------------------------- prediction-based slices
def regime_slices(
    dataset: ProcessedDataset,
    baseline: Predictions,
    candidate: Predictions,
    *,
    horizons: Sequence[int] = (6, 12),
    name: str = "regime_slices",
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """Improvement split by wind-speed bin -- the physics claim, as evidence.

    The linear probe in the evidence review predicts the shape: the wind term
    should help two to three times more in windy hours than calm ones (+0.90 %
    versus +0.34 % at h=6). If the trained model reproduces that ordering, the
    mechanism is doing what it claims to.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    if not np.array_equal(baseline.origins, candidate.origins):
        raise ValueError("regime slices require identically-aligned runs")

    speed = np.hypot(dataset.wind_uv[..., 0], dataset.wind_uv[..., 1]).mean(axis=1)
    origin_speed = speed[baseline.origins]

    bins = [(0.0, 1.5, "calm\n< 1.5 m/s"), (1.5, 3.0, "light\n1.5–3"), (3.0, 99.0, "windy\n≥ 3 m/s")]
    fig, axes = plt.subplots(
        1, len(horizons), figsize=(4.6 * len(horizons), 4.8), sharey=True
    )

    for ax, h in zip(np.atleast_1d(axes), horizons):
        index = h - 1
        gains, labels, counts = [], [], []
        for low, high, label in bins:
            rows = (origin_speed >= low) & (origin_speed < high)
            if rows.sum() < 200:
                continue
            mask = baseline.mask[rows, :, index]
            base = masked_mae(baseline.pred[rows, :, index], baseline.truth[rows, :, index], mask)
            cand = masked_mae(candidate.pred[rows, :, index], candidate.truth[rows, :, index], mask)
            gains.append(100.0 * (base - cand) / base)
            labels.append(label)
            counts.append(int(rows.sum()))

        colours = [style.SERIES[2] if g > 0 else style.SERIES[1] for g in gains]
        bars = ax.bar(labels, gains, color=colours, width=0.62, zorder=3)
        for bar, gain, count in zip(bars, gains, counts):
            ax.annotate(
                f"{gain:+.2f}%\nn={count}",
                (bar.get_x() + bar.get_width() / 2, gain),
                textcoords="offset points", xytext=(0, 6 if gain >= 0 else -22),
                ha="center", fontsize=8, color=style.TRUTH,
            )
        ax.axhline(0, color=style.TRUTH, linewidth=1.0)
        ax.set_title(f"h = {h}", fontsize=10)
        ax.set_xlabel("wind regime at the forecast origin")

        # Headroom so the value labels above each bar are not clipped.
        low, high = ax.get_ylim()
        ax.set_ylim(min(low, 0.0) * 1.3 - 0.5, max(high, 0.0) * 1.25 + 0.5)

    np.atleast_1d(axes)[0].set_ylabel("MAE improvement (%)")
    fig.suptitle(
        f"{candidate.model} versus {baseline.model}, by wind regime\n"
        "the physics prediction is that gains concentrate on the right",
        fontsize=11,
    )
    # Leave room between the suptitle and the per-axes titles.
    fig.subplots_adjust(top=0.80)
    return style.save(fig, name, root)


def station_error_map(
    dataset: ProcessedDataset,
    predictions: Predictions,
    *,
    horizon: int = 24,
    name: str = "station_error_map",
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """Test MAE per station, drawn on the map.

    Shows whether the graph helps the peripheral stations (Huairou, Dingling,
    Changping) more than the central ones -- which is where an advective prior
    should matter most, since they sit at the edge of the network.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    index = horizon - 1
    errors = np.array([
        masked_mae(
            predictions.pred[:, station, index],
            predictions.truth[:, station, index],
            predictions.mask[:, station, index],
        )
        for station in range(dataset.n_stations)
    ])

    lon, lat = dataset.coords[:, 1], dataset.coords[:, 0]
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    scatter = ax.scatter(
        lon, lat, c=errors, s=90 + 14 * (errors - errors.min()),
        cmap="YlOrBr", edgecolors="white", linewidths=1.2, zorder=3,
    )
    for station, name_ in enumerate(dataset.stations):
        ax.annotate(
            f"{name_}\n{errors[station]:.1f}",
            (lon[station], lat[station]), xytext=(7, -4),
            textcoords="offset points", fontsize=8, color=style.TRUTH,
        )

    ax.set_aspect(1.0 / np.cos(np.radians(float(lat.mean()))), adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(
        f"{predictions.model} — test MAE by station at h={horizon}\n"
        f"network {errors.mean():.1f} μg/m³, "
        f"best {dataset.stations[int(errors.argmin())]} {errors.min():.1f}, "
        f"worst {dataset.stations[int(errors.argmax())]} {errors.max():.1f}"
    )
    bar = fig.colorbar(scatter, ax=ax, shrink=0.75)
    bar.set_label("MAE (μg/m³)")
    return style.save(fig, name, root)


# ------------------------------------------------- trained-model diagnostics
def attention_by_regime(
    dataset: ProcessedDataset,
    diagnostics: dict[str, np.ndarray],
    *,
    name: str = "attention_by_regime",
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """Station-to-station attention in a windy hour beside a calm one.

    Shows the mechanism responding to conditions rather than settling into one
    fixed pattern. ``diagnostics`` comes from ``src/evaluate.py`` and must carry
    ``attention`` ``[n, L, N, N]`` and ``wind_speed`` ``[n, L]``.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    attention = diagnostics["attention"]
    speed = diagnostics["wind_speed"]

    flat = speed.reshape(-1)
    windy = int(np.argmax(flat))
    calm = int(np.argmin(flat))
    shape = speed.shape

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, position, label in (
        (axes[0], calm, f"calm ({flat[calm]:.1f} m/s)"),
        (axes[1], windy, f"windy ({flat[windy]:.1f} m/s)"),
    ):
        window, hour = np.unravel_index(position, shape)
        matrix = attention[window, hour]
        image = ax.imshow(matrix, cmap="YlOrBr", vmin=0, vmax=float(attention.max()))
        ax.set_xticks(range(dataset.n_stations))
        ax.set_yticks(range(dataset.n_stations))
        ax.set_xticklabels(dataset.stations, rotation=90, fontsize=7)
        ax.set_yticklabels(dataset.stations, fontsize=7)
        ax.set_xlabel("source")
        ax.set_ylabel("target")
        ax.set_title(label)
        ax.grid(False)

    fig.colorbar(image, ax=list(axes), shrink=0.8, label="attention weight")
    fig.suptitle(
        "Learned attention, calm hour versus windy hour\n"
        "an association diagnostic, not a causal claim",
        fontsize=11,
    )
    return style.save(fig, name, root)


def gate_by_patch(
    diagnostics: dict[str, np.ndarray],
    *,
    name: str = "fusion_gate",
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """Where the fusion gate sits: toward the spatial or the temporal context.

    ``gate`` is ``[n, N, Np, d]``. A value near 1 favours the temporal-query
    context, near 0 the mirrored spatial-query one. A gate pinned at 0.5
    everywhere means the fusion never learned to prefer either view.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    gate = diagnostics["gate"]
    per_patch = gate.mean(axis=(0, 1, 3))            # [Np]
    spread = gate.std(axis=(0, 1, 3))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    patches = np.arange(len(per_patch))
    axes[0].errorbar(patches, per_patch, yerr=spread, color=style.SERIES[0],
                     marker="o", capsize=4, linewidth=2)
    axes[0].axhline(0.5, color=style.REFERENCE, linestyle="--", linewidth=1.2)
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(patches)
    axes[0].set_xlabel("patch position (oldest → most recent)")
    axes[0].set_ylabel("gate value")
    axes[0].set_title("Gate by patch position")
    axes[0].annotate("favours temporal context", (0.02, 0.93), xycoords="axes fraction",
                     fontsize=8, color=style.REFERENCE)
    axes[0].annotate("favours spatial context", (0.02, 0.04), xycoords="axes fraction",
                     fontsize=8, color=style.REFERENCE)

    axes[1].hist(gate.reshape(-1), bins=60, color=style.SERIES[0], alpha=0.85)
    axes[1].axvline(0.5, color=style.REFERENCE, linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("gate value")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Distribution (mean {gate.mean():.3f}, sd {gate.std():.3f})")

    fig.suptitle("Fusion gate — which view the model leans on", fontsize=11)
    return style.save(fig, name, root)
