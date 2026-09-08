"""The dynamic wind graph, drawn on the station map.

The headline figure, and a debugging tool before it is a deliverable. The graph
is a deterministic function of wind, distance and two parameters that start at
fixed physical values, so this renders correctly **before any training** -- which
is the point. If the edges do not swing downwind as a front arrives, there is a
direction bug, and this finds it faster than reading the adjacency by hand.

The case study is **4 March 2016**, the strongest clear-out in the test year:
network PM2.5 falls from 369.5 to 55.7 ug/m3 in twelve hours at a mean 3.04 m/s.
Chosen by ``analysis/find_episodes.py``.

Arrows run **source to target**, matching the ``[target, source]`` adjacency
convention: an arrow from A to B means A's air is reaching B, so B should be
drawing on A.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.data.contract import ProcessedDataset
from src.figures import style
from src.graphs.dynamic import WindGraph

# Strongest clear-out in the test year; see analysis/find_episodes.py.
CLEAROUT_START = "2016-03-04T12"
EDGE_THRESHOLD = 0.05          # below this an edge is visual noise


def _hour_index(dataset: ProcessedDataset, when: str) -> int:
    hours = dataset.timestamps.astype("datetime64[h]")
    return int(np.searchsorted(hours, np.datetime64(when, "h")))


def _graph_at(
    dataset: ProcessedDataset,
    hours: Sequence[int],
    graph: WindGraph | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjacency, lag and diagnostics for the given absolute hour indices."""
    graph = graph or WindGraph(dataset.coords)
    wind = torch.from_numpy(dataset.wind_uv[list(hours)]).unsqueeze(0)   # [1, H, N, 2]
    with torch.no_grad():
        adjacency, lag, diagnostics = graph(wind)
    return (
        adjacency[0].numpy(),
        lag[0].numpy(),
        diagnostics["calm"][0].numpy().squeeze(-1),
    )


def _draw_panel(ax, dataset, adjacency, calm, wind_mean, pm25):
    lon = dataset.coords[:, 1]
    lat = dataset.coords[:, 0]

    # Edges first, so station markers sit on top.
    for target in range(dataset.n_stations):
        for source in range(dataset.n_stations):
            if source == target:
                continue
            weight = adjacency[target, source]
            if weight < EDGE_THRESHOLD:
                continue
            ax.annotate(
                "",
                xy=(lon[target], lat[target]),
                xytext=(lon[source], lat[source]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=style.SERIES[0],
                    alpha=min(0.25 + 1.6 * float(weight), 0.95),
                    linewidth=0.6 + 4.0 * float(weight),
                    shrinkA=7,
                    shrinkB=9,
                    connectionstyle="arc3,rad=0.08",
                ),
                zorder=2,
            )

    # Stations first so the axes limits are known, then the wind indicator is
    # placed *inside* the plot area -- above it, it collides with the title.
    lon_pad = 0.16 * float(lon.max() - lon.min())
    lat_pad = 0.16 * float(lat.max() - lat.min())
    ax.set_xlim(float(lon.min()) - lon_pad, float(lon.max()) + lon_pad)
    ax.set_ylim(float(lat.min()) - lat_pad, float(lat.max()) + lat_pad)

    # Mean wind for the panel. Passed in rather than looked up, so synthetic-wind
    # controls draw the wind they were actually given.
    u, v = float(wind_mean[0]), float(wind_mean[1])
    speed = float(np.hypot(u, v))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    anchor = (x0 + 0.20 * (x1 - x0), y1 - 0.13 * (y1 - y0))

    # Below ~0.5 m/s the recorded direction is instrument noise, so drawing an
    # arrow would assert a heading that is not there. Say "calm" instead.
    if speed >= 0.5:
        length = 0.16 * (x1 - x0)
        ax.arrow(
            anchor[0], anchor[1], u / speed * length, v / speed * length,
            head_width=0.11 * length, head_length=0.14 * length,
            length_includes_head=True,
            fc=style.TRUTH, ec=style.TRUTH, linewidth=1.6, zorder=6,
        )
        label = f"wind {speed:.1f} m/s"
    else:
        label = f"calm ({speed:.1f} m/s)"
    ax.text(
        x0 + 0.04 * (x1 - x0), y1 - 0.045 * (y1 - y0), label,
        fontsize=8, color=style.TRUTH, family="monospace", zorder=6,
        va="top",
    )

    # Stations, coloured by concentration and ringed when the graph fell back.
    sizes = np.where(calm, 150, 110)
    scatter = ax.scatter(
        lon, lat, c=pm25, s=sizes, cmap="YlOrBr", vmin=0, vmax=400,
        edgecolors=np.where(calm, style.SERIES[1], "white"),
        linewidths=np.where(calm, 2.0, 1.0), zorder=4,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    # Degrees of longitude shrink with latitude, so an "equal" aspect in raw
    # lat/lon would stretch the map east-west. 1/cos(lat) restores true shape.
    ax.set_aspect(1.0 / np.cos(np.radians(float(lat.mean()))), adjustable="box")
    ax.grid(False)
    return scatter


def clearout_sequence(
    dataset: ProcessedDataset,
    *,
    start: str = CLEAROUT_START,
    step_hours: int = 4,
    panels: int = 4,
    name: str = "graph_clearout_2016_03_04",
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """Multi-panel sequence through a clear-out, one panel every ``step_hours``.

    Read it left to right: as the front arrives the edges should reorient to
    point downwind and the stations should drain from dark to light.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    first = _hour_index(dataset, start)
    hours = [first + step_hours * i for i in range(panels)]

    adjacency, _, calm = _graph_at(dataset, hours)
    pm25 = dataset.target_raw[hours]
    observed = dataset.target_mask[hours]
    pm25 = np.where(observed, pm25, np.nan)

    fig, axes = plt.subplots(1, panels, figsize=(4.0 * panels, 4.6))
    scatter = None
    for panel, (ax, hour) in enumerate(zip(np.atleast_1d(axes), hours)):
        wind_mean = dataset.wind_uv[hour].mean(axis=0)
        scatter = _draw_panel(ax, dataset, adjacency[panel], calm[panel], wind_mean, pm25[panel])
        network = np.nanmean(pm25[panel])
        ax.set_title(
            f"{dataset.timestamps[hour]}\nnetwork mean {network:.0f} μg/m³",
            fontsize=10,
        )

    fig.suptitle(
        "Wind graph through the 4 March 2016 clear-out  —  "
        "arrows run source → target; ringed stations fell back to the static graph",
        fontsize=11, y=1.02,
    )
    if scatter is not None:
        bar = fig.colorbar(scatter, ax=list(np.atleast_1d(axes)), shrink=0.8, pad=0.01)
        bar.set_label("PM2.5 (μg/m³)")

    return style.save(fig, name, root)


def single_hour(
    dataset: ProcessedDataset,
    when: str,
    *,
    name: str | None = None,
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """One hour, drawn large, with the station names labelled."""
    import matplotlib.pyplot as plt

    style.apply_style()
    hour = _hour_index(dataset, when)
    adjacency, lag, calm = _graph_at(dataset, [hour])

    pm25 = np.where(dataset.target_mask[hour], dataset.target_raw[hour], np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    scatter = _draw_panel(
        ax, dataset, adjacency[0], calm[0], dataset.wind_uv[hour].mean(axis=0), pm25
    )

    for index, station in enumerate(dataset.stations):
        ax.annotate(
            station,
            (dataset.coords[index, 1], dataset.coords[index, 0]),
            xytext=(6, 6), textcoords="offset points", fontsize=8,
            color=style.TRUTH, zorder=6,
        )

    mean_lag = float(lag[0][adjacency[0] > EDGE_THRESHOLD].mean()) if (
        adjacency[0] > EDGE_THRESHOLD
    ).any() else float("nan")
    ax.set_title(
        f"Wind graph at {dataset.timestamps[hour]}\n"
        f"mean transport lag {mean_lag:.1f} h  —  "
        f"{int(calm[0].sum())}/{dataset.n_stations} stations on the static fallback"
    )
    bar = fig.colorbar(scatter, ax=ax, shrink=0.75)
    bar.set_label("PM2.5 (μg/m³)")

    return style.save(fig, name or f"graph_{when.replace(':', '').replace('-', '')}", root)


def direction_check(
    dataset: ProcessedDataset,
    *,
    name: str = "graph_direction_check",
    root: Path | str = style.FIGURE_ROOT,
) -> Path:
    """Side-by-side under opposing synthetic winds -- the visual transpose test.

    Not a result, a control. Under a westerly the arrows must point east and
    under an easterly they must point west. Anything else means the adjacency is
    transposed somewhere between construction and aggregation.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    graph = WindGraph(dataset.coords)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (label, u, v) in zip(
        axes,
        [("wind blowing EAST (a westerly)", 6.0, 0.0),
         ("wind blowing WEST (an easterly)", -6.0, 0.0)],
    ):
        wind = torch.zeros(1, 1, dataset.n_stations, 2)
        wind[..., 0], wind[..., 1] = u, v
        with torch.no_grad():
            adjacency, _, diagnostics = graph(wind)

        calm = diagnostics["calm"][0, 0].numpy().squeeze(-1)
        _draw_panel(
            ax, dataset, adjacency[0, 0].numpy(), calm, (u, v),
            np.full(dataset.n_stations, 120.0),
        )
        ax.set_title(label, fontsize=10)

    fig.suptitle(
        "Direction control: arrows must follow the wind, not oppose it",
        fontsize=11, y=0.99,
    )
    return style.save(fig, name, root)
