"""Shared figure styling.

One palette, one set of defaults, used by every figure module so the thesis
reads as one document rather than ten notebooks. The categorical colours were
checked for colour-vision-deficiency separation (worst adjacent pair dE 8.1
under protanopia, 21.3 under normal vision) before being fixed here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: these run on Kaggle and in CI, not in a window

import matplotlib.pyplot as plt  # noqa: E402

FIGURE_ROOT = Path("experiments/figures")

# Categorical series colours, assigned in fixed order and never cycled.
SERIES = ["#1F6FB2", "#C2601A", "#2E8B5A", "#8E4BA6", "#B03A2E"]
TRUTH = "#191A17"
REFERENCE = "#82857C"
GRID = "#DFDCD4"

# Sequential ramp for magnitude, single hue, light to dark.
RAMP = ["#F2E3CE", "#E4C393", "#D19E52", "#B87A2A", "#94581A", "#6B3D12"]

# Chinese Ambient Air Quality Standard PM2.5 daily bands (ug/m3). Used for
# shading rather than as a claim: an hourly value is not a 24-hour standard.
AQI_BANDS = [
    (0, 35, "#E8F0E4", "Good"),
    (35, 75, "#F4EFD9", "Moderate"),
    (75, 115, "#F2E0C8", "Lightly polluted"),
    (115, 150, "#EED4C4", "Moderately polluted"),
    (150, 250, "#E8C4C0", "Heavily polluted"),
    (250, 10000, "#DCB4C4", "Severely polluted"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#5C5F58",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig, name: str, root: Path | str = FIGURE_ROOT) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def shade_aqi_bands(ax, alpha: float = 0.32) -> None:
    """Light background bands for pollution severity, to give values context."""
    top = ax.get_ylim()[1]
    for low, high, colour, _ in AQI_BANDS:
        if low >= top:
            break
        ax.axhspan(low, min(high, top), color=colour, alpha=alpha, zorder=0, linewidth=0)
