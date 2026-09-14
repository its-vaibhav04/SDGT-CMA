"""Training diagnostics: loss curves and the convergence check.

The figure that would have caught Run 1 at a glance. Every one of its fifteen
runs had training loss falling by 70 % while validation loss rose from epoch
one -- the shape of memorisation -- and the only place that was visible was
``curve.csv``, which nobody plotted. Now it is the first thing drawn after
training.

One panel per configuration, seeds overlaid, train solid and validation dashed,
with the chosen (best-validation) epoch marked. Read it for three things: the
gap between the two curves, whether validation is still falling at the end, and
whether the marker sits inside the run rather than at epoch 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src import report
from src.figures import style


def loss_curves(
    runs_root: Path | str,
    *,
    names: Sequence[str] = report.GRID,
    name: str = "training_curves",
    root: Path | str = style.FIGURE_ROOT,
) -> Path | None:
    """Train and validation loss per epoch, one panel per configuration.

    Returns ``None`` (and draws nothing) when no run has a curve, so a notebook
    can call it unconditionally.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    runs = [r for r in report.list_runs(runs_root) if r.name in names]
    by_name: dict[str, list] = {}
    for run in runs:
        curve = report.load_curve(run.path)
        if len(curve):
            by_name.setdefault(run.name, []).append((run, curve))
    if not by_name:
        print("  no training curves to draw")
        return None

    ordered = [n for n in names if n in by_name]
    n_panels = len(ordered)
    cols = min(n_panels, 3)
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows), squeeze=False)

    # A shared y-range so the panels are comparable by eye.
    all_losses = np.concatenate(
        [np.r_[c["train_loss"].to_numpy(), c["val_loss"].to_numpy()]
         for group in by_name.values() for _, c in group]
    )
    y_low, y_high = float(np.nanmin(all_losses)) * 0.9, float(np.nanmax(all_losses)) * 1.05

    for axis, model in zip(axes.flat, ordered):
        for index, (run, curve) in enumerate(sorted(by_name[model], key=lambda p: p[0].seed)):
            colour = style.SERIES[index % len(style.SERIES)]
            epochs = curve["epoch"].to_numpy()
            axis.plot(epochs, curve["train_loss"], color=colour, linewidth=1.6,
                      label=f"seed {run.seed} train")
            axis.plot(epochs, curve["val_loss"], color=colour, linewidth=1.6,
                      linestyle="--", label=f"seed {run.seed} val")
            best = report._metrics(run).get("best_epoch")
            if best is not None and 0 <= best < len(curve):
                axis.plot(best, float(curve["val_loss"].iloc[best]), marker="o",
                          color=colour, markersize=6, markeredgecolor="white", zorder=5)
        axis.set_ylim(y_low, y_high)
        axis.set_title(model)
        axis.set_xlabel("epoch")
        axis.set_ylabel("Huber loss (standardised)")

    for axis in list(axes.flat)[n_panels:]:
        axis.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    # One legend for the figure: solid = train, dashed = validation, dot = chosen epoch.
    fig.legend(handles[:2] + [plt.Line2D([], [], marker="o", color=style.TRUTH, linestyle="")],
               ["train", "validation", "chosen epoch"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Training curves — the gap between the lines is the overfitting check", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return style.save(fig, name, root)


def convergence_bars(
    runs_root: Path | str,
    *,
    names: Sequence[str] = report.GRID,
    name: str = "training_convergence",
    root: Path | str = style.FIGURE_ROOT,
) -> Path | None:
    """Final validation/train loss ratio per run, against the 1.0 target.

    The one number that diagnosed Run 1. Bars well above 1.0 are memorising;
    bars below 1.0 mean regularisation is doing more than the data warrants.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    table = report.training_table(runs_root, names)
    if table.empty:
        print("  no training table to draw")
        return None

    models = [n for n in names if n in table.index.get_level_values("model")]
    fig, axis = plt.subplots(figsize=(1.6 * max(len(models), 3) + 2, 3.8))
    width = 0.25
    for index, model in enumerate(models):
        sub = table.loc[model]
        seeds = list(sub.index)
        for j, seed in enumerate(seeds):
            axis.bar(index + (j - (len(seeds) - 1) / 2) * width, sub.loc[seed, "val / train"],
                     width=width, color=style.SERIES[j % len(style.SERIES)],
                     label=f"seed {seed}" if index == 0 else None)
    axis.axhline(1.0, color=style.REFERENCE_STRONG, linestyle="--", linewidth=1.2)
    axis.annotate("1.0 = no gap", (0.995, 1.0), xycoords=("axes fraction", "data"),
                  ha="right", va="bottom", fontsize=8, color=style.REFERENCE_STRONG)
    axis.set_xticks(range(len(models)))
    axis.set_xticklabels(models, rotation=15, ha="right")
    axis.set_ylabel("final val loss / train loss")
    axis.set_title("Convergence — Run 1 sat at 3.0–3.9 on every bar")
    axis.legend(loc="upper right")
    return style.save(fig, name, root)
