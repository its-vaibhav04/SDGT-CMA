"""Convert a ``# %%`` cell-delimited Python file into a Jupyter notebook.

    python scripts/build_notebook.py

The Kaggle runner is kept as a plain ``.py`` file so it stays reviewable in a
diff -- notebook JSON does not review well, and a stray execution count should
never show up in a commit. This regenerates the ``.ipynb`` on demand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCE = Path("notebooks/kaggle_run_grid.py")


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    """Split on ``# %%`` markers, honouring ``# %% [markdown]``."""
    cells: list[tuple[str, list[str]]] = []
    kind, buffer = "code", []

    for line in text.splitlines():
        if line.startswith("# %%"):
            if buffer:
                cells.append((kind, buffer))
            kind = "markdown" if "[markdown]" in line else "code"
            buffer = []
            continue
        buffer.append(line)

    if buffer:
        cells.append((kind, buffer))
    return cells


def clean(kind: str, lines: list[str]) -> list[str]:
    """Trim blank edges; strip the leading ``# `` from markdown cells."""
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if kind == "markdown":
        lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines]
    return lines


def build(source: Path) -> dict:
    text = source.read_text(encoding="utf-8")

    # Drop the module docstring: it explains the build step, not the experiment.
    if text.startswith('"""'):
        text = text[text.index('"""', 3) + 3 :]

    cells = []
    for kind, lines in split_cells(text):
        body = clean(kind, lines)
        if not body:
            continue
        source_lines = [line + "\n" for line in body[:-1]] + [body[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source_lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()

    notebook = build(args.source)
    target = args.source.with_suffix(".ipynb")
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    counts = {"code": 0, "markdown": 0}
    for cell in notebook["cells"]:
        counts[cell["cell_type"]] += 1
    print(f"wrote {target}  ({counts['code']} code, {counts['markdown']} markdown cells)")


if __name__ == "__main__":
    main()
