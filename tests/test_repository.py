"""Gates on what a fresh clone actually contains, and how it is entered.

These exist because of a real failure: ``stations.csv`` lived under the
gitignored ``data/raw/`` tree, so it was never committed. Everything worked
locally and the build died immediately on Kaggle with a missing-file error. The
same class of bug -- "works here, absent there" -- is invisible to every other
test in this suite, because every other test runs in a working tree that already
has the file.

The second group covers the companion failure: ``python scripts/foo.py`` puts
``scripts/`` on ``sys.path``, not the repository root, so ``import src`` fails
unless the package happens to be pip-installed. That is fine locally and broken
in a notebook.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Files the pipeline reads from the repository itself, as opposed to downloading
# or generating them. Each one must be tracked by git.
REQUIRED_IN_REPO = [
    "configs/stations_beijing.csv",
    "configs/base.yaml",
    "configs/model/t0.yaml",
    "configs/model/s0.yaml",
    "configs/model/d0.yaml",
    "configs/model/s1.yaml",
    "configs/model/d1.yaml",
    "pyproject.toml",
]

# Scripts invoked as "python scripts/<name>" rather than "python -m".
ENTRY_SCRIPTS = [
    "scripts/run_baselines.py",
    "scripts/run_grid.py",
    "scripts/make_figures.py",
]


def tracked_files() -> set[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        pytest.skip("not a git repository, or git unavailable")
    return set(output.split())


# ------------------------------------------------------- what a clone contains
@pytest.mark.parametrize("relative", REQUIRED_IN_REPO)
def test_pipeline_inputs_are_tracked_by_git(relative):
    """A fresh clone must be able to build without any extra downloads.

    The station table is the one that bit us: hand-compiled reference data, not
    a download, but it sat under data/raw/ where .gitignore excluded it.
    """
    assert relative in tracked_files(), (
        f"{relative} is not tracked by git, so a fresh clone will not have it. "
        f"Either commit it or un-ignore it in .gitignore."
    )


def test_station_table_is_readable_and_covers_twelve_stations():
    from src.data.build_beijing import STATIONS_CSV, load_stations

    stations, coords = load_stations(STATIONS_CSV)
    assert len(stations) == 12
    assert coords.shape == (12, 2)
    assert stations == sorted(stations), "station order must be canonical"
    # Beijing sits near 40 N, 116 E; a coordinate slip shows up immediately.
    assert (39.5 < coords[:, 0]).all() and (coords[:, 0] < 40.5).all()
    assert (116.0 < coords[:, 1]).all() and (coords[:, 1] < 117.0).all()


def test_station_table_is_ascii_utf8():
    """It was CP-1252 with degree symbols once, and the loader assumes UTF-8."""
    from src.data.build_beijing import STATIONS_CSV

    raw = (ROOT / STATIONS_CSV).read_bytes()
    assert all(byte < 128 for byte in raw), "station table must be plain ASCII"
    assert raw.decode("utf-8").splitlines()[0].strip() == "station,lat,lon"


def test_large_artifacts_are_not_committed():
    """Raw data, checkpoints, arrays and figures are regenerable and must stay out.

    The Kaggle notebook is the deliberate exception: it is uploaded to Kaggle
    as-is, so it has to exist in the repository even though it is generated.
    :func:`test_committed_notebook_matches_its_source` keeps it honest.
    """
    offenders = [
        name
        for name in tracked_files()
        if name.endswith((".npy", ".npz", ".pt", ".png"))
        or "PRSA_Data_" in name
    ]
    assert not offenders, f"regenerable artifacts should not be tracked: {offenders}"


def test_kaggle_notebook_is_committed():
    """It is uploaded to Kaggle directly, so a clone must contain it."""
    assert "notebooks/kaggle_run_grid.ipynb" in tracked_files(), (
        "the Kaggle notebook must be committed -- it is uploaded as-is, and "
        "regenerating it requires a checkout that already has it"
    )


def test_committed_notebook_matches_its_source():
    """The committed .ipynb must be what build_notebook.py currently produces.

    Generated files that drift from their source are worse than no generated
    file at all: the notebook uploaded to Kaggle would silently be an older
    pipeline than the code beside it.
    """
    import json
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_notebook import build

    expected = build(ROOT / "notebooks/kaggle_run_grid.py")
    committed = json.loads(
        (ROOT / "notebooks/kaggle_run_grid.ipynb").read_text(encoding="utf-8")
    )
    assert committed["cells"] == expected["cells"], (
        "notebooks/kaggle_run_grid.ipynb is stale -- "
        "run `python scripts/build_notebook.py` and commit the result"
    )


def test_notebook_points_at_a_real_repository():
    """An empty REPO_URL leaves the clone route dead on arrival."""
    source = (ROOT / "notebooks/kaggle_run_grid.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        if line.startswith("REPO_URL"):
            url = line.split("=", 1)[1].split("#")[0].strip().strip('"')
            assert url.startswith("https://") and url.endswith(".git"), (
                f"REPO_URL is not a usable clone URL: {url!r}"
            )
            return
    raise AssertionError("REPO_URL not found in the notebook source")


# ------------------------------------------------- how the scripts are entered
@pytest.mark.parametrize("relative", ENTRY_SCRIPTS)
def test_entry_scripts_bootstrap_their_own_import_path(relative):
    """Each script must work from a fresh clone with no install step.

    Without the bootstrap, "python scripts/run_baselines.py" raises
    ModuleNotFoundError: No module named 'src' -- but only where the package is
    not pip-installed, which is exactly the notebook case.
    """
    source = (ROOT / relative).read_text(encoding="utf-8")
    assert "_ROOT = Path(__file__).resolve().parent.parent" in source, (
        f"{relative} does not add the repository root to sys.path"
    )
    assert "sys.path.insert" in source


@pytest.mark.parametrize("relative", ENTRY_SCRIPTS)
def test_entry_scripts_parse(relative):
    ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def test_notebook_source_declares_every_required_file():
    """The notebook's own completeness check must cover the required files.

    It is the first thing that runs on Kaggle, so it is where a partial upload
    should be caught -- not eight cells later inside the builder.
    """
    source = (ROOT / "notebooks/kaggle_run_grid.py").read_text(encoding="utf-8")
    for relative in ("configs/stations_beijing.csv", "configs/base.yaml"):
        assert relative in source, f"notebook does not check for {relative}"
