"""Kaggle GPU runner for the SDGT-CMA experiment grid.

This file is the *source* for the Kaggle notebook. Build the .ipynb with:

    python scripts/build_notebook.py

Then upload ``notebooks/kaggle_run_grid.ipynb`` to Kaggle. Cells are separated
by ``# %%``.

Why a notebook at all: Kaggle's free tier gives 30 GPU-hours per week on a
P100 (16 GB) or dual T4, with 9-12 hour sessions. Use **Save & Run All
(Commit)** rather than an interactive session -- a committed run survives
closing the tab, which an interactive one does not.
"""

# %% [markdown]
# # SDGT-CMA — experiment grid on Kaggle GPU
#
# Runs five configurations across three seeds and reports the paired comparisons.
#
# **Before running:**
# 1. Settings → Accelerator → **GPU P100** (or T4 x2)
# 2. Add Data → attach the **Beijing Multi-Site Air Quality** dataset
# 3. Get the project code onto the machine by *one* of:
#    - setting `REPO_URL` below and turning Settings → Internet **On**, or
#    - uploading the repository as a Kaggle Dataset and attaching it
#      (the next cell finds it automatically — no Internet needed)
#
# Then **Save & Run All (Commit)**. Do not use an interactive session for the
# full grid; a committed run keeps going after you close the tab.

# %%
# --- environment check -------------------------------------------------------
import subprocess
import sys

import torch

print(f"torch      {torch.__version__}")
print(f"cuda       {torch.cuda.is_available()}  {torch.version.cuda}")
if torch.cuda.is_available():
    device = torch.cuda.get_device_properties(0)
    print(f"gpu        {device.name}  {device.total_memory / 1e9:.1f} GB")
else:
    print("NO GPU -- switch the accelerator on before running the grid")

# %%
# --- get the project code ----------------------------------------------------
# Three routes, tried in order, so the notebook works whether or not the repo is
# on GitHub and whether or not Internet is enabled.
#
#   1. already unpacked from an earlier cell run
#   2. attached as a Kaggle Dataset (upload the repo as a zip; no Internet)
#   3. git clone from REPO_URL (needs Settings -> Internet On)
REPO_URL = "https://github.com/its-vaibhav04/research_project.git"
PROJECT = "/kaggle/working/SDGT-CMA"

import glob
import os
import shutil
from pathlib import Path

# A file that only the real project has, used to recognise it wherever it came from.
SENTINEL = "src/data/build_beijing.py"


def find_project_in_input():
    """Look for an unpacked copy of the repo among the attached datasets."""
    for candidate in glob.glob(f"/kaggle/input/**/{SENTINEL}", recursive=True):
        return str(Path(candidate).parents[2])       # .../src/data/x.py -> root
    return None


if not Path(PROJECT, SENTINEL).exists():
    source = find_project_in_input()
    if source:
        print(f"copying project from attached dataset: {source}")
        shutil.copytree(source, PROJECT, dirs_exist_ok=True)
    elif REPO_URL:
        print(f"cloning {REPO_URL}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, PROJECT], check=True)
    else:
        raise SystemExit(
            "Could not find the project. Either attach it as a Kaggle Dataset, "
            "or set REPO_URL above and enable Settings -> Internet."
        )

os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
print("working directory:", os.getcwd())

# %%
# --- confirm the copy is complete --------------------------------------------
# A partial copy fails later with a confusing error deep in the pipeline, so
# check the few files that must be present up front. stations_beijing.csv is the
# one that actually went missing once: it is hand-compiled reference data, and
# an earlier version of the project kept it under the gitignored data/raw tree,
# so a fresh clone did not get it.
REQUIRED = [
    "configs/stations_beijing.csv",
    "configs/base.yaml",
    "configs/model/d1.yaml",
    "src/data/build_beijing.py",
    "scripts/run_baselines.py",
    "scripts/run_grid.py",
]
missing = [f for f in REQUIRED if not Path(f).exists()]
if missing and "configs/stations_beijing.csv" in missing and Path("data/raw/stations.csv").exists():
    missing.remove("configs/stations_beijing.csv")     # legacy layout is accepted
if missing:
    raise SystemExit(f"the project copy is incomplete, missing: {missing}")
print("project files present")

# %%
# --- dependencies ------------------------------------------------------------
# Kaggle images already ship torch, numpy, pandas, scikit-learn, matplotlib and
# lightgbm. Only PyYAML and pyarrow are occasionally missing. The scripts add the
# repo root to sys.path themselves, so no install step is required.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "PyYAML", "pyarrow"], check=False
)

# %%
# --- locate the raw data -----------------------------------------------------
# The attached dataset lands somewhere under /kaggle/input. Find the directory
# holding the 12 PRSA station CSVs rather than hard-coding a path that changes
# whenever the dataset slug does.
target_dir = Path("data/raw/PRSA_Data_20130301-20170228")
have = len(list(target_dir.glob("PRSA_Data_*.csv"))) if target_dir.exists() else 0

if have != 12:
    candidates = glob.glob("/kaggle/input/**/PRSA_Data_*.csv", recursive=True)
    if not candidates:
        raise FileNotFoundError(
            "No PRSA CSVs under /kaggle/input. Attach the Beijing Multi-Site "
            "Air Quality dataset via Add Data."
        )
    source_dir = Path(candidates[0]).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in source_dir.glob("PRSA_Data_*.csv"):
        shutil.copy(csv_path, target_dir / csv_path.name)
    print(f"copied station files from {source_dir}")

count = len(list(target_dir.glob("PRSA_Data_*.csv")))
assert count == 12, f"expected 12 station files, found {count}"
print(f"{count} station files ready")

# %%
# --- build the processed dataset ---------------------------------------------
subprocess.run([sys.executable, "-m", "src.data.build_beijing", "--no-era5"], check=True)

# %%
# --- verify the data contract before spending any GPU time -------------------
# Leakage, off-by-one windowing, the wind convention, episode retention, graph
# direction. If any of these fail, nothing downstream is worth running.
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
    capture_output=True,
    text=True,
)
print(result.stdout[-3000:])
if result.returncode != 0:
    raise RuntimeError("tests failed -- fix before training")

# %%
# --- baselines ---------------------------------------------------------------
# Cheap, CPU-bound, and they set the bar. LightGBM is the one that matters: on
# this dataset it beats every linear model at every horizon, which matches the
# published finding. Beating it is the result worth writing up.
subprocess.run([sys.executable, "scripts/run_baselines.py"], check=True)

# %%
# --- time one epoch before committing to the whole grid ----------------------
# D1 is the most expensive configuration. Measure it, then decide whether the
# full grid fits the session limit rather than finding out eight hours in.
import time

start = time.time()
subprocess.run(
    [sys.executable, "-m", "src.train",
     "--config", "configs/model/d1.yaml", "--epochs", "1", "--seed", "42"],
    check=True,
)
per_epoch = time.time() - start

print(f"\nD1: {per_epoch / 60:.1f} min for one epoch (including startup)")
print(f"full grid upper bound (5 configs x 3 seeds, ~60 epochs before early stopping):")
print(f"  roughly {per_epoch * 60 * 15 / 3600:.1f} GPU-hours")
print("Kaggle free tier gives 30 GPU-hours per week and 9-12 hour sessions.")
print("If that exceeds the session limit, either run one config per session")
print("(--configs t0), or set graph.lag_max=4 in configs/base.yaml.")

# %%
# --- the grid ----------------------------------------------------------------
# Five configurations x three seeds. Identical split, preprocessing, masks and
# tuning budget throughout; only graph.type, spatial.type and fusion.type differ,
# which is what makes each pairwise difference attributable to one component.
subprocess.run(
    [sys.executable, "scripts/run_grid.py", "--seeds", "42", "43", "44"], check=True
)

# %%
# --- interpretability diagnostics from the full model ------------------------
# Training runs with diagnostics off; this pass turns them on and dumps the
# adjacency, attention and gate the figures are built from.
subprocess.run(
    [sys.executable, "-m", "src.evaluate",
     "--run", "experiments/runs/d1_wind_crossview_seed42"],
    check=True,
)

# %%
# --- figures -----------------------------------------------------------------
subprocess.run([sys.executable, "scripts/make_figures.py"], check=True)

# %%
# --- package the results for download ----------------------------------------
# Everything needed to reproduce, audit and re-plot: predictions, diagnostics,
# metrics, curves, configs and environment snapshots. Only checkpoints are left
# out -- they are large and regenerable from the saved config plus seed.
#
# diagnostics.npz has to come back: without it the attention and gate figures
# cannot be redrawn locally, and the first run shipped without it.
import tarfile

OUTPUT = "/kaggle/working/sdgt_results.tar.gz"
with tarfile.open(OUTPUT, "w:gz") as archive:
    for directory in ("experiments/runs", "experiments/figures"):
        for path in Path(directory).rglob("*"):
            if path.is_file() and path.suffix != ".pt":
                archive.add(path, arcname=str(path))

print(f"wrote {OUTPUT} ({Path(OUTPUT).stat().st_size / 1e6:.1f} MB)")
print("Download it from the Output tab, then unpack into the repo locally:")
print("  tar xzf sdgt_results.tar.gz")

# %%
# --- final table -------------------------------------------------------------
subprocess.run(
    [sys.executable, "scripts/run_grid.py", "--report-only", "--seeds", "42", "43", "44"],
    check=True,
)
