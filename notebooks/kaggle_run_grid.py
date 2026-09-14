"""Kaggle GPU runner for the SDGT-CMA experiment, start to finish.

This file is the *source* for the Kaggle notebook. Build the .ipynb with:

    python scripts/build_notebook.py

Then upload ``notebooks/kaggle_run_grid.ipynb`` to Kaggle. Cells are separated
by ``# %%``; ``# %% [markdown]`` starts a markdown cell.

The notebook tells the experiment as a story in thirteen sections: configuration,
data, architecture, training, diagnostics, evaluation, baselines, horizon,
station, graph, attention, failure, conclusions. Every analysis cell is one call
into ``src/report.py`` or ``src/figures``; the logic lives in the repository
where it is tested, not in a notebook where it is not.

Why a notebook at all: Kaggle's free tier gives 30 GPU-hours per week on a
P100 (16 GB) or dual T4, with 9-12 hour sessions. Use **Save & Run All
(Commit)** rather than an interactive session -- a committed run survives
closing the tab, which an interactive one does not.
"""

# %% [markdown]
# # SDGT-CMA — multi-site PM2.5 forecasting with a wind-directed graph
#
# **Spatio-Directional Graph Transformer with Cross-Modal Attention.** Beijing
# Multi-Site Air Quality, 12 stations, 2013-03 → 2017-02, forecasting 1–24 h ahead.
#
# This notebook is the whole experiment, start to finish, in thirteen sections:
#
# | # | section | what it answers |
# |---|---|---|
# | 1 | Configuration | what exactly was run, on what |
# | 2 | Dataset / split | what the model saw, and how the data was cut |
# | 3 | Model architecture | what the five configurations are and how big |
# | 4 | Training | the runs themselves |
# | 5 | Training diagnostics | did they converge, or memorise |
# | 6 | Quantitative evaluation | the numbers, with their spread |
# | 7 | Baseline comparison | was any of this needed |
# | 8 | Horizon-wise analysis | where in the 24 h the model earns its keep |
# | 9 | Station-wise analysis | which stations are hard, and why |
# | 10 | Dynamic graph analysis | does the wind graph actually change with the wind |
# | 11 | Attention / fusion analysis | what the model attends to, at what lag |
# | 12 | Error / failure analysis | where it fails, and whether the graph matters |
# | 13 | Conclusions | what can and cannot be claimed |
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
# full run; a committed run keeps going after you close the tab. Budget about
# 1.5 GPU-hours: ~50 min for the grid, ~15 min for the controls, a few minutes
# for everything else.

# %% [markdown]
# ---
# ## 1. Configuration
#
# Everything reproducible starts here: the hardware, the exact code revision,
# the resolved hyper-parameters, and the three seeds. The five configurations
# differ in exactly three settings — `graph.type`, `spatial.type`, `fusion.type`
# — and in nothing else, which is what makes every pairwise difference
# attributable to a single component.

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
REPO_URL = "https://github.com/its-vaibhav04/SDGT-CMA.git"

import glob
import os
import shutil
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()
PROJECT = "/kaggle/working/SDGT-CMA" if ON_KAGGLE else os.getcwd()

# A file that only the real project has, used to recognise it wherever it came from.
SENTINEL = "src/data/build_beijing.py"


def find_project_in_input():
    """Look for an unpacked copy of the repo among the attached datasets."""
    for candidate in glob.glob(f"/kaggle/input/**/{SENTINEL}", recursive=True):
        return str(Path(candidate).parents[2])       # .../src/data/x.py -> root
    return None


if not Path(PROJECT, SENTINEL).exists():
    source = find_project_in_input() if ON_KAGGLE else None
    if source:
        print(f"copying project from attached dataset: {source}")
        shutil.copytree(source, PROJECT, dirs_exist_ok=True)
    elif REPO_URL and ON_KAGGLE:
        print(f"cloning {REPO_URL}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, PROJECT], check=True)
    else:
        raise SystemExit(
            "Could not find the project. Either attach it as a Kaggle Dataset, "
            "or set REPO_URL above and enable Settings -> Internet."
        )

os.chdir(PROJECT)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
print("working directory:", os.getcwd())

# --- prepare ERA5 cache -------------------------------------------------------

ERA5_DEST = Path(PROJECT) / "data/raw/era5"
ERA5_DEST.mkdir(parents=True, exist_ok=True)

npy_matches = list(Path("/kaggle/input").glob("**/beijing_blh.npy"))
meta_matches = list(Path("/kaggle/input").glob("**/beijing_blh_meta.json"))

if not npy_matches:
    raise FileNotFoundError(
        "Could not find beijing_blh.npy under /kaggle/input"
    )

if not meta_matches:
    raise FileNotFoundError(
        "Could not find beijing_blh_meta.json under /kaggle/input"
    )

shutil.copy2(npy_matches[0], ERA5_DEST / "beijing_blh.npy")
shutil.copy2(meta_matches[0], ERA5_DEST / "beijing_blh_meta.json")

print("ERA5 cache prepared.")
print("  ", ERA5_DEST / "beijing_blh.npy")
print("  ", ERA5_DEST / "beijing_blh_meta.json")

# %%
# --- confirm the copy is complete --------------------------------------------
# A partial copy fails later with a confusing error deep in the pipeline, so
# check the few files that must be present up front. Two of these are derived
# data that a fresh clone cannot regenerate: the hand-compiled station table,
# and the ERA5 boundary-layer-height cache (Kaggle has no Copernicus key, and
# without the cache the builder would silently produce a 27-feature dataset).
REQUIRED = [
    "configs/stations_beijing.csv",
    "configs/base.yaml",
    "configs/model/d1.yaml",
    "data/raw/era5/beijing_blh.npy",
    "data/raw/era5/beijing_blh_meta.json",
    "src/data/build_beijing.py",
    "src/report.py",
    "scripts/run_baselines.py",
    "scripts/run_grid.py",
    "scripts/run_controls.py",
    "scripts/make_figures.py",
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
# --- display helpers for the analysis sections -------------------------------
# Every analysis cell below is one call into src/report.py or src/figures, then
# one of these. They degrade to plain printing if IPython is unavailable, and an
# empty table prints a note rather than an empty box. Self-contained on purpose:
# the analysis half of the notebook must be re-runnable on its own against a
# downloaded bundle, without the setup cells above.
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 60)

try:
    from IPython.display import Image, Markdown, display
except ImportError:  # never on Kaggle; keeps the analysis half runnable as a script
    def display(obj):
        print(obj)

    def Image(filename=None, **_):
        return f"[figure: {filename}]"

    def Markdown(text):
        return text

RUNS = Path("experiments/runs")
FIGURES = Path("experiments/figures")
SEEDS = [42, 43, 44]


def table(frame, title=None, note=None):
    """Show a DataFrame with a heading, or say why there is nothing to show."""
    if title:
        display(Markdown(f"**{title}**"))
    if frame is None or getattr(frame, "empty", False):
        print(f"  (nothing to show{': ' + note if note else ''})")
        return
    display(frame)


def figure(name, caption=None):
    """Show a saved figure inline, or say it is missing."""
    path = FIGURES / f"{name}.png"
    if not path.exists():
        print(f"  (figure {name}.png not found -- was make_figures.py run?)")
        return
    if caption:
        display(Markdown(f"*{caption}*"))
    display(Image(filename=str(path)))


print("helpers ready")

# %%
# --- what was run: grid design and resolved configuration --------------------
from src import report

table(report.grid_design(), "The five configurations, and the single question each one isolates")

config = report.resolved_config("configs/model/d1.yaml")
table(report.flatten_config(config), "Resolved configuration of the full model (base.yaml merged with d1.yaml)")
print(f"seeds: {SEEDS}  --  every configuration is trained once per seed; nothing else varies")

# %% [markdown]
# **Reading the configuration.** The capacity settings — `d_model` 32, two
# temporal layers, a 128-wide head — are deliberately small. The first full run
# used 340k parameters against roughly 243 independent 3-day episodes in the
# training split and overfitted every configuration (train/val loss ratio 3–4×,
# best epoch 0–5). The current setting is ~63–76k parameters. `persistence_anchor`
# makes the head predict the *change* from the last observed value rather than
# the level, so persistence is the model's zero-output; that is what fixed a
# collapse at the 1-hour horizon in the same first run.

# %% [markdown]
# ---
# ## 2. Dataset / split
#
# Beijing Multi-Site Air Quality: 12 stations, hourly, 35,064 hours. The split
# is **by calendar year on target time** — two years train, one validation, one
# test — so the test year is never seen and the lookback of the earliest test
# origin may read history from the validation year (that is not leakage; only
# targets must respect the boundary).
#
# Facts about this dataset that bound everything downstream, measured before
# any model was built (`SDGT-CMA_Evidence_Review.md`):
#
# - mean cross-station PM2.5 correlation **0.887** — the 12 stations behave
#   nearly as one signal;
# - mean pairwise distance **27 km**, transport signal peaking at a **2 h** lag;
# - **52 %** of hours below 1.5 m/s wind — the graph is calm half the time;
# - PM2.5 autocorrelation at 1 h is **0.969** — persistence is very hard to beat;
# - the test year holds **53 independent weeks**, so absolute MAE carries a
#   **±12.6 %** interval and models cannot be ranked by headline number.

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
# ERA5 boundary layer height comes from the committed cache (data/raw/era5), so
# no Copernicus key is needed here. The builder reports whether it was included;
# it must say "included" -- 28 features -- or the run is not comparable.
subprocess.run([sys.executable, "-m", "src.data.build_beijing"], check=True)

# %%
# --- verify the data contract before spending any GPU time -------------------
# Leakage, off-by-one windowing, the wind convention, episode retention, graph
# direction, the ERA5 time zone. If any of these fail, nothing downstream is
# worth running.
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
    capture_output=True,
    text=True,
)
print(result.stdout[-3000:])
if result.returncode != 0:
    raise RuntimeError("tests failed -- fix before training")

# %%
# --- the data, as the model sees it ------------------------------------------
from src.data import contract

dataset = contract.load("beijing")
print(dataset.describe())
assert "blh" in dataset.feature_names, "boundary layer height missing -- the ERA5 cache was not picked up"

table(report.dataset_summary(dataset), "Splits: dates, hours, scored windows, and the target distribution")
table(report.station_missingness(dataset), "Stations: coordinates, PM2.5 coverage, mean level")
table(report.feature_table(dataset), "Model inputs (28), their transform, and the training-fit statistics")

# %% [markdown]
# **Reading the split table.** Validation and test are comparable in level
# (means within ~5 µg/m³, similar p95), so model selection on validation is not
# biased by a regime shift. It is limited by *resolution* instead — see §5.
#
# **On the features.** Pollutant inputs are `log1p`-then-standardised because
# PM2.5 has a 999 tail; the *target* is standardised raw so the training
# objective stays aligned with the MAE reported in µg/m³. `blh` is ERA5
# boundary layer height, shifted from UTC onto Beijing local time — the first
# fetch skipped that shift and produced a physically upside-down diurnal cycle
# that no shape or range check could catch. Observation masks and
# hours-since-observed channels let the model know what was imputed.

# %% [markdown]
# ---
# ## 3. Model architecture
#
# Two branches and a gate.
#
# **Temporal branch** — a *multivariate* patch transformer. Each station's 48-hour
# history is cut into six 8-hour patches, embedded, and attended over. It is
# deliberately not channel-independent (unlike PatchTST): the features interact
# inside the embedding, so wind, pressure and boundary layer height can
# modulate the PM2.5 signal directly.
#
# **Spatial branch** — attention over *(source station, transport lag)* pairs.
# For each target station and hour, every other station at every lag from 0 to
# 6 h is a distinct attention key (84 keys at N=12, K=6). A physical prior enters
# the logits as `log(prior)`, so it multiplies attention probability; the prior
# itself is built from wind direction (transport = wind direction + 180°),
# distance decay, wind speed, and a travel time = distance / source wind speed.
# Below a calm threshold it falls back to a static distance graph. The prior's
# strength and the decay length are learnable, so the model can lean on the
# physics or discount it — and how far they move from 1.0 is reported.
#
# **Cross-view fusion** — a learned gate per channel between the two branches'
# patch tokens (0 = temporal, 1 = spatial, 0.5 = the initialisation). The
# `concat` alternative is parameter-matched so the comparison is about the
# mechanism, not capacity.
#
# **Head** — an MLP over the fused tokens, producing 24 hourly values as a
# *correction to the last observed value* (the persistence anchor).

# %%
# --- how big each configuration is -------------------------------------------
table(
    report.architecture_table(dataset),
    "Trainable parameters per component, per configuration (built from the committed YAML)",
)
print("Adding the graph costs ~12k parameters; cross-view over concat costs ~1.2k.")
print("The comparison between fusion mechanisms is therefore about mechanism, not capacity.")

# %%
# --- the full model, printed -------------------------------------------------
from src.models.sdgt import build_model
from src.train import model_config_from

full_model = build_model(model_config_from(config, dataset))
print(full_model)
del full_model

# %% [markdown]
# ---
# ## 4. Training
#
# Everything with a cost runs here, in order: seven baselines (CPU, minutes),
# a one-epoch timing check, the grid of five configurations × three seeds, an
# interpretability pass that dumps the graph and attention, the negative
# controls, and the figures. Sections 5–13 then only *read* what this section
# wrote, so if the session dies part-way the analysis can be re-run locally
# from the downloaded bundle.
#
# Protocol rules enforced in code, not by discipline:
# - the **test split is read exactly once**, after early stopping has chosen the
#   checkpoint on validation;
# - every model is scored on the **identical origin set** from the frozen split,
#   so any two saved runs are row-aligned and paired comparison is valid;
# - Huber loss on standardised targets, masked so imputed hours are never scored.

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
print("full grid upper bound (5 configs x 3 seeds, ~60 epochs before early stopping):")
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
    [sys.executable, "scripts/run_grid.py", "--seeds", *map(str, SEEDS)], check=True
)

# %%
# --- interpretability diagnostics from the full model ------------------------
# Training runs with diagnostics off; this pass turns them on and dumps the
# adjacency, attention-by-lag and gate that sections 10 and 11 are built from.
subprocess.run(
    [sys.executable, "-m", "src.evaluate",
     "--run", "experiments/runs/d1_wind_crossview_seed42"],
    check=True,
)

# %%
# --- negative controls -------------------------------------------------------
# "The graph did not improve MAE" is a failure to find an effect -- the weakest
# form of a negative claim. These controls turn it into a statement about the
# data: break the wind field the graph reads, relabel every edge's source, cut
# every cross-station edge, occlude one station at a time, and see whether the
# forecast notices. The suite carries its own power references
# (zero_correction, history_shuffle), so a row reading "no effect" can be told
# apart from a test with no power. Section 12 reads the results.
subprocess.run(
    [sys.executable, "scripts/run_controls.py", "--occlusion", "--runs",
     *[f"experiments/runs/d1_wind_crossview_seed{s}" for s in SEEDS]],
    check=True,
)

# The temporal-only model as a contrast: it has no graph, so only the whole-model
# controls apply, and its wind dependence is the number D1's should be read
# against.
subprocess.run(
    [sys.executable, "scripts/run_controls.py",
     "--runs", "experiments/runs/t0_temporal_only_seed42"],
    check=True,
)

# %%
# --- figures -----------------------------------------------------------------
# Every figure is regenerable from the saved artifacts; none needs a model.
subprocess.run([sys.executable, "scripts/make_figures.py"], check=True)

# %% [markdown]
# ---
# ## 5. Training diagnostics
#
# Did the models converge, or memorise? Three things to read off the curves and
# the table:
#
# 1. **The gap between train and validation loss.** The first full run sat at a
#    ratio of 3.0–3.9× on every configuration — memorisation. Near 1.0 is the
#    target; well below 1.0 means dropout and weight decay are doing more than
#    the data warrants.
# 2. **Where the chosen epoch sits.** A best epoch at 0 means the model never
#    improved on its first pass.
# 3. **Whether validation is still moving at the end.** A flat validation curve
#    that early stopping waits out is the honest picture on this dataset:
#    validation holds ~52 independent weeks and cannot resolve small changes.

# %%
table(report.training_table(RUNS), "Convergence per run: epochs, chosen epoch, final val/train loss ratio, time")
figure("training_curves", "Train (solid) and validation (dashed) loss per epoch, one panel per configuration, seeds overlaid; the dot marks the chosen epoch.")
figure("training_convergence", "Final validation/train loss ratio per run. 1.0 is no gap; the first full run sat at 3–4 on every bar.")

# %% [markdown]
# **A limit worth naming.** Across the previous full grid, the best validation
# MAE spanned under 1 µg/m³ between configurations while test MAE at h=24
# spanned five. Validation is roughly four times coarser than the differences it
# is being asked to rank, so early stopping is choosing among near-equals — and
# runs that trained longer sometimes scored *worse* on test while validation
# called them fine. This is a property of one year of hourly data from a
# network that behaves like a single signal, not of the optimiser.

# %% [markdown]
# ---
# ## 6. Quantitative evaluation
#
# Test-year metrics in µg/m³, mean ± standard deviation across the three seeds.
# Baselines are deterministic and show a single number. The seed spread is the
# first thing to compare any difference against: **when two configurations
# differ by less than their spread, the honest word is "comparable"**, and the
# smaller number is not bolded anywhere in this notebook.
#
# WAPE is reported instead of MAPE, which is unstable near zero and lets a few
# clean hours dominate. Bias is the mean signed error — negative means the
# model under-forecasts.

# %%
table(report.metrics_table(RUNS, "mae"), "MAE by horizon (µg/m³), mean ± sd over seeds")
table(report.metrics_table(RUNS, "rmse"), "RMSE by horizon (µg/m³)")
table(report.metrics_table(RUNS, "wape"), "WAPE by horizon (%)")
table(report.bias_by_horizon(RUNS), "Bias by horizon (µg/m³): negative = under-forecast")

# %% [markdown]
# ---
# ## 7. Baseline comparison
#
# Two questions, answered two ways.
#
# **Was any of this needed?** Skill against persistence — the percentage MAE
# improvement over repeating the last observation. Near zero at h=1 is
# expected: with 0.969 one-hour autocorrelation there is almost nothing left to
# add, and the persistence anchor makes every neural model start from exactly
# that forecast.
#
# **Does the neural model beat the classical bar?** A paired weekly block
# bootstrap of each configuration against LightGBM, on identical rows. Weekly
# blocks because sliding windows overlap and PM2.5 decorrelates over ~2 days;
# paired because the shared week-to-week variance cancels and differences
# resolve far below the ±12.6 % that absolute MAE carries. A verdict is
# **robust** only when every seed agrees in sign *and* every seed's interval
# excludes zero.

# %%
table(report.skill_vs_persistence(RUNS), "Skill vs persistence (% MAE improvement; negative = worse than persistence)")
table(
    report.paired_vs_reference(RUNS, "lightgbm", seeds=SEEDS),
    "Paired bootstrap against LightGBM (positive = configuration beats LightGBM)",
)
table(
    report.paired_vs_reference(RUNS, "persistence", seeds=SEEDS),
    "Paired bootstrap against persistence (positive = configuration beats persistence)",
)

# %%
figure("error_by_horizon_baselines_mae", "Baselines only: MAE against horizon.")
figure("error_by_horizon_grid_mae", "The grid against horizon, with persistence as the floor and LightGBM as the bar to clear. Bands are the seed range.")

# %%
# --- the grid's own paired comparisons (the graph, the fusion, adding a graph)
# Each pair isolates exactly one component. This is the headline table of the
# architecture question, printed by the same code that ran the grid.
subprocess.run(
    [sys.executable, "scripts/run_grid.py", "--report-only", "--seeds", *map(str, SEEDS)],
    check=True,
)

# %% [markdown]
# ---
# ## 8. Horizon-wise analysis
#
# Error grows with horizon for every model; the question is the *shape*. Read
# the figure for three things: whether the neural curves sit below persistence
# at all horizons (they should, after h=1), whether they cross LightGBM anywhere,
# and whether the seed bands of different configurations overlap (if they do,
# the configurations are not distinguishable at that horizon).
#
# The h=1 column is the persistence anchor's test: every neural model should sit
# within a few percent of persistence there. The first full run, before the
# anchor, was 60–100 % worse than persistence at h=1 because patch pooling
# diluted exactly the quantity a one-hour forecast depends on.

# %%
figure("error_by_horizon_grid_rmse", "RMSE against horizon for the grid. RMSE weights episodes more heavily than MAE; if the ranking changes between the two, the models differ on peaks.")
skill = report.skill_vs_persistence(RUNS)
if not skill.empty:
    grid_rows = [n for n in report.GRID if n in skill.index]
    table(skill.loc[grid_rows], "Skill vs persistence by horizon, grid only")

# %% [markdown]
# ---
# ## 9. Station-wise analysis
#
# Per-station MAE for the full model at h=24, beside each station's own
# difficulty. A station is hard because its level is high or its record is
# gappy, not because the model singled it out; the columns for mean observed
# level and coverage let the reader tell those apart. The skill column is the
# per-station improvement over persistence.

# %%
station = report.station_table(dataset, RUNS, "d1_wind_crossview", horizon=24)
table(station, f"Per-station error at h=24 — {station.attrs.get('model', 'd1')} seed {station.attrs.get('seed', '')}")
figure("station_error_map", "Test MAE per station on the map, for the best run.")

# %% [markdown]
# ---
# ## 10. Dynamic graph analysis
#
# The physical claim is that the wind graph *changes with the wind*. Four checks:
#
# 1. **Direction.** The prior is built so that a station downwind of a source
#    receives an edge from it. The direction-check figure draws the graph under
#    two opposite synthetic winds; the edges must flip. This is the visual form
#    of the transpose test in `tests/test_graph_physics.py`.
# 2. **A real clear-out.** The strongest clear-out of the test year, drawn hour
#    by hour: the graph should align with the front as it passes.
# 3. **Learned physics.** `prior_strength` and `decay_rate` start at 1.0. How far
#    they moved is the difference between "the graph trained" and "the graph
#    sat at its initialisation".
# 4. **Calm vs windy.** If the adjacency in a 6 m/s hour looks like the adjacency
#    in a 0.5 m/s hour, the graph is static in practice whatever it is on paper.

# %%
figure("graph_direction_check", "The same network under two opposite synthetic winds. Edges must reverse; if they did not, the adjacency would be transposed.")
figure("graph_clearout_2016_03_04", "The 4 March 2016 clear-out, hour by hour: wind arrows, PM2.5 colour, and the graph edges the prior builds at each hour.")

# %%
table(report.graph_summary(RUNS), "Learned graph physics from the diagnostics pass (both parameters initialise at 1.0)")
diagnostics = report.load_diagnostics(RUNS, "d1_wind_crossview")
if diagnostics is None:
    print("(no diagnostics.npz -- the interpretability pass in section 4 did not run)")
else:
    print(f"diagnostics from {diagnostics['_run']}: {len(diagnostics['origins'])} sampled windows, stratified by wind speed")
    table(report.adjacency_by_regime(diagnostics), "The graph in calm versus windy windows")

# %% [markdown]
# **Reading it.** The mean transport lag reported by the graph should sit near
# the 2 h advection peak measured independently in the evidence review — that
# is the graph reproducing a fact about the data it was never told. A prior
# strength well below 1.0 means the model chose to discount the physics; a decay
# rate well below 1.0 means it shortened the effective range. Both are findings,
# not failures.

# %% [markdown]
# ---
# ## 11. Attention / fusion analysis
#
# **Attention by lag** is the most direct test of the lag-aware design. Every
# source station at every lag is a separate key, so the model is free to put
# its attention anywhere from "same hour" to "six hours ago". The evidence
# review measured real station-to-station transport peaking at a 2 h lag. If
# the learned attention lands there, the model is using the graph as transport;
# if it piles up at lag 0 or 1, it is using it as a same-hour association. Note
# the last bar: every edge whose travel time exceeds `lag_max` is clamped onto
# it, so a tall bar at 6 h is the slow-wind edges collecting, not a preference.
#
# **The fusion gate** sits between the two branches, per channel. A gate pinned
# at 0.5 never learned to choose; a gate with real spread and channels at both
# extremes is doing exactly what it was designed to do.

# %%
if diagnostics is not None:
    by_lag = report.attention_by_lag(diagnostics)
    table(by_lag, f"Share of cross-station attention by transport lag (learned peak at {by_lag.attrs.get('peak_lag', '?')} h; measured advection peak at 2 h)")
    figure(f"attention_by_lag_{diagnostics['_run']}")
    figure(f"attention_{diagnostics['_run']}", "Station-to-station attention in a windy window beside a calm one.")
    table(report.gate_summary(diagnostics), "Cross-view gate: 0 = temporal, 1 = spatial, 0.5 = initialisation")
    figure(f"gate_{diagnostics['_run']}", "Gate value by patch position (with spread) and its distribution.")
else:
    print("(no diagnostics -- attention and gate analysis skipped)")

# %% [markdown]
# ---
# ## 12. Error / failure analysis
#
# Three cuts, in order of what they teach.
#
# **Where the error lives.** MAE and bias inside quintiles of the *observed*
# concentration. A model can post a fine overall MAE and still be useless on the
# top quintile, because the top quintile is where every model regresses toward
# the mean — positive bias on clean hours, strongly negative on the worst ones.
# That is what an early-warning system is judged on, and a headline MAE hides it
# completely. Compare the models on the rightmost bar.
#
# **When it failed.** The worst test weeks for the full model, with what was
# happening. Almost always an episode.
#
# **Does the graph matter at all?** The negative controls. The head is
# persistence-anchored, so no control can move more than the learned correction
# on top of persistence; `zero_correction` measures that correction (the
# denominator) and `history_shuffle` how much of it any input perturbation can
# reach (the ceiling). Every other row is a fraction of those. If wind reversal,
# edge permutation and identity adjacency all read "no effect" while the
# references do not, the graph is demonstrably inert on this network.

# %%
table(report.error_by_level(RUNS, "d1_wind_crossview", horizon=24), "Full model at h=24: error inside quintiles of observed PM2.5 (seed 42)")
figure("error_by_level", "MAE and bias by observed quintile at h=24 for persistence, LightGBM and the full model.")
table(report.top_decile_table(RUNS, horizon=24), "Top-decile (worst 10 % of observed hours) MAE and bias at h=24, averaged over seeds")

# %%
table(report.worst_weeks(dataset, RUNS, "d1_wind_crossview", horizon=24, n=6), "The six worst test weeks for the full model at h=24")
figure("severe_2017_01_01", "New Year 2017 — the worst episode of the test year. Network-mean forecast at h=24 against observed.")
figure("fireworks_2017_01_28", "Chinese New Year 2017 — a fireworks spike no feature set can anticipate. Shown deliberately: it is the concrete justification for Huber loss.")
figure("clearout_2016_03_04", "The 4 March 2016 clear-out — 369 → 56 µg/m³ in 12 h. The case the wind graph was designed for.")
figure("regime_slices", "Improvement split by wind-speed bin, for the graph comparison. The physics claim, as evidence.")

# %%
controls = report.controls_table(RUNS, "d1_wind_crossview")
table(
    controls,
    "Negative controls on the full model, across seeds (positive damage = perturbation made the forecast worse)",
    note="controls.json not found; the control cell in section 4 did not run",
)
if controls is not None:
    ref = controls.xs("zero_correction", level="control") if "zero_correction" in controls.index.get_level_values("control") else None
    if ref is not None and 24 in ref.index:
        denominator = float(ref.loc[24, "damage"])
        print(f"\nAt h=24 the learned correction is worth {denominator:.2f} ug/m3 over the persistence anchor.")
        for name in ("wind_reversal", "edge_permutation", "identity_adjacency", "wind_reversal_full"):
            if (name, 24) in controls.index:
                damage = float(controls.loc[(name, 24), "damage"])
                share = 100.0 * damage / denominator if abs(denominator) > 1e-9 else float("nan")
                print(f"  {name:22s} {damage:+.3f}  ({share:6.1f} % of it)   detected {controls.loc[(name, 24), 'detected']}")

# %% [markdown]
# ---
# ## 13. Conclusions
#
# The cell below assembles the headline numbers from this run. The framing that
# follows is written to be true of the design regardless of how one run lands;
# read the numbers first, then the framing, and disagree with the framing if the
# numbers do.

# %%
table(report.headline(RUNS), "Headline MAE (µg/m³) by horizon")

lgbm = report.paired_vs_reference(RUNS, "lightgbm", seeds=SEEDS)
if not lgbm.empty:
    print("\nAgainst LightGBM, paired weekly block bootstrap:")
    for name in report.GRID:
        if name in lgbm.index.get_level_values("candidate"):
            sub = lgbm.loc[name]
            wins = int((sub["diff (ug/m3)"] > 0).sum())
            robust = int((sub["verdict"] == "robust").sum())
            print(f"  {name:22s} better at {wins}/4 horizons; {robust}/4 verdicts robust")

controls = report.controls_table(RUNS, "d1_wind_crossview")
if controls is not None:
    detected = [
        name for name in ("wind_reversal", "edge_permutation", "identity_adjacency")
        if (name, 24) in controls.index and controls.loc[(name, 24), "detected"].split("/")[0] != "0"
    ]
    print(f"\nGraph controls with any detected effect at h=24: {detected or 'none'}")

# %% [markdown]
# ### What this experiment establishes
#
# **The system works end to end**, on free hardware, in about 1.5 GPU-hours:
# raw CSVs to a scored, controlled, figured experiment, with 180+ tests guarding
# the parts that fail silently (windowing, leakage, wind direction, time zones).
#
# **The central hypothesis was measured before it was built, and the
# measurement held.** Beijing's 12 stations sit in a 27 km cluster with 0.887
# cross-station correlation. The evidence review estimated the wind graph's
# headroom over a plain network average at under 1 % MAE; the previous full grid
# measured approximately zero, with no pairwise comparison — the graph, the
# fusion, or adding a graph at all — resolving in either direction. Whatever
# this run's numbers say in §7, read them against that prior.
#
# **The machinery works even where the signal does not.** The graph's learned
# physics moved well away from initialisation, the fusion gate developed real
# per-channel structure, and the learned transport lag matched the independently
# measured 2 h advection peak. The negative controls in §12 are what turn "no
# improvement" into "the graph is demonstrably not carrying signal here".
#
# **LightGBM is the bar, and it is an honest one.** On this dataset gradient
# boosting is the known strong baseline. If §7 shows it winning robustly at most
# horizons, that is a finding to report, not a bug to tune away — and §12's
# error-by-level panel shows *where* it wins: the clean-to-moderate range, while
# regressing to the mean hardest on the severe hours that early warning is for.
#
# ### What it does not establish
#
# - Anything about a *larger* network. The failure here is a scale mismatch;
#   Delhi-NCR, where stations are far more spread out, is the network on which
#   the hypothesis has a genuine chance and has not yet been run.
# - Fine distinctions between configurations. One test year cannot resolve
#   differences of the size involved, and validation cannot rank them. Any
#   "best configuration" claim finer than the seed spread would be noise.
#
# ### Reproducing this
#
# Every number above is regenerable from the bundle written by the next cell:
# `python scripts/run_grid.py --report-only`, `python scripts/make_figures.py`,
# and the `src.report` calls in this notebook, run against `experiments/runs`.

# %%
# --- package the results for download ----------------------------------------
# Everything needed to reproduce, audit and re-plot: predictions, diagnostics,
# metrics, curves, configs, control results, checkpoints and environment
# snapshots. Checkpoints are ~0.3 MB each at this capacity and are what make
# the controls re-runnable locally; an earlier run excluded them and stranded
# that experiment.
import tarfile

OUTPUT = "/kaggle/working/sdgt_results.tar.gz" if Path("/kaggle").exists() else "sdgt_results.tar.gz"
by_suffix = {}
with tarfile.open(OUTPUT, "w:gz") as archive:
    for directory in ("experiments/runs", "experiments/figures"):
        for path in Path(directory).rglob("*"):
            if not path.is_file():
                continue
            archive.add(path, arcname=str(path))
            by_suffix[path.suffix] = by_suffix.get(path.suffix, 0) + path.stat().st_size

print(f"wrote {OUTPUT} ({Path(OUTPUT).stat().st_size / 1e6:.1f} MB compressed)")
for suffix, total in sorted(by_suffix.items(), key=lambda kv: -kv[1]):
    print(f"  {suffix or '(none)':12s} {total / 1e6:8.1f} MB uncompressed")
print("Download it from the Output tab, then unpack into the repo locally:")
print("  tar xzf sdgt_results.tar.gz")
