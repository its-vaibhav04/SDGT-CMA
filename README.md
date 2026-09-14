# SDGT-CMA

Physics-guided dynamic graph forecasting for multi-site PM2.5.

A B.Tech project studying **where a wind-directed, lag-aware graph prior helps**
in short-to-medium range PM2.5 forecasting — not whether one more architecture
can be assembled. Beijing Multi-Site (12 stations, 2013–2017) is the primary
benchmark; CPCB Delhi-NCR is the robustness test.

## Read these first

| Document | What it is |
|---|---|
| [SDGT-CMA_Evidence_Review.md](SDGT-CMA_Evidence_Review.md) | What the data actually says. Every number measured, not quoted. |
| [SDGT-CMA_Change_Specification.md](SDGT-CMA_Change_Specification.md) | What changed from the original proposal, and why. |
| [SDGT-CMA_Implementation_Plan.md](SDGT-CMA_Implementation_Plan.md) | The build order, interfaces and gates. |
| [analysis/README.md](analysis/README.md) | The scripts behind every measurement. |
| [docs/SEEDS_AND_RANDOMNESS.md](docs/SEEDS_AND_RANDOMNESS.md) | What a seed controls, what it deliberately does not, and why a claim must clear two variances. |

Three findings shape the whole design, and are worth knowing before reading any code:

- **Wind matters enormously at city scale and barely at station scale.** Air from
  the NNW averages 22.6 µg/m³; from the ESE, 89.7. But station-to-station
  transport peaks at only +0.045 correlation advantage (2 h lag) and vanishes
  once the city-wide common factor is removed.
- **Spatial information helps at short horizons, not long ones.** +4.4 % at h=1,
  +0.76 % at h=24 — the opposite of the original proposal's claim.
- **One test year holds 53 independent weeks.** Absolute MAE carries a ±12.6 %
  confidence interval, so models cannot be ranked by headline number. Paired
  comparisons on identical rows resolve below 0.2 %.

## Current status

**For the full picture — achievements, findings, what remains, open risks — see
[STATUS.md](STATUS.md).** The table below is the short version.

**The Kaggle notebook is now the experiment's full narrative** — thirteen
sections from configuration to conclusions, every analysis cell a single call
into the tested `src/report.py` and `src/figures` modules. See
[notebooks/kaggle_run_grid.py](notebooks/kaggle_run_grid.py).

| Phase | State |
|---|---|
| 0 Environment and repo repair | done — git, pinned deps, UTF-8 coordinates, seeding, manifests |
| 1 Data contract | done — builder, transforms, windows, ERA5 hook, 27 tests |
| 2 Baselines, metrics, thin vertical slice | **done — G1 passed:** one command from raw CSV to a plotted forecast |
| 3 Temporal branch | done — patch transformer + TCN control, head, masked losses |
| 4 Graph modules | done — static + lag-aware wind prior, 20 physics tests incl. the direction gate |
| 5 Fusion and assembly | done — cross-view and parameter-matched concat, all five configs train |
| 6 Experiment grid | **done — two full GPU runs.** Run 1 overfitted; Run 2 converged and is the reportable one. See [Run 2 findings](SDGT-CMA_Run2_Findings.md). |
| 7 Figures | done — 18 figures, now including training curves, attention-by-lag and error-by-concentration. |
| 8 Delhi | **on hold** (decision, 9 Sept 2026). Beijing-only for now: run the controls and write up the null result first. |
| 9 Controls | **built, awaiting a GPU session.** Wind reversal, edge permutation, identity adjacency, station occlusion, plus two power references. See [docs/NEGATIVE_CONTROLS.md](docs/NEGATIVE_CONTROLS.md). |

221 tests passing. Headline result from Run 2, Beijing test year 2016-03 → 2017-02,
MAE in ug/m3 (mean over seeds 42/43/44):

| model | h=1 | h=6 | h=12 | h=24 |
|---|---:|---:|---:|---:|
| persistence | 10.30 | 32.03 | 44.34 | 58.13 |
| **LightGBM** | **9.35** | **28.19** | **39.02** | **50.82** |
| t0 temporal only | 10.51 | 30.83 | 42.19 | 53.09 |
| d1 wind graph + cross-view | 10.56 | 31.38 | 43.11 | 53.58 |

No component of the spatial architecture produces a robust improvement over the
purely temporal model on this network, and LightGBM remains the strongest
forecaster. The graph nonetheless trains and behaves physically (learned mean
transport lag 2.91 h against a measured 2 h advection peak). This is a measured
null result, consistent with the pre-registered power analysis, not a training
failure -- see [Run 2 findings](SDGT-CMA_Run2_Findings.md).

**Boundary layer height is now included** (9 Sept 2026). The dataset carries
**28 features**; ERA5 BLH sits at index 13, fetched for all 12 stations and
shifted from UTC onto Beijing local time. Without a Copernicus token the builder
still runs and records the feature's absence in the manifest.

**The Run 2 results below predate it.** They were computed on the 27-feature
dataset. Adding BLH changes the embedding's input width, so every Run 2
checkpoint now refuses to load, and the grid, baselines and controls have to be
recomputed together (~1.5 GPU-hours) before anything can be compared to them.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or cu121 for GPU
pip install -e ".[dev]"
pip install -e ".[era5]"          # optional: boundary layer height
```

**To enable ERA5 boundary layer height**, install the `era5` extra above and put a
Copernicus token in `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your key from https://cds.climate.copernicus.eu/how-to-api>
```

The builder then fetches and caches it automatically. Without the token the
pipeline runs unchanged and records the feature's absence in the manifest. Note
that enabling it changes the feature count from 27 to 28, so every existing
checkpoint is invalidated -- see the warning below.

## Pipeline

```bash
# 1. Build the processed dataset from the raw PRSA CSVs
python -m src.data.build_beijing

# 2. Verify the data contract (leakage, windowing, wind convention, masks)
pytest tests/ -q

# 3. Baselines: persistence, seasonal naive, climatology, ridge, DLinear, LightGBM
python scripts/run_baselines.py

# 4. Neural grid: 5 configurations x 3 seeds
python scripts/run_grid.py --seeds 42 43 44

# 5. Dump interpretability diagnostics from a trained run
python -m src.evaluate --run experiments/runs/d1_wind_crossview_seed42

# 6. Figures
python scripts/make_figures.py
```

On Kaggle, run the whole thing from the notebook instead:

```bash
python scripts/build_notebook.py     # notebooks/kaggle_run_grid.py -> .ipynb
```

Upload `notebooks/kaggle_run_grid.ipynb`, attach the Beijing dataset, set the
accelerator to GPU, and use **Save & Run All (Commit)** — a committed run keeps
going after you close the tab, an interactive session does not.

A wiring check that finishes in seconds rather than hours:

```bash
python scripts/run_grid.py --epochs 2 --smoke 192 --seeds 42
```

## The experiment grid

Each configuration differs from its neighbours in exactly one component, so every
pairwise difference is attributable.

| ID | Spatial | Fusion | Isolates |
|---|---|---|---|
| `t0` | none | none | is a graph needed at all? |
| `s0` | static distance | concat | conventional reference |
| `d0` | lag-aware wind | concat | **s0 → d0**: the graph, alone |
| `s1` | static distance | cross-view | **s0 → s1**: the fusion, alone |
| `d1` | lag-aware wind | cross-view | full model; **d0 → d1** re-tests fusion |

## Reference numbers

Test split 2016-03-01 → 2017-02-28, MAE in µg/m³, masked to observed targets.
Any model that does not clear these is not interesting yet.

| Model | h=1 | h=6 | h=12 | h=24 |
|---|---:|---:|---:|---:|
| persistence | 10.30 | 32.03 | 44.34 | 58.13 |
| seasonal naive (24 h) | 58.20 | 58.18 | 58.16 | 58.13 |
| climatology | 60.29 | 60.34 | 60.40 | 60.51 |
| ridge | 10.56 | 31.52 | 42.32 | 51.97 |
| DLinear | 10.26 | 32.58 | 44.16 | 53.97 |
| **LightGBM** | **9.35** | **28.19** | **39.02** | **50.82** |

LightGBM winning is expected, not a surprise: a published study on this exact
dataset reports the same. Beating it is the result worth writing up.

## Conventions that are not negotiable

Each of these, if broken, produces results that look fine and are wrong.

- **Adjacency is `[target, source]`.** Aggregation for a target sums messages
  from its sources. `tests/test_graph_physics.py` pins this with a synthetic
  westerly: a western source must influence an eastern target, never the reverse.
- **`wd` is the direction wind blows *from*.** Transport bearing is `wd + 180°`.
  Verified from the data, not assumed.
- **Targets are never imputed and never scored where missing.**
- **Metrics are always in µg/m³**, after inverting the scaling.
- **Every model is evaluated on identical rows**, and predictions are saved so
  any two runs are automatically paired.
- **The test split is read once**, after the protocol is frozen.
- **Seeds vary the model, never the data.** Split, origins, scaling and masks are
  identical across every run and every seed. See
  [docs/SEEDS_AND_RANDOMNESS.md](docs/SEEDS_AND_RANDOMNESS.md).

## Layout

```
analysis/     frozen evidence scripts (NumPy only, run before the pipeline exists)
configs/      base.yaml, one file per grid cell, and the station coordinate table
src/data/     the data contract: build, transform, validate, window
src/graphs/   geometry, static distance graph, lag-aware wind prior
src/models/   embedding, temporal, spatial, fusion, head, assembly
src/baselines/ naive, linear, trees
src/figures/  one module per figure
scripts/      run_baselines, run_grid, make_figures
tests/        repository completeness, data contract, graph physics, models, metrics
docs/archive/ superseded documents, kept for the record
```

## Compute

The whole grid fits in one free Kaggle account. The dataset is 45 MB and lives
entirely in GPU memory, so batching is pure indexing. Training is GPU-bound, not
data-bound — expect the CPU path to be roughly 30–50× slower and use it only for
wiring checks with `--smoke`.
