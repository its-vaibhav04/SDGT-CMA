# SDGT-CMA — project status

**Spatio-Directional Graph Transformer with Cross-Modal Attention for multi-site PM2.5 forecasting.**

| | |
|---|---|
| **Team** | Arpita Sharma, Vaibhav Tyagi, Ayesha |
| **Supervisor** | Mr. Sushil Kumar, MSIT Delhi |
| **Window** | Aug – Nov 2026 |
| **This document** | 14 September 2026 |
| **Repository** | 18 commits · 59 Python files · ~11,400 lines · **221 tests passing** |
| **One-line status** | **The system works end to end. The core scientific result is a well-supported null, and the experiment that makes it publishable is built but not yet run.** |

> This is the single place to look for where the project stands. Deep detail lives
> in the linked documents; nothing here contradicts them.

---

## 1. What the project claims to do

Forecast PM2.5 at every station in a monitoring network, 1–24 hours ahead, using a
dual-branch model:

- a **temporal branch** — a multivariate patch transformer over each station's history;
- a **spatial branch** — attention over *(source station, transport lag)* pairs, on a
  graph whose edges are built from wind direction and travel time;
- a **cross-view fusion** that gates between the two.

The central hypothesis: **pollution is advected between stations by wind, so a
wind-directed, lag-aware graph should forecast better than a distance graph or no
graph at all.**

---

## 2. Where things stand — phase by phase

| Phase | State | Evidence |
|---|---|---|
| 0 Environment and repo repair | **done** | pinned deps, seeding, manifests, UTF-8 coordinates |
| 1 Data contract | **done** | builder, transforms, windows, ERA5 hook · 27 tests |
| 2 Baselines, metrics, vertical slice | **done — G1 passed** | one command, raw CSV → plotted forecast |
| 3 Temporal branch | **done** | patch transformer + TCN control, head, masked losses |
| 4 Graph modules | **done** | static + lag-aware wind prior · 20 physics tests |
| 5 Fusion and assembly | **done** | cross-view + parameter-matched concat; all 5 configs train |
| 6 Experiment grid | **done — two full GPU runs** | Run 1 overfitted; **Run 2 is the reportable one** |
| 7 Figures | **done** | 18 figures, incl. training curves, attention-by-lag, error-by-concentration |
| **8 Delhi** | **on hold** | decision 9 Sept; CPCB API down. **Route found — see §7** |
| **9 Controls** | **built, not yet run** | code + 25 tests + docs; needs one GPU session |
| ERA5 / BLH | **done — in the dataset** | 28 features, UTC→local fix, physics verified (§3.5) |
| **Notebook narrative** | **done** (14 Sept) | 13-section Kaggle notebook, mentor's requested structure; analysis in tested `src/report.py` |
| 10 Write-up | not started | the notebook's 13 sections are its outline |

---

## 3. What has been achieved

### 3.1 A working system

The full pipeline runs end to end on a **free Kaggle GPU in 49 minutes**: raw CSVs →
processed dataset → 7 baselines → 15 grid runs (5 configs × 3 seeds) → diagnostics →
18 figures → a downloadable results bundle. Nothing crashes, no leakage, 221 tests
pass.

That is the thing the evaluation rewards most, and it is finished.

### 3.2 Measurement before implementation

Before a line of the model was written, `analysis/` measured the dataset and
`SDGT-CMA_Evidence_Review.md` recorded the results. The headline numbers:

| measurement | value | consequence |
|---|---:|---|
| Mean cross-station PM2.5 correlation | **0.887** | the 12 stations are nearly one signal |
| Mean pairwise station distance | **27 km** | too compact for station-to-station transport |
| Directional transport signal | peaks at **2 h** lag | matches 27 km at 1.4 m/s |
| Hours below 1.5 m/s | **52 %** | the graph is calm half the time |
| PM2.5 autocorrelation at 1 h | **0.969** | persistence is very hard to beat short-term |
| Wind-graph gain over a plain network average | **+0.26 to +0.65 % MAE** | the central novelty is worth under 1 % |
| Independent weeks in the test year | **53** | absolute MAE carries a **±12.6 %** interval |

**This is the project's strongest asset.** The last two rows predicted the eventual
result before it happened, which is what makes the null result a finding rather than
a disappointment.

### 3.3 Two runs, two diagnosed defects, both fixed

| | Run 1 | Run 2 |
|---|---|---|
| Train/val loss ratio | 3.0 – 3.9 | **1.13 – 1.57** |
| Best epoch | 0–5 of 11–15 | **0–13 of 13–26** |
| Parameters | 340,572 | **62,776 – 75,676** |
| MAE at h=1 | 16.68 – 21.43 | **10.51 – 10.66** |

Run 1 overfitted everywhere (340k parameters against ~243 independent training
episodes) and collapsed at h=1. Both causes were diagnosed and fixed: capacity cut
5×, warmup removed, and a **persistence-anchored head** (`forecast = y_t + head(...)`)
that makes persistence the model's zero-output. See
[Run 1 findings](SDGT-CMA_Run1_Findings.md).

### 3.4 The Run 2 result

Test MAE in µg/m³, Beijing, test year 2016-03 → 2017-02, mean over seeds 42/43/44:

| model | h=1 | h=6 | h=12 | h=24 |
|---|---:|---:|---:|---:|
| persistence | 10.30 | 32.03 | 44.34 | 58.13 |
| **LightGBM** | **9.35** | **28.19** | **39.02** | **50.82** |
| **t0** temporal only | **10.51** | **30.83** | **42.19** | **53.09** |
| s0 static + concat | 10.65 | 31.49 | 42.70 | 54.11 |
| d0 wind + concat | 10.66 | 31.37 | 42.74 | 54.24 |
| s1 static + cross-view | 10.60 | 31.43 | 43.51 | 53.88 |
| d1 wind + cross-view | 10.56 | 31.38 | 43.11 | 53.58 |

**Three findings, in order of importance:**

1. **No component of the spatial architecture produces a robust improvement.** Every
   paired comparison — the graph, the fusion, adding a graph at all — comes back
   `comparable` or `seed-dependent`. `t0`, the model with no graph, is the best
   neural configuration at every horizon and the most seed-stable.

2. **It is not a training failure**, which is what makes it reportable. The graph
   physics adapted (`prior_strength` 1.00 → **0.61**, `decay_rate` 1.00 → **0.60**),
   the fusion gate developed real per-channel structure (std 0.088 → **0.194**, range
   0.04 → 0.98), and the learned mean transport lag of **2.91 h** matches the 2 h
   advection peak measured independently. *The machinery works; the signal is not there.*

3. **One year of test data cannot resolve differences this small.** `best_val_mae`
   spans 0.91 µg/m³ across the whole grid while test h=24 spans 5.04, and
   `corr(best_epoch, test h=24) = +0.656` — longer training scored worse on test while
   validation called it fine. Validation is roughly **4× coarser** than the effects it
   is being asked to rank.

Full analysis: [Run 2 findings](SDGT-CMA_Run2_Findings.md).

### 3.5 Boundary layer height, and a timezone bug worth reporting

ERA5 boundary layer height is now fetched, verified and **in the built dataset** —
the "winter temperature inversion" mechanism the synopsis introduction names but
the original feature set never modelled. 12 stations, 35,064 hours, ~10 minutes
through the Copernicus API. The dataset now carries **28 features**, `blh` at
index 13; the contract validates clean and all 182 tests pass.

**The first fetch was wrong, and wrong in a way nothing downstream could catch.**
ERA5 is UTC; the Beijing PRSA timestamps are China Standard Time (UTC+8). The
values were physically plausible and the array was exactly the right shape — but
mean BLH peaked at "06:00" and bottomed at "18:00", which is the diurnal boundary
layer cycle upside down. A feature that is merely *offset in time* is still a
perfectly well-formed column, so no shape check, no finiteness check and no test
of the model could have found it. Only plotting the physics did.

After the +8 h correction:

| | before (UTC) | after (+8 h) |
|---|---|---|
| Peak of mean BLH | 06:00 | **15:00 local** |
| Minimum | 18:00 | **04:00 local** |
| corr(log BLH, log PM2.5), all year | −0.117 | **−0.258** |
| corr(log BLH, log PM2.5), **winter** | −0.282 | **−0.460** |

Winter PM2.5 by BLH quintile, which is the inversion mechanism made visible:

| BLH | mean PM2.5 |
|---|---:|
| 10 – 27 m | **134.6** |
| 27 – 51 m | 100.6 |
| 51 – 186 m | 85.0 |
| 186 – 817 m | 101.0 |
| 817 – 3521 m | **56.3** |

A shallow boundary layer traps emissions in a thin volume and PM2.5 more than
doubles. This is a good figure for the thesis and a genuinely physical result.

The fix is now a parameter (`utc_offset_hours`), recorded in the cache metadata,
and pinned by a test that traces each returned value back to the UTC hour it came
from. It refuses non-integer offsets outright rather than rounding — Delhi is
UTC+5:30, and silently rounding to +5 or +6 would reintroduce the same class of
bug.

### 3.6 The notebook tells the whole story

Requested by the supervisor at the 14 September meeting: the Kaggle notebook
now runs as a thirteen-section narrative — configuration, dataset/split,
architecture, training, training diagnostics, quantitative evaluation, baseline
comparison, horizon-wise, station-wise, dynamic graph, attention/fusion,
error/failure, conclusions. Every analysis cell is one call into
`src/report.py` (29 tests) or `src/figures`, so the tables the mentor reads are
computed by tested code rather than notebook-local pandas.

Three analyses were added that did not exist before and each surfaced something:

- **Paired bootstrap against LightGBM**, not just absolute MAE. On Run 2 it shows
  LightGBM beating *every* neural configuration **robustly** at h=1, 6 and 12 —
  all three seeds' intervals exclude zero. A stronger, more honest statement
  than "LightGBM has the lower number".
- **Attention by transport lag.** Learned peak at 1 h with 2 h close behind
  (20.5 % vs 17.8 %), against a measured advection peak at 2 h; and a pile-up at
  lag 6 that is the `lag_max` clamp collecting slow-wind edges, not a preference.
- **Error by observed concentration.** LightGBM's headline win comes from the
  clean-to-moderate quintiles; on the severe quintile (≥122 µg/m³) it is the
  *worst* model, with bias −115 — it regresses to the mean hardest exactly
  where early warning matters, and the full model under-forecasts less there
  (bias −95). That nuance was invisible in the headline table.

Also new: training curves (the figure Run 1 needed), skill vs persistence,
per-station skill, worst-week listing, and the controls read inline.

### 3.7 Honest framing, built in

The project refuses to overclaim by construction, not by discipline:

- Absolute MAE is never used to rank models — a **paired weekly block bootstrap**
  is, with two independent variances (test-set sampling and seed).
- A difference is called `robust` only when every seed agrees in sign *and* every
  seed's interval excludes zero.
- Run 1's one "robust" finding was **retracted** by Run 2 once the models converged.
- A physically wrong ERA5 alignment was caught by checking the diurnal cycle, not by
  any shape or range check (§3.5).
- The control suite refuses to print interpretations when its own reference
  denominator is not sound.

---

## 4. What remains

### 4.1 Run the negative controls — *the highest-value remaining experiment*

Built, tested, documented, wired into the notebook. Needs **one GPU session,
well under an hour**.

The Run 2 result is currently a *failure to find an effect*. The controls convert it
into a positive statement about the data: reverse the wind the graph reads, relabel
every edge's source, cut every cross-station edge, occlude stations one at a time —
and see whether the forecast notices. Because the components demonstrably trained
(§3.4), a null control now genuinely discriminates.

The suite carries its own **power references** so "no effect" can be told apart from
"no power": `zero_correction` measures what the learned correction is worth,
`history_shuffle` measures how much of it any perturbation can reach. See
[docs/NEGATIVE_CONTROLS.md](docs/NEGATIVE_CONTROLS.md).

### 4.2 Re-run the grid with boundary layer height

ERA5 is fetched, corrected and verified (§3.5). BLH takes the feature count
**27 → 28**, which changes the embedding's input width, so **every existing
checkpoint is invalidated** and the grid, baselines and controls must be recomputed
together. About **1.5 GPU-hours**.

This is the one genuinely new piece of physics added since Run 2, and it is the only
change with a plausible route to beating the null: BLH drives *when* pollution
accumulates, which is a city-scale mechanism the evidence review showed is where the
signal actually lives — unlike the station-to-station transport the graph models.

**Do this in the same session as the controls**, so everything is computed on the
final feature set.

### 4.3 Delhi

On hold; a viable route has been found — see §7.

### 4.4 Write-up

Not started. The structure is already implicit in the documents: evidence review →
implementation → Run 1 → Run 2 → controls → limitations.

### 4.5 Explicitly *not* doing

| | why |
|---|---|
| Capacity sweep (d=24/32/48) | It ranks by validation MAE, which §3.4 shows is ~4× coarser than the differences involved. Two GPU-hours for a table nobody can defend. |
| Lookback sweep, TCN control | Same resolution ceiling. They will return "comparable" everywhere. |
| Chasing LightGBM | A 2.3 µg/m³ gap where gradient boosting is the known strong baseline is a finding to report, not a bug to fix. |
| Switching to KnowAir | Out of scope — the synopsis is locked. |

---

## 5. Decisions taken

| date | decision |
|---|---|
| 4 Sept | **Synopsis is locked** — Beijing Multi-Site + CPCB Delhi-NCR. KnowAir out of scope. |
| 4 Sept | Evaluation rewards **a working system and good figures** over statistical rigour for its own sake. |
| 4 Sept | Grid fixed at **5 configs × 3 seeds**; ERA5 BLH added for Beijing as well as Delhi. |
| 8 Sept | Run 1 defects → capacity cut 5×, warmup removed, **persistence-anchored head**. |
| 9 Sept | **Capacity sweep cancelled** — validation cannot resolve it. |
| 9 Sept | **Delhi deferred** (CPCB API down), controls and write-up first. |
| 9 Sept | **ERA5 approved and enabled** — supersedes the earlier deferral. |
| 14 Sept | **Notebook restructured** into the 13-section narrative the supervisor asked for; BLH cache committed so Kaggle builds 28 features without a key. |

---

## 6. Known risks and open issues

| issue | severity | state |
|---|---|---|
| **Git push blocked** | **high** | This machine authenticates as `hyperreal2005`; the repos belong to `its-vaibhav04`. **All 13 commits are local only.** Needs a collaborator invite. |
| BLH invalidates checkpoints | medium | Understood and planned — do it with the controls in one session (§4.2). |
| CPCB API down | medium | Worked around — see §7. |
| One test year limits resolution | medium | Inherent to the dataset. Documented, and it bounds every claim. |
| Copernicus key was pasted in chat | low | Written to `~/.cdsapirc`, outside the repo, gitignored. **Worth rotating** once fetching is done. |

---

## 7. Delhi — the CPCB API is not the only route

The CPCB CCR portal API being down does **not** block Delhi, because the project needs
a *historical archive*, not a live feed. Three sources were checked:

| source | verdict |
|---|---|
| **OpenCity India** (39 Delhi stations, 2017–2023) | **Rejected.** The files are *AQI*, not PM2.5 in µg/m³ — AQI is a max over pollutant sub-indices and is not invertible. The sample file was also malformed. |
| **OpenAQ S3 archive** | **Verified and recommended.** |
| **Kaggle `rohanrao/air-quality-data-in-india`** | **Promising, one thing unverified.** |

### OpenAQ — verified working

Public, anonymous S3 at `s3://openaq-data-archive/` (no AWS account). I downloaded a
record file and confirmed the schema directly:

```
location_id, sensors_id, location, datetime, lat, lon, parameter, units, value
1,1,JTC - Jamestown-1,2006-11-13T07:13:00+00:00,5.5349194,-0.2124056,pm10,µg/m³,187.21
1,2,JTC - Jamestown-1,2006-11-13T07:13:00+00:00,5.5349194,-0.2124056,pm25,µg/m³,104.8
```

- **`pm25` in µg/m³** — the right quantity, not AQI.
- **`lat`/`lon` in every row** — station coordinates come with the data, so nothing
  has to be recalled or hand-compiled. This matters: a wrong coordinate silently
  corrupts the entire graph.
- Layout `records/csv.gz/locationid=<id>/year=<Y>/month=<M>/`, hourly, India ingested
  since 2015. Source is CPCB itself, so provenance is defensible.
- Remaining step: get Delhi's `location_id`s, which needs a free OpenAQ v3 API key.

### Kaggle mirror — convenient, needs one check

`rohanrao/air-quality-data-in-india` (2015–2020) has `station_hour.csv` with exactly
the pollutant set the Beijing builder already handles, plus `stations.csv`. Its big
advantage is that it **attaches directly to the Kaggle notebook as a dataset input** —
no download, no network dependency, same platform the grid already runs on.

**Unverified:** whether `stations.csv` actually carries latitude/longitude. Kaggle
renders client-side so I could not read the schema; a search summary claimed it does,
which is not good enough to rely on. **One minute of checking settles it** — and if it
does not, OpenAQ supplies coordinates anyway.

### Meteorology is already solved

The plan always specified ERA5 rather than CPCB's own sparse, gappy wind sensors, and
ERA5 now works. `u`/`v` components also sidestep the compass-convention problem
entirely.

**So Delhi is unblocked whenever you want to restart it** — no CPCB API required. The
geometry is also the reason it is worth doing: Delhi-NCR is far larger and more
heterogeneous than Beijing's 27 km cluster, so the wind-graph hypothesis has a genuine
chance there that it never had here.

---

## 8. How to run everything

```bash
# setup
python -m venv venv && venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# data (ERA5 BLH is fetched automatically if ~/.cdsapirc exists)
python -m src.data.build_beijing
pytest tests/ -q                       # 221 tests

# the full experiment, ~49 min on a free Kaggle GPU
python scripts/run_baselines.py
python scripts/run_grid.py --seeds 42 43 44
python -m src.evaluate --run experiments/runs/d1_wind_crossview_seed42
python scripts/run_controls.py --occlusion --runs experiments/runs/d1_wind_crossview_seed4{2,3,4}
python scripts/make_figures.py

# report only, from saved predictions
python scripts/run_grid.py --report-only --seeds 42 43 44
```

On Kaggle, `notebooks/kaggle_run_grid.ipynb` does all of the above in one
Save & Run All and packages the results for download.

---

## 9. Document map

| document | what is in it |
|---|---|
| [README.md](README.md) | orientation, setup, conventions, current numbers |
| **STATUS.md** | *this file* — where everything stands |
| [SDGT-CMA_Evidence_Review.md](SDGT-CMA_Evidence_Review.md) | every measurement taken before implementation |
| [SDGT-CMA_Change_Specification.md](SDGT-CMA_Change_Specification.md) | what changed from the original synopsis, and why |
| [SDGT-CMA_Implementation_Plan.md](SDGT-CMA_Implementation_Plan.md) | build order, data contract, phase gates |
| [SDGT-CMA_Run1_Findings.md](SDGT-CMA_Run1_Findings.md) | the overfitted run, its diagnosis, the two fixes |
| [SDGT-CMA_Run2_Findings.md](SDGT-CMA_Run2_Findings.md) | **the reportable result** |
| [docs/NEGATIVE_CONTROLS.md](docs/NEGATIVE_CONTROLS.md) | the control suite and how to read it |
| [docs/SEEDS_AND_RANDOMNESS.md](docs/SEEDS_AND_RANDOMNESS.md) | what a seed does and does not control |
| [analysis/](analysis/) | the frozen scripts behind the evidence review |

---

## 10. The honest summary

**The engineering succeeded and the hypothesis did not — and the project measured why
before it built anything, which is what turns that into a result.**

Beijing's 12 stations sit in a 27 km cluster with 0.887 cross-station correlation and
a 2-hour transport signal. There is very little station-to-station structure for a
wind graph to model, and the pre-registered estimate said so: under 1 % MAE headroom.
Run 2 measured approximately zero. Two independent methods, the same answer, with a
model whose graph physics and fusion gate demonstrably trained.

What is left to make that airtight is one GPU session: the controls, with boundary
layer height folded in. After that the thesis has a working system, a strong figure
set, a measured and mechanistically explained negative result, and a clear account of
what the dataset can and cannot support.
