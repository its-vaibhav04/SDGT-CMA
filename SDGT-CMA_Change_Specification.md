# SDGT-CMA — Change Specification

**Exactly what changes, and what stays, before implementation begins.**

| | |
|---|---|
| **Date** | 4 September 2026 |
| **Evidence base** | `SDGT-CMA_Evidence_Review.md` — every claim below traces to a measurement there |
| **Supersedes** | Parts of `SDGT-CMA_Implementation_Roadmap.md` and `SDGT-CMA_Pre-Implementation_Audit.docx` (mapping in §9) |
| **Status** | Awaiting approval. Nothing here is implemented yet. |

---

## 0. Fixed decisions (confirmed, not up for revisit)

| Decision | Value | Consequence |
|---|---|---|
| **Datasets** | Beijing Multi-Site (primary) + CPCB Delhi-NCR (validation). **Locked by the submitted synopsis.** | KnowAir is out of scope. Do not expect or promise a large headline gain from the graph. |
| **What the evaluation rewards** | A working system and strong figures | Figures move to the front of the plan. Statistical protocol gets lighter, not optional. |
| **Compute** | Free Kaggle (30 GPU-h/week) — one account | No multi-account workarounds needed. |
| **Architecture family** | Dual-branch spatial + temporal, cross-attention fusion, as per synopsis | Kept. The corrections below are to *how* it is built, not *what* it is. |

---

## 1. Changes to the project's claims and framing

These change what you *write*, not what you *build*. They are the cheapest changes here and prevent the most damage in a viva.

| # | Current claim | Change to | Why |
|---|---|---|---|
| 1.1 | Advantage appears "particularly over longer forecast horizons" | **Advantage appears at short horizons (1–6 h) and in high-wind regimes; it decays to nothing by 24 h** | Measured: spatial value is +4.4 %/+5.5 %/+2.3 %/+0.75 % at h=1/6/12/24. The claim is reversed. |
| 1.2 | 168-hour window captures long-range/weekly periodicities | **Lookback is a swept hyperparameter over {24, 48, 96, 168}; report the sweep as a finding** | Lag-168 autocorrelation is 0.023. Error rises monotonically past ~24 h. |
| 1.3 | "Cross-Modal Attention" | **Cross-view attention** (or cross-branch / spatial-temporal co-attention) | Spatial and temporal are two views of one tabular data source, not two modalities. Keep SDGT-CMA as the project label; use the accurate term in the methodology. |
| 1.4 | "PatchTST channel-independent temporal encoder" | **Multivariate patch transformer** | The design flattens all variables within a patch and shares weights across stations. That is not PatchTST channel-independence. |
| 1.5 | Graph edges use "historical pollutant correlation" | **DECIDED: remove the claim.** The graph prior is distance + wind alignment + lag only. | Stated in the synopsis, absent from the roadmap formula and all code. Implementing it would add a training-data leakage surface for negligible gain. |
| 1.6 | Implicit "our model beats prior work" framing | **A physically-constrained dynamic graph and a controlled test of where it helps** | AST-GT (Aug 2026) already covers this combination on Beijing + India. Predict-then-confirm reads far better than a small number oversold. |
| 1.7 | Station-level meteorology assumed per-station | **State that the 12 stations map to 8 distinct wind-direction series and 11 temperature series** | Measured. The wind field's effective resolution is 8 nodes, not 12. This is a limitation that must be declared. |

---

## 2. Changes to the bibliography (mandatory — contains a non-existent reference)

| Ref | Action |
|---|---|
| **[6] AGATNet** | **Replace.** Correct record: Dimri, Choi, Salman, Park & Singh, *JGR: Machine Learning and Computation*, 2024, DOI `10.1029/2024JH000244`. It performs **CMAQ bias correction over South Korea** — a different task. Describe the difference; do not present it as a direct predecessor. The current "Z. Wang et al., IEEE TKDE 2022" does not resolve to anything. |
| **[9] AirQFormer** | **Correct details.** *Sustainable Cities and Society* **vol. 119, February 2025**, DOI `10.1016/j.scs.2024.106113`. Not vol. 106, 2024. |
| **[11] TransNet** | **Remove** unless a DOI matching the claimed title and record can be produced. No reliable match found. |
| **[13] DSGT** | **Replace.** Correct record: Xia, Chen, Chen & Hu, *Neurocomputing* vol. 616, 2025, DOI `10.1016/j.neucom.2024.128924`. Not Liu et al., ESWA 2024. |
| **[8] AirFormer** | **Rewrite the comparison-table row.** AirFormer uses efficient spatial and temporal self-attention in stacked blocks, not the "static GCN → temporal" pipeline the table asserts. |
| **[3] PM2.5-GNN** | **Expand the row.** It combines a knowledge graph with geographic and meteorological information plus recurrent temporal modelling — the one-line "wind-informed static graph" description loses this. |
| **NEW** | **Add AST-GT**: Shi, Yang & Qin, *J. Environ. Manage.*, Aug 2026, DOI `10.1016/j.jenvman.2026.130515`. Closest prior work; Beijing + India. Cite it and state your differences explicitly. |
| **NEW** | **Add DLinear**: *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023, DOI `10.1609/aaai.v37i9.26317`. Required baseline justification. |
| **DO NOT CITE** | "MAGICFormer" (`10.1016/j.asoc.2025.113033`) — it is a **subway station** paper, not ambient monitoring. And "AirFlow" (arXiv `2608.09775`) — **unverifiable, likely non-existent**. Both appear in the Codex audit. |

Also restructure the comparison table (synopsis §4.6) to distinguish **four** dimensions rather than one vague label per paper: graph construction, temporal operator, spatial-temporal interaction, and information available at forecast time.

---

## 3. Changes to preprocessing

| # | Current spec | Change to | Evidence |
|---|---|---|---|
| 3.1 | Z-score outlier removal (z=3) | **Delete nothing by Z-score.** Reject only physically impossible values. Flag instrument ceilings (`CO == 10000`, `PM == 999`) with a binary indicator column. Keep every high episode. | z=3 threshold is 322 µg/m³ and deletes 7,354 hours (1.78 %) — every one a Severe-AQI episode. Longest constant run in PM2.5 is 2 h, so there are no stuck sensors to catch. |
| 3.2 | Min-Max normalisation | **`log1p` then standardise.** Fit on training targets only. Save the transform. Invert before any reported metric. | Min-Max compresses 99 % of PM2.5 into `[0, 0.37]`. |
| 3.3 | Linear interpolation for short gaps | **Forward-fill inputs up to 3 h, then training-derived station-hour climatology.** Never interpolate across a forecast origin. | Interpolation across an origin uses future information. Gaps ≤3 h are 87 % of gaps but only 39.5 % of missing hours. |
| 3.4 | (absent) | **Add an observation mask and a time-since-observed channel per imputed variable.** | Required so the model can distinguish observed from filled values. |
| 3.5 | (absent) | **Add a target-validity mask; compute loss and metrics only where the true target exists.** | 2.08 % of PM2.5 is missing. Scoring imputed labels as truth silently inflates every result. |
| 3.6 | Wind direction cyclically encoded | **Keep both: `sin`/`cos` for the feature embedding, and raw degrees for the graph module.** Convert to transport bearing as `wd + 180°`. | `wd` is direction-from. Confirmed empirically: NNW → 22.6 µg/m³, ESE → 89.7. |
| 3.7 | 70/15/15 split | **Whole-season blocks by target time:** train 2013-03-01 → 2015-02-28, val 2015-03-01 → 2016-02-29, test 2016-03-01 → 2017-02-28. Freeze and reuse identically for every model. | Avoids splitting mid-season; gives one clean test year. |
| 3.8 | (absent) | **Add `dow_sin`/`dow_cos`.** | Weekday spread is 18.1 % of the mean (Mon 73.0 → Sat 87.5), comparable to hour-of-day. |
| 3.9 | (absent) | **DECIDED: add ERA5 boundary layer height to Beijing as well as Delhi.** Sample at each station's coordinates; `log1p` then standardise. Treat as a soft dependency — if the download fails, the pipeline runs without it and the feature is dropped from the manifest. | It is the strongest physical driver of winter accumulation — the exact "temperature inversion" mechanism the synopsis introduction names but never models. |

---

## 4. Changes to the architecture

| # | Component | Current spec | Change to | Evidence |
|---|---|---|---|---|
| 4.1 | **Graph indexing** | Edges defined as "i influences j" but aggregated row-i-from-columns-j | **Fix to `row = target`, `column = source`.** Aggregation for target *i* sums messages from sources *j*. | The two conventions are transposes. The bug is completely silent — nothing crashes, the model just learns anti-physics. |
| 4.2 | **Graph lag** | Same-hour edges | **Lag-aware.** Travel time = `dist / max(wind_speed, v_min)`, clipped to 1–6 h. Pull the source station's state from that earlier hour. | Directional signal peaks at **2 h lag**, gone by 12. |
| 4.3 | **Calm-wind handling** | `ReLU(align · v)` → self-loops only | **Fall back to the static distance graph below a speed threshold.** Log the fraction of hours each mode fires. | 52.0 % of hours are below 1.5 m/s. The graph collapses for half the dataset. |
| 4.4 | **Distance decay** | `1/exp(λ·dist)`, unconstrained λ | **`softplus(λ)`** so it cannot go negative. Normalise distance in km by a fixed scale. **Row-normalise** the prior. | An unconstrained λ can make edge weight grow exponentially with distance. |
| 4.5 | **Self-loops** | `A = A + I` | **Set self-edges explicitly** rather than adding to an unknown diagonal. Skip the undefined self-bearing. | Adding `I` can push diagonal weights above 1 if the diagonal already has a value. |
| 4.6 | **Spatial branch output** | GAT runs 168× per window; only the last hour retained | **Run the graph layer per hour, pool within each patch, keep all patch tokens** → `[B, N, Np, d]`. | Currently 167 of 168 graph computations are discarded, which contradicts the claim of modelling graph evolution. |
| 4.7 | **Temporal branch output** | Pooled to `[B, N, d]` before fusion | **Keep the patch axis** → `[B, N, Np, d]`. | Required for 4.8. |
| 4.8 | **Fusion** | Cross-attention over 12 station tokens | **Cross-view attention over retained patch tokens**, per station. If you *do* pool first, rename the module cross-station attention and drop all "temporally significant events" language. | With both branches pooled to one token per station, there is no time axis left to attend to. The stated mechanism does not exist in the current tensor design. |
| 4.9 | **Diagnostics** | (absent) | **Every model returns a diagnostics dict alongside its prediction — `A_dyn`, `α`, `g` — behind a flag** so training stays fast and evaluation dumps everything. | This is the one architectural decision that cannot be deferred. Retro-fitting it means re-running every experiment. See §6. |
| 4.10 | **Lookback `L`** | Fixed at 168 | **Swept hyperparameter over {24, 48, 96, 168}.** | Error rises monotonically past ~24 h in a controlled ridge sweep. Expect 24–48 to win. |
| 4.11 | **Model size** | unspecified | **Keep small: `d_model` 32–64, 2–3 encoder layers, ≈200 k params.** Dropout and edge dropout. | Only ~243 independent 3-day episodes in the training split. Capacity is the enemy here. |

**Unchanged and correct:** dual parallel branches, learnable gate fusion, MLP prediction head, Huber loss, direct multi-horizon output (all 24 hours at once, evaluated at 1/6/12/24 h), chronological split, train-only scaler fitting.

---

## 5. Changes to the experiment design

### 5.1 Replace the 3-model comparison with a 5-cell matrix

The roadmap's Model 1/2/3 cannot separate the graph effect from the fusion effect, because it never runs a dynamic graph with plain concatenation.

| ID | Spatial branch | Fusion | What the comparison isolates |
|---|---|---|---|
| T0 | none | none | Is a graph needed at all? |
| S0 | static distance | concatenation | Conventional reference point |
| D0 | lag-aware wind | concatenation | **S0 → D0**: the graph, alone |
| S1 | static distance | cross-view | **S0 → S1**: the fusion, alone |
| **D1** | **lag-aware wind** | **cross-view** | Full model; **D0 → D1** re-tests fusion |

Add one temporal-backbone ablation: replace the patch transformer with a TCN or LSTM while holding the spatial branch fixed. This tests whether the Transformer is necessary at all.

### 5.2 Baselines (none optional)

1. Persistence
2. Seasonal naive at 24 h and 168 h
3. Station-hour historical average
4. Ridge, direct multi-output
5. DLinear or NLinear
6. **Gradient-boosted trees (XGBoost / LightGBM)** — *not optional*; the published reference on this exact dataset reports XGBoost winning at 6 h. A thesis omitting it has a hole a viva will find.

### 5.3 Protocol

- **3 seeds** per configuration (not 5 — scaled to the evaluation criteria), report mean and spread.
- Identical split, preprocessing, target masks and tuning budget for every model. **Store predictions, not just scores.**
- Paired weekly block-bootstrap confidence interval on the **headline comparison only** (S0 vs D0).
- If two configurations differ by less than the seed spread, **write "comparable"** — do not pick a winner.
- Never report absolute MAE as if precise: its 95 % CI is ±12.7 % on this test set.

### 5.4 Metrics

- Primary: **MAE and RMSE in µg/m³**, per horizon and per station.
- Report R² as secondary only. If MAPE is retained to match prior work, state the zero-handling rule and a minimum denominator.
- **Add decision-relevant slices:** top-decile PM2.5 MAE, performance by wind-speed bin, by season, by station, and improvement over persistence per horizon.

### 5.5 Negative controls (one afternoon, protects every physical claim)

- **Reverse the wind direction.** If performance does not degrade, the graph was never using wind — and no attention heatmap will rescue the claim.
- Permute edges. Occlude individual stations. Feed an identity adjacency.

---

## 6. New work item — the figure set (design before building)

Since figures are what the evaluation rewards, design them first. Every one below falls out of a tensor the architecture already produces, **if** §4.9 is implemented.

| Priority | Figure | Concrete target | Requires |
|---|---|---|---|
| **1** | **Dynamic graph animated on a Beijing basemap** — edges from `A_dyn`, hour by hour | **2016-03-04**, strongest clear-out in the test year: 369.5 → 55.7 µg/m³ in 12 h at 3.04 m/s. The graph must visibly swing downwind. | `A_dyn` per hour |
| 2 | Pollution rose — 4× swing by wind direction | Raw data only; already computed (§3.1 of the evidence review) | nothing |
| 3 | Attention side-by-side — 12×12 `α` for a high-wind hour vs a calm hour | any test-set pair | `α` |
| 4 | Severe-episode forecast vs actual, persistence overlaid | **2017-01-01**: 24 h mean 443, peak 521.8 µg/m³ — worst in the test year | per-hour predictions |
| 5 | **The honest failure case** | **2017-01-28** — Chinese New Year, fireworks-driven peak of 607 µg/m³, unpredictable from the feature set. Show the miss and explain it. This is also the concrete justification for Huber loss. | per-hour predictions |
| 6 | Error vs horizon, one line per configuration, persistence as floor | — | metrics by horizon |
| 7 | Regime-sliced gains by wind-speed bin | Linear probe predicts larger gains in windy hours — if the trained model reproduces it, that is the physics claim evidenced | per-row errors + wind |
| 8 | Gate values `g` across horizons — spatial vs temporal reliance | — | `g` |
| 9 | Per-station error bubble map | Shows whether the graph helps peripheral stations (Huairou, Dingling) more than central ones | metrics by station |
| 10 | Delhi robustness panel — missingness heatmap + metric table | delivers the synopsis's "realistic operating conditions" claim | Delhi pipeline output |

---

## 7. New work item — Delhi / CPCB acquisition

CPCB is a committed deliverable. The portal is a live operational system with per-station, per-pollutant manual export and request-window limits.

### 7.1 Pollutants

- Use an unofficial Python client — `sakethramanujam/cpcbccr-python-client` or `gsidhu/cpcbccr-data-scraper`. **Expect to patch them**; they track a portal that changes. Budget ~2 days.
- Fallback if the client breaks: manual export for a reduced station set rather than losing the deliverable.
- **Freeze on download:** station IDs and coordinates, exact date range, download timestamp, units, missing-data codes, checksum per file. Commit the raw files. **Never re-download mid-project** — results would silently change.

### 7.2 Meteorology — take it from ERA5, not CPCB

CPCB's on-site wind sensors are sparse and heavily gapped, and the entire graph module depends on per-station wind.

- Register free on the Copernicus Climate Data Store. Use `reanalysis-era5-single-levels-timeseries`, which returns **point time series** — request ~40 station coordinates rather than downloading grids. Google Earth Engine also carries ERA5 hourly and is the lower-friction route from a Colab notebook.
- Pull `10m_u_component_of_wind` and `10m_v_component_of_wind`, plus temperature, dewpoint, precipitation, and **boundary layer height**.
- **Two side benefits:** u/v components give the transport vector directly, sidestepping the compass-conversion and direction-convention bug entirely; and boundary layer height is the strongest physical driver of winter PM2.5 accumulation.

### 7.3 Scope

Delhi is the **robustness demonstration** the synopsis promised, not a second full factorial. Train the best Beijing configuration from scratch on Delhi, report the same metric table plus a missingness panel, discuss where it degrades. One model, not five.

---

## 8. Changes to the repository

### 8.1 Current state

Everything except `src/data/phase1_build_dataset.py` and `src/utils/reproducibility.py` is a docstring placeholder. `src/models/**`, `src/train.py` and `src/evaluate.py` contain no implementation. **There is essentially nothing to throw away** — which is why this pivot is cheap.

### 8.2 Required repairs

| # | Issue | Fix |
|---|---|---|
| 8.1 | `data/raw/stations.csv` is CP-1252 with degree symbols in headers (`Longitude (°E)`); the loader assumes UTF-8 and can fail before the rename logic runs | Convert to UTF-8 with ASCII headers `station,lat,lon` |
| 8.2 | Not a git repository — no history, no immutable source identifier | `git init`, commit the audited starting point, record the commit hash in every run's metadata |
| 8.3 | `requirements.txt` unpinned; `pyproject.toml` declares no dependencies | Move dependencies into `pyproject.toml`, pin them, document supported Python and PyTorch versions |
| 8.4 | Docs say `.venv`, directory is `venv`; `.gitignore` covers only `.venv` | Standardise on one name, ignore both during transition |
| 8.5 | `set_seed` sets `PYTHONHASHSEED` inside a running process, which has no effect | Set it in the launcher, or document the limitation. Add deterministic DataLoader generators and worker init. Record the seed per run rather than hard-coding 42 in entry points. |
| 8.6 | No data manifest | Add one: filenames, sizes, checksums, source URLs, download dates, row counts, station order, date ranges |
| 8.7 | Tests check row count, schema and RNG repetition only | Add the gates in §8.3 |
| 8.8 | Configs describe the old 3-model design | Replace with the 5-cell matrix (§5.1) |
| 8.9 | The measurement scripts behind this review are not saved | **Open item** — see §11 |

### 8.3 Minimum test gates

| Gate | Test |
|---|---|
| Data integrity | Exact station × timestamp grid, no duplicates, encoding, checksums, missingness report |
| Leakage | Scalers, imputers and climatology fitted on training targets only |
| Windowing | First and last origin per split, horizon containment, valid target masks |
| **Graph physics** | **Synthetic west-to-east wind test** (a western source must influence an eastern target, never the reverse); transpose test; calm-wind fallback; missing-wind fallback; lag and distance limits |
| Tensor semantics | Station and patch axes preserved through fusion |
| Numerical safety | No NaN/Inf under extreme distance, wind and missing inputs |
| Training | Tiny-batch overfit; checkpoint round-trip |
| Evaluation | Hand-calculated metric fixtures including zero targets and missing labels |

**Note:** the roadmap's proposed test — zero out one branch and expect an untrained gate to shift toward the other — is invalid. An untrained sigmoid gate has no semantics. Test the algebra by setting gate logits explicitly.

---

## 9. Mapping: old roadmap → this specification

| Old roadmap phase | Disposition |
|---|---|
| Phase 0 — Environment, repo skeleton | **Keep**, plus the repairs in §8.2 |
| Phase 1 — Dataset acquisition | **Keep** (done) |
| Phase 2 — EDA | **Largely done** — `SDGT-CMA_Evidence_Review.md` §7 already contains it. Reproduce the plots for the report. |
| Phase 3 — Cleaning, imputation, normalisation | **Rewrite** per §3. Z-score removal and Min-Max are both out. |
| Phase 4 — Windowing | **Keep**, plus masks (§3.4, §3.5) and swept `L` (§4.10) |
| Phase 5 — Graph construction | **Rewrite** per §4.1–4.5. Direction, lag, fallback, constrained decay. |
| Phase 6 — GCN baseline | **Keep** |
| Phase 7 — Adaptive wind GAT | **Rewrite** per §4.6. Keep patch tokens; do not discard 167 of 168 hours. |
| Phase 8 — PatchTST branch | **Keep, rename** (§1.4). Keep patch tokens (§4.7). |
| Phase 9 — CMA fusion | **Rewrite** per §4.8. Cross-view over patch tokens, or rename and drop the temporal claim. |
| Phase 10 — Head, loss, assembly | **Keep**, plus diagnostics dict (§4.9) |
| Phase 11 — Training pipeline | **Keep** |
| Phase 12 — Sanity checks | **Keep**, minus the invalid gate test; add graph physics tests (§8.3) |
| Phase 13 — Metrics | **Extend** per §5.4 |
| Phase 14 — Comparison / ablation | **Replace** with the 5-cell matrix (§5.1) and full baseline set (§5.2) |
| Phase 15 — Interpretability | **Promote and expand** — now §6, and moved much earlier in the schedule |
| Phase 16 — Report | **Keep**, with the reframed claims in §1 |
| — | **New:** Delhi/ERA5 acquisition (§7); negative controls (§5.5) |

**Retired assumptions from the roadmap's §2:**
- **R2** (PatchTST shared across all models) — keep, still correct.
- **R5** (horizons 1/6/12/24) — keep.
- **R6** (last hour taken as `S`) — **retired**; see §4.6.
- **R7** ("channel-independent" = across stations) — **retired**; see §1.4.
- **R8** (two separate λ) — keep, but both now `softplus`-constrained.

---

## 10. Revised schedule (Aug–Nov)

| Week | Deliverable | Exit gate |
|---|---|---|
| 1 | Data contract: manifest, encoding fix, split, masks, `log1p` scaling | Leakage tests pass |
| 2 | **Thin vertical slice** — persistence + ridge, metrics module, one predicted-vs-actual plot | **One command goes from raw CSV to a figure** |
| 3–4 | Temporal branch + gradient-boosted tree + one temporal control; lookback sweep | Beats persistence and ridge at 12 h and 24 h |
| 5–6 | Static graph, then lag-aware wind graph; build the map animation as soon as `A_dyn` exists | **Graph visibly points downwind on 2016-03-04** |
| 7 | Fusion: parameter-matched concat vs cross-view over patch tokens | Both receive identical tokens |
| 8–9 | 5 configs × 3 seeds; full figure set from saved diagnostics; regime slices | Results table + figures complete |
| 10 | Delhi: CPCB pull + ERA5 meteorology, frozen; run best config | Robustness panel complete |
| 11 | Negative controls: wind reversal, edge permutation, station occlusion | Physical claims supported or narrowed |
| 12 | Write-up | — |

**Rationale for the reordering:** the old roadmap left a working end-to-end system until Phase 10–11. Week 2 here produces something that runs raw-CSV-to-figure, so every later step replaces a piece of a working system rather than building toward one that has never run.

**Go/no-go gates:**
- Model work starts only after the environment, encoding, split, masks and baseline metrics are reproducible.
- Dynamic-graph work starts only after the temporal-only model beats a naive baseline on validation.
- Cross-view fusion starts only after D0 is numerically stable across seeds.
- Delhi starts only after the Beijing matrix is complete.

---

## 11. Decisions taken (4 September 2026)

| # | Item | Decision |
|---|---|---|
| 11.1 | Save the measurement scripts | **Done.** Archived under `analysis/` — `common.py`, `profile_dataset.py`, `wind_transport.py`, `spatial_value_probe.py`, `find_episodes.py`. All four verified to reproduce the published numbers. Two tables in the evidence review (travel times, lookback sweep) were re-synced to the scripts' canonical output. |
| 11.2 | ERA5 boundary layer height for Beijing | **Approved.** Folded into §3.9. Soft dependency — the pipeline must run without it. |
| 11.3 | "Historical pollutant correlation" edge term | **Removed.** Folded into §1.5. The graph prior is distance + wind alignment + lag only. |
| 11.4 | Show the supervisor the §1 reframing before implementation | **Not required.** Proceeding directly to implementation. |

### Workspace cleanup performed

- Deleted `codex artifacts/` (117 MB of docx-rendering scaffolding: a LibreOffice installer, PDF/PNG renders, vendored PIL and pdf2image, build logs), `.pytest_cache/`, and all `__pycache__` directories.
- Moved to `docs/archive/` rather than deleted, since the workspace has no git history yet: `superseded_roadmap_v1.md` (the original 16-phase roadmap), `codex_audit_source.md` (was `report-source.md`), `SDGT-CMA_Pre-Implementation_Audit.docx`, and the two Codex phase notes. These can be purged on request.

## 12. What is *not* changing

To be explicit, because most of the project survives intact:

- The dual-branch parallel architecture and its motivation
- Cross-attention fusion with a learnable gate
- Huber loss
- Direct multi-horizon prediction at 1/6/12/24 h
- PM2.5 at all stations as the target; other pollutants and meteorology as inputs
- Chronological splits with train-only scaler fitting
- Beijing as primary, CPCB Delhi-NCR as validation
- PyTorch + free-tier GPU
- The overall Gantt structure of data → architecture → evaluation → report

The changes above are corrections to execution and honesty of framing, not a redesign of the project.
