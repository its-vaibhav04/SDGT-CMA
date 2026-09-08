# SDGT-CMA — Evidence Review

**Independent pre-implementation assessment, grounded in direct measurement of the project's own data.**

| | |
|---|---|
| **Date** | 4 September 2026 |
| **Reviewed** | `Proj Syn final.docx`, `SDGT-CMA_Implementation_Roadmap.md`, `SDGT-CMA_Pre-Implementation_Audit.docx` / `report-source.md`, repo at current state |
| **Method** | Direct computation on all 420,768 rows in `data/raw/PRSA_Data_20130301-20170228/`, plus independent verification of every cited work |
| **Status of prior docs** | The roadmap and the Codex audit are both superseded in part; see `SDGT-CMA_Change_Specification.md` for exactly what survives |

Every number in this document was measured, not quoted. Nothing was taken on trust from the synopsis, the roadmap, or the Codex audit.

---

## 1. Verdict

**Build it, but change what it is testing.**

Nothing here is a reason to abandon the project. The pipeline is sound, the dataset is clean enough, and the entire experiment fits inside one free Kaggle account with room to spare. But three load-bearing assumptions in the proposal are contradicted by its own data, and building to the current specification means spending the semester measuring an effect that cannot be detected.

**The core finding:** in Beijing's 12-station network, wind is a powerful **city-scale** driver and a negligible **station-to-station** one. A wind-aware graph over stations 27 km apart is modelling the wrong spatial scale, and it will not produce a large headline improvement no matter how it is built.

**Given the locked scope** (Beijing + CPCB) and an evaluation that rewards a working system with strong figures, that is a manageable problem rather than a fatal one. The deliverable becomes a pipeline that visibly works, a figure set that shows the mechanism doing something real, and honest numbers reported alongside.

---

## 2. Headline measurements

| Measurement | Value | Consequence |
|---|---|---|
| Mean cross-station PM2.5 correlation | **0.887** (range 0.779–0.970) | The 12 stations are close to a single signal |
| PM2.5 autocorrelation at lag 168 h | **0.023** | No weekly periodicity exists to recover |
| Wind-graph gain over a plain network average | **+0.26 to +0.65 % MAE** | The central novelty is worth well under 1 % in a linear probe |
| Hours with wind speed < 1.5 m/s | **52.0 %** | `ReLU(align · v)` zeroes the graph for half the dataset |
| Independent weeks in the 1-year test split | **53** | Absolute MAE has a ±12.6 % confidence interval |

---

## 3. Finding 1 — Wind matters enormously, at the wrong scale

### 3.1 Wind direction drives a 4× swing in concentration

Mean PM2.5 by recorded wind direction, all stations, 2013–2017, restricted to hours with wind speed ≥ 2 m/s:

| `wd` | Mean PM2.5 (µg/m³) | n | | `wd` | Mean PM2.5 (µg/m³) | n |
|---|---:|---:|---|---|---:|---:|
| N | 27.5 | 9,143 | | S | 77.0 | 6,719 |
| NNE | 31.8 | 6,635 | | SSW | 75.7 | 8,904 |
| NE | 36.8 | 8,097 | | SW | 68.6 | 12,477 |
| ENE | 46.0 | 5,084 | | WSW | 62.0 | 6,622 |
| E | 73.4 | 4,663 | | W | 41.3 | 4,469 |
| **ESE** | **89.7** | 6,579 | | WNW | 26.4 | 11,347 |
| SE | 84.1 | 5,326 | | NW | 24.2 | 15,814 |
| SSE | 82.3 | 5,733 | | **NNW** | **22.6** | 10,241 |

Cleanest: NNW at 22.6. Dirtiest: ESE at 89.7. A **fourfold swing**, and exactly the geography you would predict — clean air off the Mongolian plateau to the north-west, loaded air off the Hebei industrial corridor to the south-east.

**This settles the wind-direction convention empirically, without needing a standards citation.** `wd` records the direction the wind blows *from*. Transport bearing is `wd + 180°`. This table is itself a unit test.

But note what it shows: a *regional* air mass arriving over the whole city at once. It says almost nothing about station A informing station B.

### 3.2 Station-to-station transport is real but an order of magnitude smaller

For every ordered station pair and every hour with wind ≥ 2 m/s, hours were split into those where the source station is upwind of the target (`align > 0.7`) and downwind (`align < −0.7`), then the source's ability to predict the target after a given lag was compared between the two groups.

| Lag (h) | Aligned corr. | Anti-aligned corr. | **Difference** | After removing city-wide common factor |
|---:|---:|---:|---:|---:|
| 0 | 0.8244 | 0.8509 | **−0.0265** | −0.0722 |
| 1 | 0.8265 | 0.8011 | **+0.0254** | −0.0207 |
| **2** | 0.7827 | 0.7377 | **+0.0449** | −0.0039 |
| 3 | 0.7268 | 0.6842 | **+0.0426** | −0.0048 |
| 4 | 0.6743 | 0.6407 | **+0.0336** | −0.0082 |
| 6 | 0.5860 | 0.5596 | **+0.0264** | −0.0133 |
| 8 | 0.5171 | 0.4902 | **+0.0269** | −0.0108 |
| 12 | 0.4136 | 0.3966 | **+0.0170** | −0.0081 |

The raw column is the fingerprint of genuine advection: negative at lag 0, peaking at **+0.045 at 2 h**, decaying afterwards — exactly the timescale implied by a median 27 km separation at a median 1.4 m/s wind.

The final column is the same computation after regressing out the network-wide mean, and it is **flat at zero**. Nearly all apparent inter-station structure is one common city-scale factor, not station-to-station transport.

### 3.3 Transport physics of this network

| Wind speed | Value | Travel time across median 28 km | Across closest pair (3.9 km) |
|---|---:|---:|---:|
| p25 | 0.9 m/s | 8.8 h | 1.2 h |
| p50 | 1.4 m/s | 5.7 h | 0.8 h |
| p75 | 2.2 m/s | 3.6 h | 0.5 h |
| p90 | 3.4 m/s | 2.3 h | 0.3 h |

- Hours below 1.5 m/s (near-calm): **52.0 %**
- Hours at or above 3 m/s: **14.0 %**

---

## 4. Finding 2 — Spatial value decays with horizon (the opposite of the claim)

The synopsis expects the advantage to appear "particularly over longer forecast horizons." Four ridge forecasters were built differing **only** in what spatial information they see, with identical rows, features, split and tuning protocol.

Split: train 2013-03-01 → 2015-02-28 (730 days), validate → 2016-02-29 (366), test → 2017-02-28 (365).
Ridge penalty tuned on validation only. These are the same date-derived boundaries the
production builder computes, so the probes and the trained models share one test year.

**Test MAE (µg/m³):**

| Variant | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---:|---:|---:|---:|
| A — own-station history only | 10.18 | 31.79 | 41.98 | 51.07 |
| B — + network-mean history | 9.73 | 29.80 | 40.99 | 50.68 |
| C — + wind-weighted upwind aggregate | **9.67** | **29.67** | **40.79** | **50.64** |
| D — + all 11 neighbours, raw | 9.91 | 30.48 | 41.94 | 52.30 |

**Improvement over A:**

| Variant | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---:|---:|---:|---:|
| B | +4.43 % | +6.24 % | +2.35 % | +0.76 % |
| C | +5.05 % | +6.65 % | +2.83 % | +0.85 % |
| D | +2.70 % | +4.13 % | +0.10 % | **−2.42 %** |
| **C vs B — the pure wind-direction gain** | **+0.65 %** | **+0.44 %** | **+0.49 %** | **+0.09 %** |

**Reading:**
- Spatial context is worth 4–6 % at 1–6 h and essentially nothing at 24 h. The horizon claim in the synopsis is **reversed**.
- Feeding all 11 neighbours in raw is actively harmful at long range — the model overfits cross-station noise.
- The entire measured contribution of wind direction is the C-minus-B row: **under 0.7 % everywhere**.

### 4.1 The wind term does behave physically

Regime-conditional gain of C over B:

| Horizon | Regime | n | MAE (B) | MAE (C) | Gain |
|---|---|---:|---:|---:|---:|
| h = 6 | calm < 1.5 m/s | 45,309 | 34.92 | 34.80 | +0.34 % |
| h = 6 | light 1.5–3 | 35,027 | 26.69 | 26.56 | +0.49 % |
| h = 6 | **windy ≥ 3 m/s** | 11,561 | 19.19 | 19.02 | **+0.90 %** |
| h = 6 | top-decile PM2.5 | 9,206 | 72.46 | 72.20 | +0.36 % |
| h = 12 | calm < 1.5 m/s | 45,139 | 49.46 | 49.34 | +0.24 % |
| h = 12 | light 1.5–3 | 34,857 | 35.21 | 34.91 | +0.87 % |
| h = 12 | **windy ≥ 3 m/s** | 11,541 | 25.34 | 25.13 | **+0.83 %** |
| h = 12 | top-decile PM2.5 | 9,162 | 112.36 | 111.66 | +0.62 % |

The gain is largest exactly when advection should matter — roughly 2.5–3.5× larger in windy hours than calm ones. **This is a genuine, reportable, physically consistent result.** It is also about a twentieth of the size the project implicitly promises.

---

## 5. Finding 3 — The 168-hour lookback is unsupported

### 5.1 Autocorrelation

| Lag | 1 h | 3 h | 6 h | 12 h | 24 h | 48 h | 168 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| Per-station PM2.5 | 0.969 | 0.883 | 0.767 | 0.596 | 0.404 | 0.149 | **0.023** |
| Network-mean PM2.5 | — | — | — | — | 0.437 | 0.158 | — |

There is no weekly cycle in the series. The justification "uncovering long-term periodicities that recurrent models tend to miss" does not hold for this data.

### 5.2 Lookback sweep

Test MAE (µg/m³), ridge on own-station history, identical protocol at every row:

| Lookback | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---:|---:|---:|---:|
| 6 hours | 10.20 | 31.89 | 42.26 | 51.18 |
| **24 hours** | **10.18** | **31.79** | **41.98** | **51.07** |
| 48 hours | 10.21 | 31.88 | 42.06 | 51.22 |
| 72 hours | 10.22 | 31.94 | 42.14 | 51.38 |
| **168 hours** | 10.28 | 32.12 | 42.62 | 52.04 |

Reproduce with `python analysis/spatial_value_probe.py`. All lookbacks are
evaluated on an identical row set, so the differences are attributable to the
lookback alone.

Error rises **monotonically** past ~24 h. L=168 is worst at every horizon.

### 5.3 There *is* a weekday effect — but it is one feature, not 168 timesteps

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---:|---:|---:|---:|---:|---:|---:|
| 73.0 | 77.1 | 77.9 | 79.2 | 83.2 | **87.5** | 80.6 |

Spread 18.1 % of the overall mean — comparable to the hour-of-day spread of 19.6 %. Keep `dow_sin`/`dow_cos`; treat `L` as a swept hyperparameter.

---

## 6. Finding 4 — Statistical power (the question nobody asked)

PM2.5 decorrelates in about two days, so four years of hourly data does not contain 35,064 independent observations.

| Quantity | Value |
|---|---:|
| Independent 3-day episodes in the 2-year train split | ≈ 243 |
| Independent weeks in the 1-year test split | 53 |
| 95 % CI on absolute MAE (h=6, weekly block bootstrap) | 29.80 → **[26.53, 34.02]**, i.e. ±12.6 % |
| 95 % CI on a *paired* difference (same rows, both models) | ±0.15 % |

**Comparing headline MAE figures between models on this dataset is statistically meaningless** — a 10 % difference sits comfortably inside the noise band. Paired comparisons on identical rows resolve effects below half a percent.

Implication: every model must be evaluated on **exactly the same rows** with the same target mask, and differences must be reported as paired, not as a table of standalone MAEs with the smallest one bolded.

---

## 7. Dataset facts

### 7.1 Structure

- 12 stations × 35,064 hourly steps = **420,768 rows**, 2013-03-01 00:00 → 2017-02-28 23:00
- 81.7 % of hours have all 12 stations reporting PM2.5
- PM2.5: mean 79.8, std 80.8, p50 55, p90 185, p99 370, max 999 µg/m³

### 7.2 Missingness

| Variable | Missing rows | % | Observed max |
|---|---:|---:|---:|
| PM2.5 | 8,739 | 2.08 | 999 |
| PM10 | 6,449 | 1.53 | 999 |
| SO2 | 9,021 | 2.14 | 500 |
| NO2 | 12,116 | 2.88 | 290 |
| CO | 20,701 | 4.92 | 10,000 |
| O3 | 13,277 | 3.16 | 1,071 |
| wd | 1,822 | 0.43 | categorical |
| TEMP / PRES / DEWP / RAIN / WSPM | 318–403 | 0.08–0.10 | — |

### 7.3 Gap structure (PM2.5)

- 2,837 gaps; mean 3.1 h, median 1 h, p90 4 h, **max 343 h** (14 days)
- Gaps ≤ 3 h are **87.2 % of gaps** but only **39.5 % of missing hours** — long gaps dominate the missing mass

### 7.4 Data quality is better than feared

- **Longest constant run in PM2.5 is 2 hours** → no stuck sensors
- PM2.5 = 999 occurs **once**, alongside genuine 941 and 957 readings
- PM10 = 999 occurs 3 times, with 991–995 present
- CO = 10,000 occurs 56 times, with 9,900 and 9,800 present

These read as **instrument ceilings, not sentinel codes**. Flag them; do not build a QC subsystem around them.

### 7.5 Meteorology is coarser than the proposal assumes

The UCI documentation notes meteorology is matched from the nearest weather station. Measured directly:

- **8 distinct wind-direction series** across 12 stations
  - `[Aotizhongxin, Guanyuan]`, `[Changping, Dingling]`, `[Dongsi, Nongzhanguan, Tiantan]`, `[Gucheng]`, `[Huairou]`, `[Shunyi]`, `[Wanliu]`, `[Wanshouxigong]`
- **11 distinct temperature series** (`Dongsi` and `Tiantan` share one)
- Mean unique wind directions among the 12 stations per hour: 4.79 (median 5)
- Cross-station WSPM correlation: 0.693 · TEMP correlation: 0.994

The wind field is **not** degenerate, but its effective resolution is 8 nodes, not 12. This must be stated as a limitation.

### 7.6 Spatial geometry

- Pairwise distance: min 3.9 km, max 59.5 km, mean 27.0 km
- Correlation vs distance across the 66 pairs: **r = −0.921** → real distance decay exists, and the coordinates in `configs/stations_beijing.csv` are validated by this
- Closest pair correlation 0.965; farthest pair 0.861

### 7.7 Distribution drift and seasonality

| Split | Mean | Std | p90 | Hours |
|---|---:|---:|---:|---:|
| train 2013-03 → 2015-02 | 83.94 | 79.02 | 191 | 17,544 |
| val 2015-03 → 2016-02 | 73.31 | 81.89 | 169 | 8,784 |
| test 2016-03 → 2017-02 | 77.98 | 82.81 | 185 | 8,736 |

Monthly means range from **53.5 (Aug)** to **104.6 (Dec)** — a 2× seasonal swing. Drift across splits is moderate (~12 %), not fatal.

RAIN is **96.07 % zeros**; mean when non-zero is 1.64.

### 7.8 Impact of the proposed preprocessing

**Z-score outlier removal at z = 3:**
- Threshold lands at **322 µg/m³**
- Removes **7,354 hours (1.78 %)**, every one of them a Severe-AQI episode
- Per-station z > 3 removes 7,384 hours (1.79 %) — same problem

**Min-Max normalisation:**

| Percentile | Value | Min-Max scaled | log1p |
|---|---:|---:|---:|
| p50 | 55 | 0.053 | 4.03 |
| p90 | 185 | 0.184 | — |
| p99 | 370 | 0.369 | 5.92 |
| p99.9 | 564 | 0.564 | — |
| max | 999 | 1.000 | 6.91 |

99 % of all data compresses into `[0, 0.37]`. `log1p` is far better conditioned.

---

## 8. Concrete figure targets in the test year

Identified from the test split (2016-03-02 → 2017-02-28) so nothing here touches training data.

**Severe episodes** (24 h rolling network mean):

| Date | 24 h mean | Peak hour | Mean wind |
|---|---:|---:|---:|
| 2016-03-04 08:00 | 345.2 | 378.6 | 1.95 m/s |
| 2016-03-17 22:00 | 318.7 | 355.3 | 1.53 m/s |
| 2016-12-04 02:00 | 307.3 | 408.8 | 1.22 m/s |
| 2016-12-21 13:00 | 395.1 | 432.2 | 1.83 m/s |
| **2017-01-01 14:00** | **443.0** | **521.8** | 1.19 m/s |
| **2017-01-28 11:00** | 340.2 | **607.0** | 2.72 m/s |

**Clear-out events** (best for the dynamic-graph figure):

| Date | PM2.5 change over 12 h | Mean wind |
|---|---|---:|
| **2016-03-04 19:00** | 369.5 → 55.7 | **3.04 m/s** |
| 2016-12-04 04:00 | 385.8 → 65.9 | 0.93 m/s |
| 2016-12-21 19:00 | 431.2 → 13.8 | 1.91 m/s |
| 2017-01-01 19:00 | 475.3 → 57.2 | 1.05 m/s |
| 2017-01-28 02:00 | 607.0 → 218.9 | 1.44 m/s |

**Notes:**
- **2016-03-04** has the strongest wind of any clear-out — the best case for demonstrating the dynamic graph pointing downwind.
- **2017-01-01** is the worst episode in the test year — the best forecast-vs-actual case study.
- **2017-01-28 is Chinese New Year 2017.** The 607 µg/m³ peak is fireworks-driven and unpredictable from the available features. Showing that miss and explaining it is worth more than hiding it, and it is the concrete justification for Huber loss.

---

## 9. Calibration — what "good" looks like on this dataset

A July 2026 arXiv study (2607.07279) uses this **exact** dataset (same 420,768 rows, same 12 stations) and compares persistence, seasonal naive, ridge, elastic net and XGBoost. **No deep model was needed to win any horizon.**

| Horizon | Best published model | RMSE | vs persistence | My measured persistence RMSE |
|---|---|---:|---:|---:|
| 1 h | Ridge | 20.15 | −5.7 % | 20.10 |
| 6 h | XGBoost | 54.13 | −11.8 % | 54.97 |
| 24 h | Elastic Net | 80.83 | −17.6 % | 87.71 |

Its conclusion — that most performance is attributable to recent PM2.5 history and synchronous PM10 — matches my independent measurements.

**Realistic expectations: the ceiling for any architecture on Beijing-12 is roughly 5–20 % better than persistence, and a well-tuned gradient-boosted tree gets most of the way there.** Beating persistence by 15 % at 24 h is a good result. Beating XGBoost at all is the result worth writing up.

**Persistence baselines measured on all rows** (µg/m³):

| | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---:|---:|---:|---:|
| MAE | 10.54 | 32.67 | 45.44 | 58.53 |
| RMSE | 20.10 | 54.97 | 72.31 | 87.71 |

---

## 10. Literature verification

All DOIs checked against Crossref / Europe PMC. **The synopsis bibliography contains at least one non-existent reference and two with wrong details.**

### 10.1 Errors in the synopsis bibliography

| Synopsis entry | Finding | Action |
|---|---|---|
| **[6] AGATNet** — "Z. Wang et al., IEEE Trans. Knowl. Data Eng., 2022" | Does not resolve. The real AGATNet is **Dimri, Choi, Salman, Park & Singh**, *JGR: Machine Learning and Computation*, 2024, DOI `10.1029/2024JH000244` — CMAQ **bias correction over South Korea**, a different task. | Replace citation and describe the task difference |
| **[9] AirQFormer** — "vol. 106, 2024" | Actually *Sustainable Cities and Society* **vol. 119, February 2025**, DOI `10.1016/j.scs.2024.106113` | Correct volume and year |
| **[13] DSGT** — "Y. Liu et al., Expert Systems with Applications, 2024" | Actually **Xia, Chen, Chen & Hu**, *Neurocomputing* vol. 616, 2025, DOI `10.1016/j.neucom.2024.128924`, titled *Dynamic synchronous graph transformer network for region-level air-quality forecasting* | Replace citation |
| **[11] TransNet** — "M. Li et al., Pattern Recognition, 2023" | No reliable match found for the supplied title and record | Remove unless a DOI can be supplied |

### 10.2 Verified prior work (relevant to positioning)

| Work | Verified record | Relevance |
|---|---|---|
| **AST-GT** | *An Adaptive Spatiotemporal Graph Transformer for Multi-Site PM2.5 Multi-Step Forecasting with Non-stationary and Sparse-Aware Method*, Shi, Yang & Qin, *J. Environ. Manage.*, **Aug 2026**, DOI `10.1016/j.jenvman.2026.130515` | **Closest prior work.** Transformer temporal encoding, dual-scale spatial attention, Temporal-Spatial Cooperative Attention, adaptive non-stationary normalisation — evaluated on **Beijing and India**, the same two regions this project proposes |
| DP-DDGCN | Xiao, Jin, Wang & Xu, *Sci. Total Environ.*, 2022, DOI `10.1016/j.scitotenv.2022.154298` | Distance + wind dynamic **directed** graphs for air quality |
| Multi-scale ST-GAT | Zhou, Wang, Wang & Guan, *Information Sciences*, 2024, DOI `10.1016/j.ins.2024.121072` | Dynamic spatiotemporal attention for air quality |
| DSGT | Xia et al., *Neurocomputing*, 2025 | Dynamic graph + Transformer for air quality |
| PM2.5-GNN | SIGSPATIAL 2020, DOI `10.1145/3397536.3422208` | Wind-informed graph; **KnowAir** dataset (184 cities, 3-hourly, 2015–2018, 17 met variables) with public baselines |
| DLinear | *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023, DOI `10.1609/aaai.v37i9.26317` | Required baseline |

### 10.3 Errors in the Codex audit's own literature claims

| Codex claim | Finding |
|---|---|
| "MAGICFormer" fuses graph and time-series features by cross-attention, DOI `10.1016/j.asoc.2025.113033` | That DOI resolves to *Integrating multi-dimensional graph attention networks and transformer architecture for predicting air pollution in **subway stations***. It is an enclosed-environment paper, not an ambient monitoring-network one. **Do not cite it as evidence for the general point.** |
| "AirFlow", arXiv `2608.09775`, an August 2026 dual-stream preprint with gated bidirectional cross-attention | **Could not verify by identifier or by title search. Treat as non-existent and do not cite.** |
| AST-GT described as having a "wind-driven dynamic attention bias" | Overstated. The abstract describes general multi-source meteorological context encoding, not a wind-directed graph. The wind-specific graph may still be a genuine differentiator. |

---

## 11. Compute feasibility

**Compute is the least constrained resource in this project.** Multiple accounts are unnecessary.

| Quantity | Value |
|---|---|
| Full feature tensor (35,064 × 12 × 25, float32) | **42 MB** — fits entirely in GPU memory, so windowing is pure indexing |
| Model size at `d_model=64`, 3 encoder layers | ≈ 200 k parameters |
| Compute per epoch | ≈ 1.6 TFLOPs → seconds on a T4/P100 |
| Estimated training time per run | 10–30 minutes |
| Whole experiment grid (5 configs × 5 seeds + baselines + tuning) | **20–40 GPU-hours** |

**Free tier available:** Kaggle gives 30 GPU-hours/week on a P100 (16 GB) or dual T4, with 9–12 hour sessions and background "Save & Run All" commits that survive closing the tab. That is one to two weeks of quota for the entire project from a single account. Colab free is sufficient for development and interactive debugging.

**Practical guidance:**
- Use Kaggle committed background runs for the real experiment grid, not interactive sessions.
- Write every run's config, seed, source hash and metrics to a versioned CSV so a disconnect never costs a result.
- The genuine resource risk is **time**, not GPU hours. The data contract and baseline suite deserve more of the semester than the model does.

---

## 12. Assessment of the Codex pre-implementation audit

### 12.1 What it got right (verified)

- **AST-GT is real and very close** to the proposal's positioning.
- **The AGATNet / AirQFormer / DSGT citation corrections are all correct.**
- **The fusion layer loses the time axis.** Both branches are pooled to one token per station before cross-attention, so the attention runs over 12 station tokens. There is no temporal axis left to attend to. This is a genuine tensor-design defect.
- **PatchTST is misnamed.** The roadmap flattens all variables within a patch and treats stations as channels. That is a legitimate multivariate patch encoder, but it is not PatchTST channel-independence.
- **The wind-direction / transpose risk is real.** The roadmap defines edges one way and aggregates the other; those two conventions are transposes and the resulting bug is silent.
- **Travel time cannot be ignored.** Confirmed and quantified: the directional signal peaks at a **2-hour lag** and is gone by 12.
- **The spatial branch discards 167 of 168 graph computations.**
- **The roadmap's gate-response unit test is invalid** — an untrained sigmoid gate has no semantics.
- **Graph weights need constrained parameterisation and normalisation.**
- **Delhi needs a frozen data contract.**
- **The "historical pollutant correlation" edge term is claimed in the synopsis but absent everywhere else.**

### 12.2 What it got wrong or overstated

- Two of its cited works are mischaracterised or unverifiable (§10.3).
- **Data-quality alarm is overstated.** Longest constant run is 2 hours; the extreme maxima are instrument ceilings. No QC subsystem is needed.
- Its 11-phase roadmap is heavier than an Aug–Nov semester supports.

### 12.3 What it missed entirely

- **It never asks whether the effect is detectable.** No power analysis, no effect-size estimate.
- **It never measures the correlation structure it theorises about.** The scale mismatch — the single most important finding in this review — is invisible without measurement.
- **It does not question the 168-hour lookback**, which the data rejects outright.
- **It treats the horizon claim as merely unproven** rather than measurably reversed.

---

## 13. Reproducing these numbers

Every figure in this document came from small NumPy scripts run against `data/raw/PRSA_Data_20130301-20170228/`. They compute, in order:

1. Missingness, gap-length distribution, cross-station correlation, distance matrix, autocorrelation, persistence baselines
2. Shared meteorological series detection, sentinel/ceiling values, weekday and hour-of-day effects
3. Wind-direction convention test, aligned vs anti-aligned advection signature, travel-time distribution
4. Incremental ridge probes (variants A/B/C/D) and regime-conditional slices
5. Weekly block-bootstrap power analysis
6. Distribution drift, seasonality, effective sample size
7. Severe-episode and clear-out event identification

These are saved in `analysis/` and are runnable against the raw CSVs with NumPy alone:

| Script | Reproduces |
|---|---|
| `analysis/profile_dataset.py` | §7 dataset facts, §9 persistence baselines |
| `analysis/wind_transport.py` | §3 wind convention, advection signature, transport physics |
| `analysis/spatial_value_probe.py` | §4 spatial value and lookback sweep, §6 statistical power |
| `analysis/find_episodes.py` | §8 figure targets |

```bash
PYTHONPATH=analysis python analysis/profile_dataset.py
```

---

## 14. Bottom line

This is a good project with a correctable aim. The engineering is well within reach, the compute is free, and the dataset is cleaner than the Codex audit feared.

What it needs is not more architecture. It needs a hypothesis matched to the scale of the data, a claim that survives the numbers coming in, and — given how this project is evaluated — a figure set designed before the model is written rather than after.

Proceed to `SDGT-CMA_Change_Specification.md` for the exact change list.
