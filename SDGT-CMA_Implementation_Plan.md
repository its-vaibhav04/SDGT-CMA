# SDGT-CMA — Implementation Plan

**The executable plan. What to build, in what order, with what interfaces and what proves it works.**

| | |
|---|---|
| **Date** | 4 September 2026 |
| **Depends on** | `SDGT-CMA_Evidence_Review.md` (why), `SDGT-CMA_Change_Specification.md` (what changes) |
| **Replaces** | `docs/archive/superseded_roadmap_v1.md` |
| **Window** | 12 weeks, three people |
| **Target** | A pipeline that visibly works, a figure set that shows the mechanism, honest numbers alongside |

---

## 0. How to use this document

Each phase has an **exit gate**. Do not start the next phase until the current gate passes. The gates are ordered so the cheapest thing that could invalidate the project fails first.

The single most important structural choice here: **Phase 2 produces a working end-to-end system in week 2.** Everything after that replaces a component of something that already runs, rather than building toward something that has never run. The old roadmap left the first working system until phase 10 of 16; that is how projects arrive at November with nothing to demonstrate.

Three conventions that are not negotiable, because each one silently corrupts results if broken:

1. **Adjacency is `[target, source]`.** Aggregation for target *i* sums messages from sources *j*. Write the west-to-east test before the graph.
2. **Every model is evaluated on identical rows** with the same target mask, and predictions are saved, not just scores.
3. **Metrics are always in µg/m³**, after inverse-transforming. Never report a scaled metric.

---

## 1. Notation

| Symbol | Meaning | Default |
|---|---|---|
| `B` | batch size | 64 |
| `N` | stations | 12 (Beijing) |
| `T` | total hours in the record | 35,064 |
| `F` | input features per station-hour | 27 |
| `L` | lookback window, hours | 48 (swept over 24/48/96/168) |
| `τ` | forecast horizon, hours predicted at once | 24 |
| `P` | patch length, hours | 8 (scales with `L`) |
| `Str` | patch stride | 8 (non-overlapping) |
| `Np` | patches per window = `(L − P)/Str + 1` | 6 |
| `d` | model width | 64 |
| `K` | maximum transport lag, hours | 6 |
| `H` | attention heads | 4 |

**Patch schedule as `L` is swept** — chosen to keep `Np` in the 6–8 range:

| `L` | `P` | `Str` | `Np` |
|---:|---:|---:|---:|
| 24 | 4 | 4 | 6 |
| **48** | **8** | **8** | **6** |
| 96 | 12 | 12 | 8 |
| 168 | 24 | 24 | 7 |

---

## 2. Repository layout

```
SDGT-CMA/
  analysis/                     # pre-implementation evidence scripts (done, frozen)
  configs/
    base.yaml                   # everything shared
    data/beijing.yaml  delhi.yaml
    model/t0.yaml  s0.yaml  d0.yaml  s1.yaml  d1.yaml
    ablation/*.yaml             # lookback sweep, temporal backbone, loss, etc.
  data/
    raw/                        # gitignored; PRSA CSVs, CPCB pulls, ERA5 cache
    processed/beijing/  delhi/  # the data contract artifacts (§3)
  docs/archive/                 # superseded documents
  experiments/
    runs/<config>_seed<k>/      # config.yaml, metrics.json, predictions.npz,
                                # diagnostics.npz, curve.csv, env.json
    figures/
  src/
    data/
      build_beijing.py          # raw CSV -> data contract
      build_delhi.py            # CPCB + ERA5 -> data contract
      era5.py                   # point time-series fetch, cached
      contract.py               # load/validate/describe a processed dataset
      windows.py                # origin index sets, Dataset, collate
      transforms.py             # log1p + standardise, fit on train only, invertible
    graphs/
      geometry.py               # haversine, bearing, from stations.csv
      static.py                 # thresholded-Gaussian distance graph
      dynamic.py                # lag-aware wind prior
    models/
      embedding.py              # FeatureEmbedding + station embedding
      spatial/gcn.py  wind_gat.py
      temporal/patch_transformer.py  tcn.py
      fusion/cross_view.py  concat.py
      head.py
      sdgt.py                   # assembles any config
    baselines/
      naive.py                  # persistence, seasonal naive, climatology
      linear.py                 # ridge, DLinear
      trees.py                  # LightGBM
    metrics.py                  # MAE/RMSE/WAPE, masked, per horizon and station
    bootstrap.py                # paired weekly block bootstrap
    train.py                    # one config, one seed
    evaluate.py                 # checkpoint -> predictions.npz + metrics.json
    figures/                    # one module per figure in §11
    utils/
      seeding.py  logging.py  manifest.py
  scripts/
    run_grid.py                 # configs x seeds
    make_figures.py
  tests/
```

---

## 3. The data contract

Both cities produce **the same artifacts**, so every downstream module is city-agnostic. Written to `data/processed/<city>/`.

| File | Shape / type | Notes |
|---|---|---|
| `features.npy` | `[T, N, F]` float32 | Scaled and imputed. This is what the model sees. |
| `target_raw.npy` | `[T, N]` float32 | PM2.5 in µg/m³, **never scaled**. All metrics use this. |
| `target_scaled.npy` | `[T, N]` float32 | Standardised PM2.5. The training target. |
| `target_mask.npy` | `[T, N]` bool | True where PM2.5 was genuinely observed. |
| `wind_uv.npy` | `[T, N, 2]` float32 | **Raw m/s**, direction of motion. The graph needs physical units. |
| `timestamps.npy` | `[T]` datetime64[h] | Contiguous hourly grid, no gaps in the index. |
| `stations.csv` | N rows | `station,lat,lon` in canonical order. **UTF-8, ASCII headers.** |
| `feature_names.json` | list[str] | Length `F`, in column order. |
| `scaler.json` | dict | Per-feature transform and parameters, fitted on train only. |
| `split.json` | dict | Target-time boundaries as both dates and hour indices. |
| `manifest.json` | dict | Source files, checksums, row counts, station order, build timestamp, code version. |

### 3.1 Feature set (`F = 27`)

| Group | Count | Transform |
|---|---:|---|
| Pollutants: PM2.5, PM10, SO2, NO2, CO, O3 | 6 | `log1p` → standardise |
| Meteorology: TEMP, PRES, DEWP, WSPM | 4 | standardise |
| RAIN | 1 | `log1p` → standardise (96 % zeros) |
| Wind components u, v | 2 | standardise |
| ERA5 boundary layer height | 1 | `log1p` → standardise |
| Calendar: `hour_sin/cos`, `dow_sin/cos` | 4 | none |
| Observation masks: 6 pollutants + wind direction | 7 | none (0/1) |
| Time since observed: PM2.5, PM10 | 2 | clip at 24 h, `log1p`, standardise |

**Wind components.** `wd` is direction-**from**, so the direction of motion is:

```python
u = -speed * sin(radians(wd))      # eastward
v = -speed * cos(radians(wd))      # northward
```

Check: `wd = 270` (from the west) gives `u = +speed`, air moving east. Correct.

**ERA5 is a soft dependency.** If the fetch fails, drop the BLH column, record its absence in `manifest.json`, and continue with `F = 26`. Never block the pipeline on an external service.

### 3.2 Splits (by target time)

| Split | Target range | Hour indices |
|---|---|---|
| train | 2013-03-01 00:00 → 2015-02-28 23:00 | `[0, 17544)` |
| val | 2015-03-01 00:00 → 2016-02-29 23:00 | `[17544, 26328)` |
| test | 2016-03-01 00:00 → 2017-02-28 23:00 | `[26328, 35064)` |

An origin `t` is valid for a split when every target hour `t+1 … t+τ` falls inside that split's range, `t − L + 1 ≥ 0`, and at least one target in the window is unmasked.

**The lookback of the first val/test origins reaches back into the preceding split. That is correct and is not leakage** — the model is reading history, exactly as it would in deployment. Document it; a reviewer will ask.

### 3.3 Imputation policy

1. Forward-fill each channel up to **3 hours**.
2. Beyond that, fill with the **training-split station-hour-of-day-month climatology**.
3. Record `obs_mask` before filling and `time_since_observed` alongside.
4. **Never fill a target.** `target_mask` records genuine observations; the loss and every metric are masked by it.
5. **Never interpolate across a forecast origin.** Forward-fill only looks backwards, which is what makes it causal.

### 3.4 Quality rules (replacing Z-score removal)

- Reject physically impossible values only: negatives, and values above a generous per-pollutant physical bound.
- Flag instrument ceilings (`PM == 999`, `CO == 10000`) with an indicator column; **keep the values**.
- Do not delete high episodes. The measured cost of a `z > 3` rule is 1.78 % of hours, all Severe-AQI.

---

## 4. Phase 0 — Environment and repo repair

**Week 1, days 1–2.**

| # | Task |
|---|---|
| 0.1 | `git init`, commit the current state as the audited baseline. Every run records the commit hash. |
| 0.2 | Standardise on `.venv`; ignore both `.venv` and `venv` during transition. |
| 0.3 | Move dependencies into `pyproject.toml` with pinned versions; document Python and PyTorch versions. |
| 0.4 | Rewrite the station table as UTF-8 with headers `station,lat,lon` (it was CP-1252 with degree symbols) **and move it to `configs/stations_beijing.csv`**. Under `data/raw/` it was caught by `.gitignore`, so a fresh clone never received it and the Kaggle build failed on the first cell. It is hand-compiled reference data, not a download, and belongs in version control. |
| 0.5 | `src/utils/seeding.py`: seed Python/NumPy/Torch/CUDA, deterministic DataLoader generators and worker init, `torch.use_deterministic_algorithms` under a strict flag. **Set `PYTHONHASHSEED` in the launcher, not inside the process** — setting it at runtime does nothing. |
| 0.6 | `src/utils/manifest.py`: capture Python, OS, CUDA, GPU, package versions, git hash into `env.json` per run. |
| 0.7 | Delete the placeholder modules under `src/models/` — they will be rewritten, and stale docstrings referencing old phase numbers cause confusion. |

**Exit gate:** a clean clone installs with one documented command and `pytest` passes.

---

## 5. Phase 1 — Data pipeline

**Week 1, days 3–5.** Owner: one person, full time.

### Deliverables

- `src/data/build_beijing.py` — raw CSVs → the full contract in §3
- `src/data/transforms.py` — `fit_transforms(train_slice)`, `apply`, `invert`; serialisable to `scaler.json`
- `src/data/contract.py` — `load(city)` returning a validated `ProcessedDataset`; `validate(city)` checking artifacts without rewriting them
- `src/data/windows.py` — `valid_origins(split, L, tau)`, `WindowDataset`, collate
- `src/data/era5.py` — cached point time-series fetch for BLH

### Implementation notes

**Keep everything resident.** The full feature tensor is `35064 × 12 × 27 × 4 B ≈ 45 MB`. Load it once onto the GPU and let the Dataset return *index slices*, not copies. Windowing is then pure indexing and the DataLoader is never the bottleneck.

```python
class WindowDataset(Dataset):
    """Returns index views into preloaded tensors. No per-item copying."""
    def __getitem__(self, i):
        t = self.origins[i]                      # forecast origin
        return {
            "features": self.features[t-L+1 : t+1],      # [L, N, F]
            "wind_uv":  self.wind_uv[t-L+1 : t+1],       # [L, N, 2]
            "target":   self.target_scaled[t+1 : t+1+τ], # [τ, N] -> transpose to [N, τ]
            "target_mask": self.target_mask[t+1 : t+1+τ],
            "origin": t,
        }
```

**ERA5.** Register on the Copernicus CDS, then use `reanalysis-era5-single-levels-timeseries` to pull point series at the 12 station coordinates rather than downloading grids. Variables: `boundary_layer_height` (plus `10m_u/v_component_of_wind` for Delhi later). Cache raw responses under `data/raw/era5/` and checksum them into the manifest.

### Tests (`tests/test_data_contract.py`)

| Test | Assertion |
|---|---|
| Grid integrity | `T == 35064`, 12 stations, timestamps strictly hourly with no duplicates |
| No NaN | `features` has no NaN after imputation |
| Target honesty | `target_mask.sum()` equals the raw non-null PM2.5 count exactly |
| Leakage | Transform parameters recomputed on the train slice alone reproduce `scaler.json` bit-for-bit |
| Invertibility | `invert(apply(x)) ≈ x` to 1e-5 |
| Wind convention | `u > 0` when `wd == 270`; `v < 0` when `wd == 0` |
| Window containment | For every split, first and last origin have all `τ` targets inside the split range |
| Off-by-one | For a random origin, `target[0]` equals `target_raw[t+1]` — the single most common silent bug in this problem |

**Exit gate:** every test passes and `python -m src.data.build_beijing` is reproducible from raw to manifest.

---

## 6. Phase 2 — Baselines, metrics, and the thin vertical slice

**Week 2.** This is the most important gate in the plan.

### Deliverables

- `src/metrics.py` — masked MAE, RMSE, WAPE; by horizon, by station, by arbitrary boolean slice; always in µg/m³
- `src/bootstrap.py` — paired weekly block bootstrap over test origins
- `src/baselines/naive.py` — persistence, seasonal naive (24 h, 168 h), station-hour climatology
- `src/baselines/linear.py` — ridge (direct multi-output), DLinear
- `src/baselines/trees.py` — LightGBM, one model per horizon
- `src/figures/forecast_trace.py` — predicted vs actual for a date range
- `scripts/run_baselines.py`

### The prediction format — fix it now, everything depends on it

Every model, baseline or neural, writes the same `predictions.npz`:

```
origins   : [n_test]        int32,   forecast origin hour indices
pred      : [n_test, N, τ]  float32, µg/m³
truth     : [n_test, N, τ]  float32, µg/m³
mask      : [n_test, N, τ]  bool
```

Because the origin set is derived from the frozen split and is identical for all models, **any two runs are automatically paired**. This is what makes the bootstrap in §6 of the evidence review possible.

### Expected values (from `analysis/profile_dataset.py`)

Persistence MAE, all rows: **10.54 / 32.67 / 45.44 / 58.53** at h = 1/6/12/24. Ridge on own history reaches roughly **10.19 / 31.56 / 41.94 / 51.03**. If your implementations land far from these, the bug is in the pipeline, not the model.

### Exit gate

**One command runs raw CSV → trained baseline → a plotted forecast figure.** Persistence and ridge scores land within a few percent of the numbers above. From this point the project always has something to demonstrate.

---

## 7. Phase 3 — Temporal branch

**Weeks 3–4.**

### Deliverables

- `src/models/embedding.py`
- `src/models/temporal/patch_transformer.py`
- `src/models/temporal/tcn.py` — the non-Transformer control
- `src/models/head.py`
- `src/train.py` — full training loop
- Config `t0.yaml` (temporal-only, no graph, no fusion)

### Feature embedding

```python
class FeatureEmbedding(nn.Module):
    """Per station-hour projection plus a learned station identity."""
    def __init__(self, n_features, n_stations, d):
        self.proj = nn.Linear(n_features, d)
        self.station = nn.Parameter(torch.zeros(n_stations, d))
        nn.init.normal_(self.station, std=0.02)

    def forward(self, x):                    # [B, L, N, F]
        return self.proj(x) + self.station   # [B, L, N, d]
```

The station embedding matters: the temporal encoder shares weights across stations, so without it the model cannot tell Huairou from Dongsi.

### Patch transformer

```
input   [B, L, N, d]
        -> permute to [B, N, L, d] -> reshape [B*N, L, d]
patchify                            [B*N, Np, P*d]
project + positional encoding       [B*N, Np, d]
TransformerEncoder (pre-norm, 3 layers, H heads, GELU)
                                    [B*N, Np, d]
        -> reshape                  [B, N, Np, d]
```

Shared weights across all `B*N` sequences. Use PyTorch's built-in `nn.TransformerEncoder` — patching and weight sharing are the contribution, not a custom attention kernel.

**Name it `MultivariatePatchTransformer`.** It is not PatchTST channel-independence and the report must not claim it is.

Expose `patch_indices(L, P, Str) -> LongTensor[Np, P]` from this module. **The spatial branch imports the same function** so both branches produce tokens on identical patch boundaries — otherwise the cross-attention in Phase 5 is aligning tokens that do not correspond.

### Head

```python
z: [B, N, Np, d] -> flatten -> [B, N, Np*d]
   -> Linear(Np*d, 256) -> GELU -> Dropout -> Linear(256, τ)  ->  [B, N, τ]
```

### Loss

Huber (`δ = 1.0`) on the **standardised raw** target, masked:

```python
loss = (F.huber_loss(pred, target, reduction="none", delta=1.0) * mask).sum() / mask.sum()
```

Note the deliberate choice: pollutant **inputs** get `log1p` for conditioning, but the **target** is standardised raw PM2.5, not log-transformed. This keeps the training objective aligned with the reported MAE and RMSE in µg/m³ and makes inversion exact. A log-target variant belongs in the ablation set, not the default.

### Tests

- `Np == (L - P) // Str + 1` matches the actual unfold length
- Same station duplicated in a batch produces identical outputs
- Two different stations produce different outputs from the same weights
- Tiny-batch overfit: 32 samples, no regularisation, loss → near zero in a few hundred steps

**Exit gate:** T0 beats persistence and ridge at h = 12 and h = 24 on validation. If it does not, stop — a graph cannot rescue a broken temporal encoder.

---

## 8. Phase 4 — Graph modules

**Weeks 5–6.** The riskiest phase. Build the figure before the model.

### 8.1 Geometry

`src/graphs/geometry.py` — `haversine_km(coords) -> [N,N]`, `bearing_deg(coords) -> [N,N]` (compass degrees, `i → j`). Precomputed once, cached.

### 8.2 Static graph

```
A_static[i, j] = exp(-dist(i,j)² / σ²)   if dist(i,j) ≤ κ else 0
```

`σ` = standard deviation of pairwise distances, `κ` = 75th percentile. Add explicit self-loops, then row-normalise. Symmetric, so the transpose question does not arise here — which is exactly why it is built first.

### 8.3 Lag-aware wind prior

For target *i*, source *j*, hour *t*:

```
speed_j      = ‖(u_j, v_j)‖
transport_j  = compass bearing of (u_j, v_j)          # direction of motion
align_ij     = cos(transport_j − bearing(j → i))      # >0 means j's air moves toward i
decay_ij     = exp(−softplus(λ_d) · dist_ij / d₀)     # d₀ = 20 km
gate_j       = tanh(speed_j / s₀)                     # s₀ = 2 m/s
prior_ij     = relu(align_ij) · decay_ij · gate_j     # all terms in [0, 1]
lag_ij       = clip(dist_ij / max(speed_j, v_min), 1, K)
```

Then, in order:

1. **Sparsify** — keep the top `k = 4` sources per target, zero the rest.
2. **Self-edge** — set `prior_ii` to a learnable scalar **explicitly**; do not add `I` to an unknown diagonal, and do not compute a self-bearing.
3. **Row-normalise** so each target's incoming weights sum to 1.
4. **Calm fallback** — where a target's pre-normalisation row sum is below `ε`, replace that row with `A_static`'s row. Record the fallback rate; 52 % of hours are near-calm, so this fires constantly and its rate is a reportable diagnostic.

Every term is bounded, `λ_d` cannot go negative through `softplus`, and the prior is a probability distribution over sources by construction.

### 8.4 Wind-aware GAT

**Design decision: attend over `(source, lag)` pairs.** Instead of gathering a `[B,L,N,N,d]` lagged tensor, treat station *j* at lag *k* as a distinct key. With `N = 12` and `K = 6` that is 84 keys per target — trivially cheap, and it makes `α` directly interpretable as *"how much did target i draw from station j, k hours ago"*, which is exactly what the attention figure needs.

```python
# h: [B, L, N, d];  prior: [B, L, N, N];  lag: [B, L, N, N] float
shifted = stack([shift_back(h, k) for k in range(K+1)], dim=-2)   # [B, L, N, K+1, d]
kernel  = softmax(-(arange(K+1) - lag.unsqueeze(-1)).abs() / temp, dim=-1)  # [B,L,N,N,K+1]

e = leaky_relu(a_src @ Wh_i + a_dst @ Wh_jk)                      # [B, L, N, N, K+1]
e = e + softplus(lam_bias) * log(prior.unsqueeze(-1) * kernel + eps)
e = e.masked_fill(prior.unsqueeze(-1) == 0, -inf)
alpha = softmax(e.flatten(-2), dim=-1).unflatten(-1, (N, K+1))
out_i = einsum("blnmk,blmkd->blnd", alpha, shifted)
```

The prior enters as a **log-space bias**, which makes it a multiplicative prior on attention probability rather than an additive nudge to an unnormalised logit on an unrelated scale.

Set `K = 0` to recover a plain same-hour GAT — a free ablation.

### 8.5 Spatial pooling

Both spatial branches emit `[B, L, N, d]`. Pool within patches using `patch_indices` **imported from the temporal module**:

```
[B, L, N, d]  --mean over each patch window-->  [B, N, Np, d]
```

### 8.6 Tests (`tests/test_graph_physics.py`) — write these first

| Test | Assertion |
|---|---|
| **West-to-east** | Synthetic uniform wind from the west; a western source must have high weight into an eastern target and near-zero in reverse |
| **Transpose** | Reversing the wind transposes the adjacency to within tolerance |
| Row-stochastic | Every row sums to 1; every entry ≥ 0 |
| Calm fallback | At zero wind speed, the adjacency equals `A_static` |
| Missing wind | NaN wind falls back rather than propagating NaN |
| Lag bounds | `1 ≤ lag ≤ K` everywhere; distant pairs at low speed clip to `K` |
| Sparsity | At most `k` non-self sources per target |
| Attention | `α` sums to 1 over `(source, lag)` jointly |
| Numerical | No NaN/Inf at extreme distance, zero speed, or all-missing input |

### Exit gate

The **dynamic-graph map animation for 2016-03-04 visibly points downwind** as the front arrives, and D0 trains stably across three seeds. This figure will find a direction bug faster than any unit test — build it here, not in week 11.

---

## 9. Phase 5 — Fusion and full model

**Week 7.**

### Cross-view fusion

Both inputs are `[B, N, Np, d]`. Reshape to `[B*N, Np, d]` and attend **over the patch axis, per station**:

```python
ctx_t = MHA(query=T, key=S, value=S)      # what spatial context matters per temporal patch
ctx_s = MHA(query=S, key=T, value=T)      # mirrored
g = sigmoid(Linear(cat([ctx_t, ctx_s], -1)))
Z = LayerNorm(g * ctx_t + (1 - g) * ctx_s)
```

This is the correction that makes the synopsis's central claim technically true: with the patch axis retained, the spatial branch genuinely attends to *when* something happened.

### Parameter-matched concatenation

The comparison is only meaningful if capacity is held roughly constant. Cross-view uses ~10 d²; give concat `Linear(2d → 4d) → GELU → Linear(4d → d) → LayerNorm` ≈ 12 d². **Report both parameter counts in the results table** so the match is auditable.

### Assembly

```python
class SDGT(nn.Module):
    def forward(self, batch, return_diagnostics=False):
        h = self.embed(batch["features"])                   # [B, L, N, d]
        t = self.temporal(h)                                # [B, N, Np, d]
        if self.spatial is None:                            # T0
            z = t
        else:
            adj, lag = self.graph(batch["wind_uv"])         # [B, L, N, N]
            s, alpha = self.spatial(h, adj, lag)            # [B, L, N, d]
            s = pool_patches(s)                             # [B, N, Np, d]
            z, gate = self.fusion(s, t)                     # [B, N, Np, d]
        pred = self.head(z)                                 # [B, N, τ]
        diag = {"adjacency": adj, "attention": alpha, "gate": gate} if return_diagnostics else None
        return pred, diag
```

**`return_diagnostics` is the one architectural decision that cannot be deferred.** Training runs with it off; evaluation runs with it on and dumps `diagnostics.npz`. Retro-fitting it after the grid has run means running the grid again.

### Tests

- `Z` has identical shape whether `S` came from the GCN or the wind GAT — fusion must be agnostic to the spatial branch
- **Gate algebra:** force gate logits to `+∞` and confirm `Z == LayerNorm(ctx_t)`. *Do not* test an untrained gate for semantic behaviour — an untrained sigmoid has no semantics, which is why the original roadmap's version of this test was invalid.
- All five configurations overfit a tiny batch
- Parameter counts of `s1` and `s0` are within 15 % of each other

**Exit gate:** all five configs train two epochs on a small subset with loss trending down.

---

## 10. Phase 6 — The experiment grid

**Weeks 8–9.**

| ID | Spatial | Fusion | Isolates |
|---|---|---|---|
| T0 | none | none | Is a graph needed at all? |
| S0 | static distance | concat | Conventional reference |
| D0 | lag-aware wind | concat | **S0 → D0**: the graph, alone |
| S1 | static distance | cross-view | **S0 → S1**: the fusion, alone |
| **D1** | lag-aware wind | cross-view | Full model; **D0 → D1** re-tests fusion |

**5 configs × 3 seeds = 15 runs.** At 10–30 minutes each, that is under 8 GPU-hours — well inside one week of Kaggle's free quota.

Plus these ablations, one seed each:
- Lookback sweep: `L ∈ {24, 48, 96, 168}` on D1
- Temporal backbone: TCN instead of the patch transformer, spatial branch fixed
- `K = 0` (same-hour graph) versus `K = 6`
- Loss: Huber versus MAE
- Log-transformed target

### Protocol

- Identical split, preprocessing, masks, and tuning budget for every config. Tune on validation only.
- **Seeds vary only the model** — weight init, dropout, edge dropout and batch shuffling. Split boundaries, valid origins, scaling and masks are fixed at build time and identical everywhere, which is what makes the paired bootstrap valid. Full detail in `docs/SEEDS_AND_RANDOMNESS.md`.
- **A difference must clear two independent variances:** the block bootstrap (which test weeks you drew) *and* the seed spread (where the optimiser landed). `scripts/run_grid.py` grades each comparison `robust` / `seed-dependent` / `comparable` accordingly; only `robust` belongs in the thesis as an improvement.
- AdamW, cosine schedule with warmup, gradient clipping at 1.0, early stopping on validation MAE averaged over h = 1/6/12/24, patience 10, maximum 100 epochs.
- Each run writes `config.yaml`, `env.json`, `curve.csv`, `metrics.json`, `predictions.npz`, `diagnostics.npz`.
- **The test set is read once, after the protocol is frozen.**

### Reporting rules

- Report mean ± spread across seeds, plus parameter count and training time.
- Paired weekly block bootstrap on **S0 vs D0** (the headline comparison).
- **If two configs differ by less than the seed spread, write "comparable."** Do not bold the smaller number.
- Never present absolute MAE as precise — its 95 % CI on this test set is ±12.7 %.

**Exit gate:** a results table with seeds, spread and one paired interval; and the honest sentence describing what it shows, written before the figures are made.

---

## 11. Phase 7 — Figures

**Week 9, in parallel with the grid.** One module per figure under `src/figures/`, each reading from `experiments/runs/`.

| # | Figure | Case study | Source |
|---|---|---|---|
| **1** | **Dynamic graph on a Beijing basemap**, animated hour by hour | **2016-03-04**, 369.5 → 55.7 µg/m³ in 12 h at 3.04 m/s | `diagnostics["adjacency"]` |
| 2 | Pollution rose — the 4× swing by wind direction | full record | raw data |
| 3 | Attention side-by-side, high-wind hour vs calm hour | any test pair | `diagnostics["attention"]` |
| 4 | Forecast vs actual through a severe episode | **2017-01-01**, 24 h mean 443, peak 522 | `predictions.npz` |
| 5 | **The honest failure case** | **2017-01-28**, Chinese New Year, fireworks peak 607 µg/m³ | `predictions.npz` |
| 6 | Error vs horizon, one line per config, persistence as floor | — | `metrics.json` |
| 7 | Regime-sliced gains by wind-speed bin | — | `predictions.npz` + wind |
| 8 | Gate values across horizons — spatial vs temporal reliance | — | `diagnostics["gate"]` |
| 9 | Per-station error bubble map | — | `metrics.json` |
| 10 | Delhi robustness panel — missingness + metrics | — | Delhi run |

Figure 5 is not optional. No model with this feature set can predict a fireworks spike; showing the miss and explaining why is worth more in a viva than hiding it, and it is the concrete justification for choosing Huber loss.

---

## 12. Phase 8 — Delhi

**Week 10.**

1. **Pollutants** — pull hourly station data via `sakethramanujam/cpcbccr-python-client` or `gsidhu/cpcbccr-data-scraper`. Expect to patch them; they track a portal that changes. Budget two days. If the client breaks, fall back to manual export for a reduced station set rather than losing the deliverable.
2. **Freeze on download** — station IDs and coordinates, exact date range, download timestamp, units, missing-data codes, checksum per file. Commit the raw files. **Never re-download mid-project.**
3. **Meteorology from ERA5, not CPCB.** CPCB's on-site wind sensors are sparse and heavily gapped, and the graph depends entirely on per-station wind. Pull `10m_u/v_component_of_wind`, temperature, dewpoint, precipitation and boundary layer height at each station's coordinates. The u/v components sidestep the compass-conversion and direction-convention problem entirely.
4. **Emit the same data contract** (§3), so every downstream module works unchanged.
5. **Scope:** train the best Beijing configuration from scratch on Delhi. One model, not five. Report the same metric table plus a missingness panel.

**Exit gate:** `src.data.contract.validate("delhi")` passes the same checks as Beijing.

---

## 13. Phase 9 — Controls, and Phase 10 — Write-up

**Week 11 — controls.** One afternoon of work that protects every physical claim in the thesis:

- **Wind reversal.** Negate `u, v` and re-evaluate D1. If performance does not degrade, the graph was never using wind — and no attention heatmap will rescue the claim.
- Edge permutation, station occlusion, identity adjacency.
- Report each as a delta against D1 with the paired bootstrap.

**Week 12 — write-up.** Results, limitations, honest framing per §1 of the change specification. Reuse this document's phase structure as the methodology outline.

---

## 14. Config schema

```yaml
# configs/model/d1.yaml
name: d1_windgraph_crossview
seed: 42
data:
  city: beijing
  lookback: 48
  horizon: 24
  patch_length: 8
  patch_stride: 8
model:
  d_model: 64
  n_heads: 4
  dropout: 0.1
  embedding: {station_embedding: true}
  temporal: {type: patch_transformer, n_layers: 3, ffn_mult: 4}
  graph:
    type: dynamic              # none | static | dynamic
    lag_max: 6                 # 0 recovers a same-hour graph
    top_k: 4
    distance_scale_km: 20.0
    speed_scale: 2.0
    calm_threshold: 1e-3
  spatial: {type: wind_gat, n_layers: 2, edge_dropout: 0.1}
  fusion: {type: cross_view}   # cross_view | concat | none
  head: {hidden: 256}
train:
  batch_size: 64
  epochs: 100
  optimizer: {name: adamw, lr: 1.0e-3, weight_decay: 1.0e-4}
  scheduler: {name: cosine, warmup_epochs: 5}
  grad_clip: 1.0
  loss: {name: huber, delta: 1.0}
  early_stopping: {monitor: val_mae_mean, patience: 10}
eval:
  horizons: [1, 6, 12, 24]
  save_diagnostics: true
```

The five model configs differ **only** in `graph.type`, `spatial.type` and `fusion.type`. Anything else differing between them breaks the attribution.

---

## 15. Hyperparameter starting points

These are defaults, not results. Tune on validation only, with the same budget for every config.

| Parameter | Start | Search if needed |
|---|---|---|
| `d_model` | 64 | {32, 64, 128} |
| Transformer layers | 3 | {2, 3, 4} |
| Heads | 4 | {2, 4} |
| Dropout | 0.1 | {0.0, 0.1, 0.2} |
| Learning rate | 1e-3 | {3e-4, 1e-3, 3e-3} |
| Weight decay | 1e-4 | {1e-5, 1e-4, 1e-3} |
| Batch size | 64 | {32, 64, 128} |
| Lookback `L` | 48 | swept as an ablation |
| `top_k` sources | 4 | {2, 4, 8} |
| `K` (max lag) | 6 | {0, 3, 6, 12} |

**Keep the model small.** There are only ~243 independent 3-day episodes in the training split against ~200 k parameters. Capacity is the enemy here, not the constraint.

---

## 16. Anti-goals

Things not to do, each of which has already cost someone a semester somewhere:

- **Do not tune until the graph wins.** If D0 ≈ S0, that is the result. Report it with the regime slices.
- **Do not add components to improve a number.** Every addition breaks the attribution the whole experiment exists to provide.
- **Do not read the test set before the protocol is frozen.**
- **Do not report a metric on scaled data.** Nobody can interpret it.
- **Do not use future observed meteorology** as a model input. If you ever run that variant, label it explicitly as an oracle upper bound and keep it out of the headline table.
- **Do not delete high-pollution hours** for any reason.
- **Do not build the interpretability figures last.** Figure 1 is a debugging tool for Phase 4.
- **Do not create multiple Kaggle accounts.** The whole grid is under 8 GPU-hours.
- **Do not scope-creep into KnowAir or cross-city transfer** until Phases 1–9 are genuinely finished.

---

## 17. Division of work

Three people, minimal blocking:

| | Weeks 1–2 | Weeks 3–7 | Weeks 8–12 |
|---|---|---|---|
| **A — data** | Phase 0 + 1: contract, transforms, windows, tests | ERA5, then Delhi acquisition (Phase 8) early so a broken scraper surfaces in week 4, not week 10 | Delhi run, robustness panel |
| **B — models** | Phase 2 metrics + bootstrap | Phases 3, 4, 5: temporal, graphs, fusion | Phase 6 grid, Phase 9 controls |
| **C — evaluation** | Phase 2 baselines + first figure | Figure modules against baseline outputs; config and run infrastructure | Phase 7 figures, results tables, write-up |

**Start Delhi acquisition in week 3, not week 10.** It is the only task with an external dependency that can fail in a way you cannot fix, so it needs slack.

---

## 18. Master gate list

| Gate | When | Condition |
|---|---|---|
| G0 | end wk 1 | Clean clone installs; data contract tests pass |
| **G1** | **end wk 2** | **One command: raw CSV → baseline → figure.** Persistence and ridge match the expected values |
| G2 | end wk 4 | T0 beats persistence and ridge at h = 12 and 24 on validation |
| **G3** | **end wk 6** | **Graph physics tests pass; the 2016-03-04 animation points downwind** |
| G4 | end wk 7 | All five configs overfit a tiny batch; gate algebra test passes |
| G5 | end wk 9 | Results table with seeds, spread, one paired interval; figures 1–9 built |
| G6 | end wk 10 | Delhi passes the same contract validation as Beijing |
| G7 | end wk 11 | Wind reversal degrades D1 — the graph demonstrably uses wind |
| G8 | end wk 12 | A clean clone reproduces one headline table |

If a gate fails, **stop and fix it**. Every gate here exists because skipping it produces a result that looks fine and is wrong.
