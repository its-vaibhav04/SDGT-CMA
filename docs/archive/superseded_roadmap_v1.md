# SDGT-CMA Implementation Roadmap
### Spatiotemporal Dynamic Graph Transformer with Cross-Modal Attention — Air Quality Forecasting

This roadmap turns the fixed architecture from `Proj_Syn_final.docx` into working, verified code, one phase at a time. Each phase is self-contained: finish it, verify it against the "success" criteria, then move on. No phase assumes you've written the final model yet.

---

## 0. Notation (used consistently in every phase)

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `N` | number of monitoring stations (graph nodes) |
| `F` | number of raw input features per station per hour (pollutants + meteorology + time encodings) |
| `L` | look-back window length in hours (168 = 7 days, per synopsis) |
| `τ` | forecast horizon in hours (multi-step output) |
| `d` | embedding dim after the feature-embedding layer (`d_model`) |
| `d_s` | spatial branch output dim per station |
| `d_t` | temporal branch output dim per station |
| `P` | PatchTST patch length (hours per patch) |
| `Str` | PatchTST patch stride |
| `Np` | number of patches per station = ⌊(L − P)/Str⌋ + 1 |
| `H` | number of attention heads |
| `A^(t)` | N×N adjacency/edge-weight matrix at hour `t` |

---

## 1. What's architecturally fixed vs. what's your free choice

Per your instructions, nothing below in the **Fixed** column is up for redesign. Everything in **Free choice** is ordinary engineering (hyperparameters, dataset scope, code layout) and doesn't touch the architecture.

| Fixed by the synopsis | Free implementation choice |
|---|---|
| Two parallel branches (spatial + temporal), fused by cross-attention, not sequential | Number of GAT/transformer layers, hidden sizes, dropout |
| Dynamic wind-aware adjacency formula (§7.4.1) | Patch length `P`, stride `Str` |
| GAT attention with wind-bias term (§7.4.2) | Optimizer, LR schedule, batch size |
| PatchTST-style patch/channel-independent temporal encoder (§7.4.3) | Exact train/val/test split ratios |
| Bidirectional cross-attention + learnable gate fusion (§7.4.4) | Code framework choice (PyG vs. hand-rolled GAT) |
| MLP prediction head + Huber loss (§7.4.5) | Logging/experiment-tracking setup |

---

## 2. Ambiguity resolutions this roadmap assumes (see chat message for full reasoning)

These are labeled assumptions, not architecture changes — they fill gaps the synopsis prose leaves open. Each is called out again inline in the phase where it matters.

- **R1 — Fusion combine rule:** `Z = g ⊙ S' + (1 − g) ⊙ T'`, then LayerNorm, where `g = σ(MLP([S'; T']))`.
- **R2 — Temporal branch shared across all 3 models:** PatchTST is used in Model 1, 2, and 3 so the comparison isolates the graph and the CMA independently (per your "change only the evaluated component" instruction).
- **R3 — Baseline "GCN":** literal Kipf–Welling GCN (spectral, static distance-thresholded graph) — attention/adaptivity is reserved for the proposed model only.
- **R4 — Target variable:** PM2.5, all `N` stations, multi-step. Other 5 pollutants + 6 meteorological variables are input features.
- **R5 — Forecast horizons evaluated:** 1h, 6h, 12h, 24h.
- **R6 — Spatial branch time handling:** the adaptive GAT runs at every hour of the window (fresh wind snapshot each hour), then the **last** hour's station embeddings are taken as `S` (most recent wind is most predictive of near-term advection). Mean-pooling over the window is a documented alternative to try in Phase 11.
- **R7 — "Channel-independent" temporal branch:** independence across **stations**, not per-pollutant — one shared-weight transformer processes each station's full multivariate history independently of other stations, exactly as the synopsis phrases it ("process each station's own week-long history").
- **R8 — Two λ's:** the wind edge-weight formula's `λ` and the GAT bias term's `λ` are treated as two separate learnable scalars (not tied), since nothing in the text forces them equal.

---

## Phase 0 — Environment, Repo Skeleton, Reproducibility

**1. Objective:** A working Python/PyTorch environment and a repo layout you'll use for every later phase, with a seeded RNG so results are reproducible.

**2. What to learn/understand:** Why research code separates `data/`, `models/`, `experiments/` — so you can swap Model 1 → 2 → 3 by changing one config, not one script per model. Why seeding `torch`, `numpy`, and `random` matters for comparing models fairly.

**3. What to implement:**
```
sdgt-cma/
  data/            # raw + processed datasets (gitignored)
  src/
    data/          # dataset classes, preprocessing
    graphs/        # static + dynamic adjacency construction
    models/
      spatial/     # gcn.py, adaptive_gat.py
      temporal/    # patchtst.py
      fusion/      # cma.py
      full_model.py
    train.py
    evaluate.py
  configs/         # model1.yaml, model2.yaml, model3.yaml
  notebooks/       # EDA only — no production code here
  tests/           # shape/unit tests per phase
```
Recommended libraries: PyTorch (core), PyTorch Geometric only if it genuinely simplifies the GCN/GAT (it does for the baseline GCN; for the wind-aware GAT you're likely better off hand-rolling the attention so the wind-bias term is transparent and easy to debug — PyG's `GATConv` doesn't expose a slot for a custom edge-bias term without subclassing, and for a first research implementation you want to see every line of the attention computation). Pandas/NumPy/Scikit-learn for data. Matplotlib/Seaborn for plots.

**4. Expected input:** None (setup phase).

**5. Expected output:** A runnable `python -c "import torch; print(torch.__version__)"`, a git repo with the skeleton above, a `set_seed(42)` utility used everywhere.

**6. Tensor shapes:** N/A.

**7. How to test it:** `pytest tests/` runs (even with zero tests) with no import errors. Run any tiny script twice with the same seed and confirm identical output.

**8. What success looks like:** You can `import src.models.spatial.gcn` from anywhere in the repo without path hacks.

**9. Next:** Phase 1.

---

## Phase 1 — Dataset Acquisition & Raw Understanding

**1. Objective:** Get the Beijing Multi-Site Air-Quality dataset (primary, per §7.1) onto disk and understand its raw structure before touching it programmatically.

**2. What to learn/understand:** The dataset (UCI, Chen 2017, ref [17]) has ~12 station-level CSVs, hourly, 2013–2017, with PM2.5, PM10, SO₂, NO₂, CO, O₃, temperature, pressure, dew point, precipitation, wind speed, wind direction. You need each station's **latitude/longitude** separately (the UCI files give station names, not coordinates) — you'll have to attach coordinates yourself from a small reference table for the wind-graph module in Phase 5. Start with Beijing only (cleaner); defer CPCB Delhi-NCR to after your full pipeline works end-to-end, matching your own Gantt chart ordering — CPCB is a stress-test, not where you debug your first working model.

**3. What to implement:** A download/load script that reads all 12 station CSVs into one long-format DataFrame `[station, datetime, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, wd]`, plus a small `stations.csv` with `[station, lat, lon]` you compile manually (12 rows — quick to find via public station lists).

**4. Expected input:** Downloaded UCI zip.

**5. Expected output:** One clean Parquet/CSV: long-format table, and `stations.csv`.

**6. Tensor shapes:** N/A (still a DataFrame stage).

**7. How to test it:** `df.station.nunique() == 12`, `df.datetime.is_monotonic` per station, `stations.csv` has 12 unique lat/lon pairs that plot sensibly on a map of Beijing (quick sanity scatter plot).

**8. What success looks like:** One DataFrame, one coordinates table, no missing station IDs.

**9. Next:** Phase 2.

---

## Phase 2 — Exploratory Data Analysis

**1. Objective:** Understand the data's actual behavior (missingness, spikes, seasonality, cross-station correlation) before you preprocess it — preprocessing decisions should follow from what you see here, not be assumed blind.

**2. What to learn/understand:** Why EDA precedes preprocessing in every real pipeline — e.g., you can't choose an imputation strategy without first knowing your gap-length distribution.

**3. What to implement:** A notebook (not production code) that plots: (a) missingness heatmap per station/feature, (b) PM2.5 time series for 2–3 stations with visible spikes (Diwali/Spring Festival-analog events won't apply in Beijing, but winter heating season should show clearly), (c) hour-of-day and month-of-year boxplots of PM2.5, (d) a station×station Pearson correlation matrix of PM2.5, (e) gap-length histogram (how many consecutive missing hours, typically).

**4. Expected input:** Output of Phase 1.

**5. Expected output:** A short written summary (few paragraphs, in the notebook) of what you found — this is what justifies your Phase 3 choices to your mentor.

**6. Tensor shapes:** N/A.

**7. How to test it:** N/A — this phase's "test" is that your findings are internally consistent (e.g., correlation between geographically close stations should exceed correlation between far ones — if it doesn't, your coordinates in Phase 1 may be wrong).

**8. What success looks like:** You can state, in one sentence each: your typical gap length, your peak pollution season, and which station pairs are most/least correlated.

**9. Next:** Phase 3.

---

## Phase 3 — Cleaning, Imputation, Outlier Removal, Normalization

**1. Objective:** Implement the exact preprocessing pipeline the synopsis specifies (§7.2), fit only on the training split.

**2. What to learn/understand:** *Why* scalers must be fit on train-only data — fitting on the full dataset leaks test-set statistics into training, inflating your reported accuracy in a way a reviewer/mentor will catch immediately. Why short gaps get linear interpolation but long gaps get spatial averaging (linear interpolation over long gaps just draws a straight line through real dynamics; borrowing from correlated neighboring stations is more physically justified once the gap is too long to trust interpolation).

**3. What to implement:**
- `impute_short_gaps(df, max_hours=3)` → per-station linear interpolation for gaps ≤ threshold (pick the threshold from your Phase 2 gap-length histogram; the synopsis leaves the exact cutoff unspecified).
- `impute_long_gaps(df, stations_geo)` → for remaining NaNs, average the same timestamp across the `k` nearest stations (use your `stations.csv` distances).
- `remove_outliers(df, z_thresh=3)` → per-station, per-feature Z-score flagging, then treat flagged points as missing and re-run imputation (don't just delete rows — that breaks your hourly grid).
- `fit_minmax_scaler(train_df)` → fit on train split only, then `.transform()` on val/test.
- **Chronological split** (not random!) — spatio-temporal forecasting requires train/val/test to be contiguous time blocks (e.g., 70/15/15 by date), never shuffled, or you'll leak future information into training.

**4. Expected input:** Long-format DataFrame from Phase 1.

**5. Expected output:** Cleaned, imputed, scaled long-format DataFrame + saved `scaler.pkl` (needed later to invert predictions back to real units for reporting).

**6. Tensor shapes:** N/A (still DataFrame).

**7. How to test it:** `df.isna().sum().sum() == 0` after this phase. Plot a station's PM2.5 before/after imputation — the imputed sections should look plausible, not flat or spiky. Re-run your Phase 2 gap-length check — it should now report zero gaps.

**8. What success looks like:** Zero NaNs, scaler saved, train/val/test split boundaries recorded as explicit dates (write them down — you'll reuse the identical boundaries for every one of the 3 models, which is required for a fair comparison).

**9. Next:** Phase 4.

---

## Phase 4 — Feature Engineering & Sequence/Window Construction

**1. Objective:** Turn the cleaned long-format table into the `(X, Y)` tensor pairs the model actually trains on.

**2. What to learn/understand — conceptual:** Cyclical time encoding: passing `hour=23` and `hour=0` as raw integers tells the model they're 23 apart, when they're actually adjacent. Sine/cosine encoding fixes this by mapping the 24-hour cycle onto a circle.
**Math:** `hour_sin = sin(2π·hour/24)`, `hour_cos = cos(2π·hour/24)` (same pattern for day-of-week with period 7). §7.3 also specifies passing everything through a shared Feature Embedding Layer before the branches — that's `Linear(F → d_model)` applied per (station, hour).

**3. What to implement:**
- `add_cyclical_features(df)` → adds `hour_sin, hour_cos, dow_sin, dow_cos` (raw wind direction is kept **separately, in radians, unencoded** — Phase 5's graph module needs the actual bearing angle, not a sine/cosine pair, so don't cyclically-encode wind direction away; store both a raw-radians copy for the graph module and a sin/cos copy for the general feature embedding).
- A sliding-window function that, for each valid start hour `t`, produces one training example: `X = features[t : t+L]` (shape `[L, N, F]`) and `Y = PM2.5[t+L : t+L+τ]` for all stations (shape `[N, τ]`). Slide by 1 hour (standard for this data volume) within each split only — never let a window cross a split boundary.
- The shared `FeatureEmbedding = nn.Linear(F, d_model)` module (used identically by Model 1/2/3, applied later inside the model, not baked into the dataset — keep raw `F`-dim tensors on disk, embed at train time so `d_model` stays a tunable hyperparameter).

**4. Expected input:** Cleaned DataFrame from Phase 3.

**5. Expected output:** A PyTorch `Dataset`/`DataLoader` yielding `(X, Y, wind_raw)` batches. `wind_raw` (speed + raw-radian direction, per station per hour) is kept alongside `X` since Phase 5's graph module needs it directly.

**6. Tensor shapes:**
```
X:        [B, L, N, F]        # L=168, F = pollutants+meteo+cyclical time (embedding happens in-model)
wind_raw: [B, L, N, 2]        # (speed, direction_radians) per station per hour
Y:        [B, N, τ]           # PM2.5 only, all stations, τ future steps
```

**7. How to test it:** Pull one batch, manually trace `X[0, -1, 0]` (last hour, station 0) back to the raw DataFrame row it came from and confirm the values match. Assert `Y` never overlaps `X`'s time range for the same sample (off-by-one errors here are the single most common bug in ST-forecasting code and are silent — they don't crash, they just quietly inflate your validation accuracy).

**8. What success looks like:** `next(iter(train_loader))` returns correctly-shaped, leak-free tensors, and you can name exactly which real calendar hours produced any given sample.

**9. Next:** Phase 5.

---

## Phase 5 — Graph Construction (Static + Dynamic Wind-Aware)

**1. Objective:** Build the two adjacency representations: `A_static` for the baseline GCN (Models 1 & 2), and `A_dyn^(t)` for the proposed adaptive branch (Model 3), computed fresh from live wind data every hour.

**2. What to learn/understand — conceptual:** A static graph says "these two stations are always related because they're geographically close." A dynamic, wind-aware graph says "these two stations are related *right now* because the wind is currently blowing from one toward the other" — connectivity strengthens for downwind pairs and weakens or vanishes otherwise. This is literally why the proposed model should beat a static-graph baseline on days when wind direction shifts sharply.

**Math — static graph (R3, standard thresholded-Gaussian construction from Yu et al., ref [2], used here only for the baseline):**
```
A_static[i,j] = exp(-dist(i,j)² / σ²)   if dist(i,j) ≤ κ, else 0
```
`σ` and `κ` (distance threshold) are free hyperparameters — pick `σ` = std of all pairwise distances, `κ` = e.g. 75th percentile distance, and treat both as tunable.

**Math — dynamic wind-aware graph (fixed, from §7.4.1, both drafts):**
```
θ_ij            = bearing angle from station i to station j          (atan2 of coordinate delta)
Align_ij^(t)    = cos(θ_wind,i^(t) − θ_ij)
A_ij^(t)        = [1 / exp(λ1 · dist(i,j))] · ReLU(Align_ij^(t) · v_i^(t))
```
`v_i^(t)` = wind speed at station i, hour t. `λ1` is a learnable scalar (R8: kept separate from the GAT's own `λ`). Note the `ReLU`: whenever a station pair isn't wind-aligned (`Align ≤ 0`) or wind speed is ~0, that edge weight goes to exactly zero — so this graph can become sparse or briefly disconnect a node during calm wind. **Add a self-loop (`A_ii^(t) = 1`, always)** — not stated explicitly in the synopsis, but without it a station can be entirely cut off from the graph during still air, which is a numerical-stability failure mode, not a modeling choice; flag this addition to your mentor as a standard, necessary implementation safeguard.

**3. What to implement:**
```python
def dynamic_wind_adjacency(coords, wind_speed, wind_dir_rad, lam1):
    # coords: [N, 2]; wind_speed: [B, N]; wind_dir_rad: [B, N]
    dist   = pairwise_haversine(coords)                      # [N, N], precomputed once
    theta  = pairwise_bearing(coords)                        # [N, N], precomputed once
    align  = torch.cos(wind_dir_rad[..., None] - theta)      # [B, N, N]
    decay  = 1.0 / torch.exp(lam1 * dist)                    # [N, N] -> broadcast
    A      = decay * F.relu(align * wind_speed[..., None])   # [B, N, N]
    A      = A + torch.eye(N, device=A.device)                # self-loops
    return A
```
Precompute `dist` and `theta` once in Phase 1/5 setup (they never change) — only `align` and the final `A` are recomputed per hour, per batch.

**4. Expected input:** `stations.csv` (coords), `wind_raw` batch from Phase 4.

**5. Expected output:** `A_static` (computed once, reused for every batch), `A_dyn` (computed fresh per forward pass, per hour of the window).

**6. Tensor shapes:**
```
A_static: [N, N]                # fixed, precomputed once
A_dyn:    [B, L, N, N]           # recomputed every forward pass, per hour in the window
```

**7. How to test it:** Pick one hour with strong, consistent wind in one direction. Confirm stations downwind of a high-emission station get noticeably higher `A_ij` than stations upwind of it. Confirm `A_dyn.min() >= 0` (ReLU held) and every row has at least the self-loop nonzero (no fully-isolated node).

**8. What success looks like:** You can visualize one hour's `A_dyn` as a directed graph overlaid on the Beijing map and it visibly "points" downwind.

**9. Next:** Phase 6.

---

## Phase 6 — Spatial Branch, Component A: Conventional GCN (Models 1 & 2 baseline)

**1. Objective:** Implement the standard Kipf–Welling GCN spatial encoder used as the baseline spatial branch (R3), on the static graph from Phase 5.

**2. What to learn/understand — conceptual:** A GCN aggregates each node's own features with its neighbors' features, weighted uniformly by the (normalized) static adjacency — every neighbor gets the same importance regardless of current conditions. This is exactly what Model 3's adaptive branch improves on, so implementing this correctly matters for a fair comparison, not just as a throwaway baseline.
**Math:** `H' = σ(D̂^{-1/2} Â D̂^{-1/2} H W)`, where `Â = A_static + I`, `D̂` is `Â`'s degree matrix.

**3. What to implement:** Either PyTorch Geometric's `GCNConv` directly (this is the one component where using the library is the right call — it's not the novel part of your project, and hand-rolling it adds risk with no learning payoff for *this* branch), or a 3-line manual version if you want full transparency. Stack 2 GCN layers with ReLU between them.

**4. Expected input:** `X_emb[b, t]` for one hour, shape `[N, d_model]`, plus `A_static`.

**5. Expected output:** Per-station spatial embedding for that hour.

**6. Tensor shapes:**
```
Input:  X_emb [B, N, d_model],  A_static [N, N]
Output: S_gcn [B, N, d_s]
```
(Applied once per window — unlike the adaptive branch, the static graph doesn't change per hour, so you don't need to loop over `L` here; a common simple choice is to run the GCN on the **last hour's** embedded features, matching R6's convention for the adaptive branch so both baseline and proposed spatial branches are compared on the same time-slicing rule.)

**7. How to test it:** Feed a batch through, confirm output shape, confirm no NaNs. Zero out one station's input features and confirm its *neighbors'* embeddings change slightly (proof the graph conv is actually aggregating, not just passing features through a per-node MLP).

**8. What success looks like:** A `GCNSpatialBranch` module that runs standalone, unit-tested, before it ever touches the temporal branch or CMA.

**9. Next:** Phase 7 (or skip ahead to Phase 8 first if you'd rather build both spatial variants back-to-back before the temporal branch — order between 7 and 8 doesn't matter, both are prerequisites for Phase 9).

---

## Phase 7 — Spatial Branch, Component B: Wind-Aware Adaptive GAT (Model 3, proposed)

**1. Objective:** Implement the fixed proposed spatial branch — attention-weighted aggregation over the dynamic wind graph.

**2. What to learn/understand — conceptual:** Instead of the GCN's uniform neighbor weighting, this layer *learns* how much attention to pay each neighbor, and biases that learned attention with the physical wind-alignment signal from Phase 5 — so the model isn't purely data-driven, it's nudged toward physically sensible connectivity from the start.

**Math (fixed, §7.4.2, both drafts, standard GAT term + wind bias):**
```
e_ij   = LeakyReLU(aᵀ[W·h_i ‖ W·h_j]) + λ2 · w_ij        # w_ij = A_dyn[i,j] from Phase 5
α_ij   = softmax_j(e_ij)                                  # normalized over neighbors of i
S_i^(t) = σ( Σ_{j∈N(i)} α_ij^(t) · W_S · X_j^(t) )
```
Multi-head: repeat with `H` independent `(W, a, λ2)` sets, concatenate (or average) head outputs. `‖` is concatenation, `λ2` a learnable scalar per R8.

**3. What to implement:** A hand-rolled `AdaptiveWindGAT` module (recommended over `GATConv` here — you need to inject the `λ2·w_ij` bias term directly into the attention logits before softmax, which the standard PyG `GATConv` doesn't expose without subclassing; writing it by hand also means you can literally print `α_ij` later for the interpretability plots in Phase 15).
```python
class AdaptiveWindGAT(nn.Module):
    def forward(self, X, A_dyn):                 # X:[B,N,d], A_dyn:[B,N,N]
        h = self.W(X)                             # [B,N,d_s]
        e = self.attn_logits(h)                    # [B,N,N] via LeakyReLU(a^T[Wh_i||Wh_j])
        e = e + self.lam2 * A_dyn                  # wind bias
        e = e.masked_fill(A_dyn == 0, float('-inf'))  # no edge -> no attention mass
        alpha = torch.softmax(e, dim=-1)
        out = torch.einsum('bij,bjd->bid', alpha, h)
        return torch.sigmoid(out) if self.final_layer else F.elu(out)
```
Run this **once per hour of the window** (loop or vectorize over `L`), producing per-hour, per-station embeddings; take the **last hour's** output as `S` per R6 (flag mean-pooling as an alternative to try in Phase 11's hyperparameter sweep).

**4. Expected input:** `X_emb[b, :, :, :]` for the full window, `A_dyn[b, :, :, :]` from Phase 5.

**5. Expected output:** Final per-station spatial embedding, `S`.

**6. Tensor shapes:**
```
Input:  X_emb [B, L, N, d_model],  A_dyn [B, L, N, N]
Per-hour output: [B, N, d_s]  (repeated for each of the L hours if you keep the full sequence)
Final S: [B, N, d_s]           # after taking the last hour (R6)
```

**7. How to test it:** Confirm `alpha.sum(dim=-1) ≈ 1` for every node (softmax sanity check). Compare `AdaptiveWindGAT` output against `GCNSpatialBranch` output on the **same** input on a calm-wind hour vs. a strong-wind hour — the adaptive branch's embeddings should shift noticeably more between those two hours than the GCN's (since GCN doesn't see wind at all).

**8. What success looks like:** A standalone `AdaptiveWindGAT` module, unit-tested, whose attention weights you can already inspect and sanity-check against Phase 5's wind visualization.

**9. Next:** Phase 8.

---

## Phase 8 — Temporal Branch: PatchTST-style Encoder (shared by Models 1, 2, 3 — R2)

**1. Objective:** Implement the patch-based, station-independent transformer temporal encoder (§7.4.3), shared identically across all three models per R2 so the comparison isolates the graph and CMA components cleanly.

**2. What to learn/understand — conceptual:** Instead of feeding 168 individual hourly steps into self-attention (expensive, and each single hour carries little signal on its own), PatchTST first chops the sequence into short patches (e.g., 24-hour chunks) and treats each **patch** as one token — attention then happens between patches, capturing periodicities (daily, weekly cycles) far more efficiently than hour-by-hour attention or a recurrent model.

**Math:**
```
Patching:        X_patches[b,n] = patchify(X[b, :, n, :], P, Str)      # -> [Np, P*d_model]
Patch embedding: E = X_patches @ W_patch + PosEnc                       # -> [Np, d_t]
Self-attention:  Q,K,V = E W_Q, E W_K, E W_V
                 Attn   = softmax(QKᵀ/√d_k) V
```
stacked over several transformer encoder layers, applied with **shared weights across stations** (R7: this is where "channel-independent" applies here — same transformer weights process every station's window, station identity isn't baked into the weights, only into which data is passed through).

**3. What to implement:**
- `patchify(x, P, Str)`: unfold the `L`-length sequence into `Np` overlapping (or non-overlapping) patches.
- `PatchEmbedding = nn.Linear(P * d_model, d_t)` + learned/sinusoidal positional encoding over the `Np` patch positions.
- A standard `nn.TransformerEncoder` (PyTorch's built-in is fine here — this part is *not* the novel piece of your architecture, PatchTST's contribution is the patching + channel-independence, not a custom attention mechanism, so using the library implementation is appropriate and won't obscure anything you need to explain).
- A pooling head that turns `[Np, d_t]` per station into a single `[d_t]` vector for CMA: **flatten + linear** (`Np*d_t → d_t`), matching PatchTST's own forecasting-head design (mean-pooling over patches is a reasonable ablation alternative).
- Fold the `[B, N]` dimensions together as the effective batch when calling the transformer, since weights are shared across stations: reshape `[B, N, L, d_model] → [B*N, L, d_model]` before patching, reshape back after.

**4. Expected input:** `X_emb[b, :, n, :]`, the full embedded window for each station.

**5. Expected output:** Per-station temporal embedding, `T`.

**6. Tensor shapes:**
```
Input:            X_emb [B, N, L, d_model]      (note: N and L swapped vs. spatial branch's ordering — reshape accordingly)
After patchify:   [B*N, Np, P*d_model]
After embedding:  [B*N, Np, d_t]
After encoder:    [B*N, Np, d_t]
After pooling:    [B*N, d_t] -> reshape -> T [B, N, d_t]
```

**7. How to test it:** Confirm `Np = (L - P)//Str + 1` matches your actual unfold output length. Feed the same station's history twice (duplicated in batch) and confirm identical output (determinism check). Feed two *different* stations through and confirm the *same* transformer weights produce *different* outputs purely from different input data (proof weight-sharing is working, not accidentally learning per-station embeddings some other way).

**8. What success looks like:** A standalone `PatchTSTBranch` module, unit-tested, that you'll import unchanged into all three model configs.

**9. Next:** Phase 9.

---

## Phase 9 — Cross-Modal Attention (CMA) Fusion Layer

**1. Objective:** Implement the bidirectional cross-attention + learnable gate that fuses `S` and `T` (Model 2 and Model 3 only — Model 1 skips this phase entirely and concatenates instead, see Phase 10).

**2. What to learn/understand — conceptual:** Instead of just concatenating the spatial and temporal embeddings (which forces the model to *implicitly* figure out how they relate), cross-attention lets each branch explicitly query the other: "given what I know spatially about this station, which parts of its temporal history matter most?" and the mirrored question in reverse. This is the specific mechanism the synopsis's whole novelty claim rests on, so this phase deserves the most careful testing of any in the roadmap.

**Math (fixed, §7.4.4, final draft, with R1's combine-rule resolution):**
```
S'_i = softmax(Q_s K_tᵀ / √d) V_t          # Q from S (spatial), K/V from T (temporal)
T'_i = softmax(Q_t K_sᵀ / √d) V_s          # Q from T (temporal), K/V from S (spatial)   [mirrored pass]
g    = σ(MLP([S'_i ; T'_i]))               # learnable gate, R1
Z_i  = LayerNorm( g ⊙ S'_i + (1 − g) ⊙ T'_i )
```
Both passes attend **across the N stations** (the "sequence" the attention runs over is the set of stations, not time or patches — `S` and `T` are both already `[B, N, d]` at this point, one token per station), consistent with the per-station-`i` subscript notation the synopsis uses throughout §7.4.4.

**3. What to implement:**
```python
class CMAFusion(nn.Module):
    def __init__(self, d_s, d_t, d_model):
        self.q_s, self.k_t, self.v_t = [nn.Linear(d_s, d_model)]*1 + [nn.Linear(d_t, d_model)]*2
        self.q_t, self.k_s, self.v_s = [nn.Linear(d_t, d_model)]*1 + [nn.Linear(d_s, d_model)]*2
        self.gate = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, S, T):                       # S:[B,N,d_s], T:[B,N,d_t]
        S_prime = attention(self.q_s(S), self.k_t(T), self.v_t(T))   # [B,N,d_model]
        T_prime = attention(self.q_t(T), self.k_s(S), self.v_s(S))   # [B,N,d_model]
        g = self.gate(torch.cat([S_prime, T_prime], dim=-1))
        Z = self.norm(g * S_prime + (1 - g) * T_prime)
        return Z, g   # keep g -- useful later for interpretability
```

**4. Expected input:** `S` (Phase 6 or 7 output), `T` (Phase 8 output).

**5. Expected output:** Fused spatio-temporal representation `Z` per station.

**6. Tensor shapes:**
```
Input:  S [B, N, d_s],  T [B, N, d_t]
Output: Z [B, N, d_model],  g [B, N, d_model]
```

**7. How to test it:** Zero out `T` entirely and confirm `g` shifts toward favoring `S'` (sanity-check the gate is actually responsive, not stuck at 0.5 everywhere from a bad init). Confirm `Z`'s shape matches regardless of whether `S` came from the GCN (Phase 6) or the adaptive GAT (Phase 7) branch — this module must be agnostic to which spatial branch feeds it, since Model 2 and Model 3 share this exact fusion layer and only differ in spatial branch.

**8. What success looks like:** A standalone `CMAFusion` module, unit-tested with both spatial branch variants plugged in interchangeably.

**9. Next:** Phase 10.

---

## Phase 10 — Prediction Head, Loss, and Full Model Assembly (Models 1, 2, 3)

**1. Objective:** Wire Phases 6–9 into the three exact models you need to compare, changing only the specified component between them.

**2. What to learn/understand:** Why Huber loss over plain MSE for this data — pollution spikes (festivals, temperature inversions) are exactly the kind of large-but-real outliers that MSE over-penalizes, destabilizing gradients; Huber behaves like MSE near zero error and like MAE far from it.
**Math (fixed, §7.4.5):**
```
L_δ(Y, Ŷ) = ½(Y − Ŷ)²          if |Y − Ŷ| ≤ δ
           = δ|Y − Ŷ| − ½δ²     otherwise
```

**3. What to implement — the three models, differing only where specified:**

| | Spatial branch | Fusion | Temporal branch |
|---|---|---|---|
| **Model 1 (baseline)** | GCN (Phase 6), static graph | **concat**(S, T) → Linear | PatchTST (Phase 8) |
| **Model 2** | GCN (Phase 6), static graph | **CMA** (Phase 9) | PatchTST (Phase 8) |
| **Model 3 (proposed)** | Adaptive GAT (Phase 7), dynamic wind graph | **CMA** (Phase 9) | PatchTST (Phase 8) |

Notice: Model 1 → Model 2 isolates the effect of CMA (everything else identical). Model 2 → Model 3 isolates the effect of the adaptive wind graph (everything else identical). This is exactly the "change only the evaluated component" design you asked for, and it's why R2 (shared PatchTST across all three) matters — without it, this clean isolation breaks.

Each model ends with: `flatten(Z) → MLP → Ŷ [B, N, τ]`.

**4. Expected input:** A full batch from Phase 4's DataLoader.

**5. Expected output:** `Ŷ`, the multi-step PM2.5 forecast for all stations.

**6. Tensor shapes:**
```
Z or concat(S,T): [B, N, d_model]
Flatten:          [B, N, d_model]  (already per-station; flatten is per-node here, not across N)
MLP head:         d_model -> τ
Ŷ:                [B, N, τ]
Y (ground truth):  [B, N, τ]
```

**7. How to test it:** Overfit each model on a tiny slice (e.g., 20 samples, no val split) for a few hundred steps and confirm training loss goes near-zero — a model that *can't* overfit a handful of samples has a wiring bug, not a generalization problem, and this catches it in minutes instead of after a full training run.

**8. What success looks like:** Three `nn.Module` classes (or one parameterized class + 3 configs), each importable and each able to overfit a tiny batch.

**9. Next:** Phase 11.

---

## Phase 11 — Training Pipeline, Splits, and Hyperparameter Selection

**1. Objective:** A single training script that trains any of the 3 models under **identical** conditions, so differences in results reflect the architecture, not the training setup.

**2. What to learn/understand:** Why a fair comparison requires locking dataset splits, preprocessing, batch size, optimizer, LR schedule, number of epochs (or early-stopping patience), and random seed identically across all 3 runs, varying *only* the model config — this is the actual scientific content of your ablation, more than the architecture itself.

**3. What to implement:** `train.py --config configs/model{1,2,3}.yaml`, using AdamW, a cosine or step LR schedule, gradient clipping (transformers + GAT stacks can spike early in training), early stopping on validation Huber loss, and checkpointing the best-val-loss model. Log train/val loss per epoch (plain CSV or TensorBoard — don't over-engineer this for a first project).

**4. Expected input:** Phase 4's DataLoaders, Phase 10's model.

**5. Expected output:** A trained checkpoint per model + a loss curve per model.

**6. Tensor shapes:** Same as Phase 10, per batch, across epochs.

**7. How to test it:** Run 2 epochs on a small data subset for all 3 models and confirm all 3 loss curves trend downward — this is your "wiring is correct end-to-end" checkpoint before committing to a full multi-hour training run.

**8. What success looks like:** Val loss plateaus and checkpoints save correctly for all 3 models, using literally the same `train.py` invocation with only `--config` changed.

**9. Next:** Phase 12.

---

## Phase 12 — Debugging & Sanity-Check Toolkit

**1. Objective:** A small set of standing checks you re-run whenever something looks wrong, rather than re-deriving your debugging approach from scratch each time.

**2. What to learn/understand:** The most common silent failure modes in ST-forecasting code: (a) train/test temporal leakage (caught in Phase 4), (b) predicting `t` from `t` instead of `t+1..τ` (off-by-one), (c) a spatial branch that's actually ignoring the graph (caught in Phase 6/7's neighbor-perturbation test), (d) a scaler fit on the wrong split, (e) NaN propagation from Phase 5's `ReLU`-zeroed graph rows before you added the self-loop fix.

**3. What to implement:** A `sanity_checks.py` with: gradient-flow check (`sum(p.grad.abs() for p in model.parameters()) > 0` after one backward pass — catches disconnected modules), a "shuffle-baseline" (train on time-shuffled data — if accuracy doesn't degrade meaningfully, your model isn't actually using temporal structure), and a "graph-ablation baseline" (feed an identity/no-graph adjacency — if Model 3's accuracy doesn't drop, the spatial branch isn't contributing).

**4–6.** N/A — this is a diagnostic toolkit, not a data-flow phase.

**7. How to test it:** Deliberately break something (e.g., swap `A_static` for a random matrix) and confirm your sanity checks actually flag the degradation.

**8. What success looks like:** You trust a training run's result because your sanity checks passed, not because the loss curve alone looked reasonable.

**9. Next:** Phase 13.

---

## Phase 13 — Evaluation Metrics & Experimental Protocol

**1. Objective:** Implement MAE, RMSE, MAPE, R² (§8.2/final draft) and run them consistently across the forecast horizons (R5: 1h/6h/12h/24h) for all 3 models.

**2. What to learn/understand — math:**
```
RMSE = sqrt( (1/n) Σ (Y - Ŷ)² )
MAE  = (1/n) Σ |Y - Ŷ|
MAPE = (1/n) Σ |Y - Ŷ| / |Y|  × 100%      # guard against Y≈0 (rare but possible for pollutants)
R²   = 1 − Σ(Y-Ŷ)² / Σ(Y-Ȳ)²
```
Compute these **per horizon** (separately for the 1h-ahead prediction, 6h-ahead, etc.), not just averaged across the whole `τ` window — the synopsis's core claim is that the proposed model wins by a larger margin at longer horizons, so horizon-level breakdown is where your actual result lives.

**3. What to implement:** `evaluate.py` that loads a checkpoint, runs it on the held-out test split, inverse-transforms predictions back to real PM2.5 units via Phase 3's saved scaler (metrics on normalized data aren't interpretable to anyone reading your report), and outputs a metrics table `[model, horizon, MAE, RMSE, MAPE, R²]`.

**4. Expected input:** Trained checkpoints, test DataLoader.

**5. Expected output:** One CSV with all 3 models × 4 horizons × 4 metrics.

**6. Tensor shapes:** Aggregation over `[B, N, τ]` predictions → scalar metrics per (model, horizon).

**7. How to test it:** Sanity-check by evaluating a trivial "persistence" baseline (predict `Ŷ_t+h = Y_t`, i.e., "tomorrow = today") — all 3 real models should beat this comfortably; if one doesn't, something upstream is broken, not just underperforming.

**8. What success looks like:** A complete metrics table, with a persistence-baseline row included for context.

**9. Next:** Phase 14.

---

## Phase 14 — Running the Full Comparison + Ablation Studies

**1. Objective:** Execute the actual experiment: Model 1 vs. 2 vs. 3, plus the ablations your synopsis's §8.3 specifies (w/o wind-graph, w/o Transformer, w/o cross-attention — these map directly onto Model 1/2/3 already, so this phase is largely "make sure your 3-model results ARE your ablation table," not extra work).

**2. What to learn/understand:** How to read your own results honestly — if Model 3 doesn't clearly beat Model 2 at short horizons but does at long horizons, that's a genuine, reportable finding (matches the synopsis's own framing that the advantage should show up "particularly over longer forecast horizons"), not a failure to fix.

**3. What to implement:** A results notebook that runs Phase 13's evaluation for all 3 checkpoints and assembles the final comparison table + horizon-wise line plots (metric vs. horizon, one line per model).

**4–6.** Same as Phase 13, aggregated across models.

**7. How to test it:** Confirm the same test-set date range and scaler were used for all 3 — re-check this explicitly, it's the easiest thing to accidentally desync across separate training runs.

**8. What success looks like:** One table, one plot, that together tell the "does the proposed architecture help, and where" story.

**9. Next:** Phase 15.

---

## Phase 15 — Result Visualization & Attention-Based Interpretability

**1. Objective:** Produce the interpretability artifacts §9 of your earlier draft promised — spatial attention heatmaps and temporal attention visualizations, made possible because Phase 7's `AdaptiveWindGAT` and Phase 9's `CMAFusion` both expose their attention/gate weights.

**2. What to learn/understand:** Why attention weights are a *plausible*, not guaranteed, explanation — they show what the model weighted heavily, not proof of causal importance; present them as "the model's learned emphasis," not as ground-truth physical explanation, especially to a non-ML audience like your mentor's evaluation committee.

**3. What to implement:** For a chosen high-pollution test-set day: plot `α_ij` from Phase 7 as arrows/edge-weights on the Beijing station map (does it visibly track the real wind direction that day?), and plot the gate values `g` from Phase 9 over the forecast horizon (does the model lean more on spatial or temporal information at different horizons?).

**4–6.** Uses tensors already produced by Phases 7 & 9 during a forward pass on chosen test examples — no new data-flow.

**7. How to test it:** Cross-check one attention-heavy station pair against that day's actual recorded wind direction (Phase 1 data) — they should broadly agree; if they never do, revisit Phase 5/7's wind-bearing math.

**8. What success looks like:** 2–3 clear figures you'd be comfortable presenting in your final report or defense.

**9. Next:** Phase 16.

---

## Phase 16 — Report / Thesis Consolidation

**1. Objective:** Assemble Phases 1–15's findings into your final report, following the structure your existing synopsis already sets up (§9/§10 framing on interpretability and societal impact still apply to the finished work).

**2. What to learn/understand:** How to write an honest "Results and Discussion" section — lead with the metrics table (Phase 14), interpret it against your objectives (§6 of the synopsis), then use the interpretability plots (Phase 15) as supporting evidence, not decoration.

**3. What to implement:** N/A — this is writing, not code. Reuse this roadmap's phase structure loosely as your report's Methodology section outline (it already mirrors your synopsis's §7.4 module breakdown).

**4–8.** N/A.

**9. Next:** Done — circle back to CPCB Delhi-NCR validation (§7.1, Dataset B) as a follow-on extension once the Beijing-only pipeline and report are solid, per your own Gantt chart's phase ordering.

---

## A note on pacing

Phases 0–4 are pure data engineering — expect them to take real time even though there's no architecture novelty yet; rushing them is the most common source of bugs that only surface much later (usually as suspiciously good validation accuracy from leakage). Phases 5–10 are where the fixed architecture actually gets built — go slowly there and don't start Phase 9 until Phases 6, 7, and 8 each pass their own unit tests independently. Everything from Phase 11 onward moves fast once the components are verified.
