# SDGT CMA Pre Implementation Research and Technical Audit

## Literature Validation Architecture Corrections Experimental Redesign and Repository Readiness

Prepared for Arpita Sharma, Vaibhav Tyagi, Ayesha, and Mr. Sushil Kumar  
Maharaja Surajmal Institute of Technology  
4 September 2026

## Executive Decision

**Decision: conditional go after redesign. Do not implement the current roadmap verbatim.** The project has a sound educational core: multi-site PM2.5 forecasting, explicit spatial structure, long temporal context, controlled ablations, and reproducible experimentation. It can become a strong Bachelor of Technology project. However, the current proposal and roadmap contain four pre-implementation blockers:

1. The claimed novelty is no longer defensible. Closely related work already combines dynamic or wind-aware graphs, temporal Transformers, and explicit spatial-temporal interaction or cross-attention. The July 2026 AST GT paper is especially close to the proposed combination.
2. The roadmap does not implement PatchTST as defined by the original paper. It flattens multivariate features within each patch and treats stations as channels, while PatchTST defines each channel as a univariate series processed with shared weights.
3. The planned cross-attention has no temporal tokens left to attend to. Both branches are collapsed to one token per station before fusion, so the attention operates across stations rather than between temporal events and spatial context.
4. The wind graph has direction, indexing, scale, and causality problems. Wind direction is recorded as the direction from which wind blows; the proposed edge orientation and aggregation convention are inconsistent; only the final graph state is used; travel time is ignored; and the claimed historical-correlation term is absent.

The right response is not to abandon the project. It is to recast the contribution as a **physics-guided, lag-aware dynamic graph forecasting study with a carefully specified cross-view fusion ablation**. The empirical contribution should be the controlled evidence: when does a physically directed graph help, when does explicit cross-view attention help, and do those gains survive strong naive and modern temporal baselines?

The minimum pre-coding action is to replace the architecture section of the roadmap with the specification in this report and approve the seven decisions in the next table.

| Decision | Recommended choice | Why it matters |
|---|---|---|
| Project claim | Methodological study, not fundamentally novel architecture | Avoids a claim contradicted by recent literature |
| Temporal encoder | Multivariate patch transformer, named honestly | Matches the intended tensor construction without misusing the PatchTST name |
| Fusion | Cross-view attention over retained patch tokens | Makes the stated spatial-temporal interaction technically real |
| Dynamic graph | Target-by-source, wind-to direction, lag-aware and normalized | Corrects the physical direction and message-passing semantics |
| Forecast information | Observation-only benchmark; future observed weather is forbidden | Prevents look-ahead leakage and overclaiming operational readiness |
| Experiments | Full graph by fusion factorial plus temporal and naive baselines | Separates the effects of the graph, fusion, and temporal backbone |
| Dataset scope | Beijing primary; Delhi optional after a data-contract gate | Keeps the core deliverable feasible and reproducible |

## What Should Be Preserved

The proposal has several strong choices that should survive the redesign.

- **Multi-site prediction is the correct problem formulation.** The Beijing data provide a complete hourly grid for 12 monitoring sites, so the project can study spatial dependence without inventing synthetic nodes.
- **A one-week lookback is a reasonable hypothesis.** It exposes daily and weekly patterns and is computationally manageable after patching. It should remain a tuned design choice rather than a guaranteed optimum.
- **Direct multi-horizon prediction is preferable to recursive rollout.** A single model can produce all 24 future hours and be evaluated at 1, 6, 12, and 24 hours.
- **The static versus dynamic graph comparison is valuable.** It is one of the clearest research questions available in this scope, provided the dynamic graph is physically and causally defined.
- **Huber loss is a defensible starting point.** It reduces the influence of large residuals without deleting high pollution episodes. The targets still require masks and the result must be compared with a simple loss such as MAE.
- **Ablation-first development is appropriate.** The strongest thesis will explain which component helps under which conditions, rather than merely reporting one final score.

## What Must Change Before Implementation

### Reframe The Research Questions

The current proposal asks whether a new dual-branch architecture can outperform sequential graph-to-recurrent systems. That framing is too broad and rests on an outdated account of the literature. Use the following testable questions instead:

1. Does a lag-aware, wind-directed graph improve 1 to 24 hour PM2.5 forecasts over distance-only and learned static graphs?
2. Does cross-view attention improve over parameter-matched concatenation when both approaches receive the same spatial and temporal tokens?
3. Are improvements concentrated in high-wind transport regimes, pollution episodes, particular seasons, or particular horizons?
4. Does the full model outperform persistence, daily and weekly seasonal naive forecasts, a linear forecasting model, and a temporal-only neural model?
5. Are the gains stable across random seeds, stations, and blocked test periods?

These questions lead to falsifiable results even if the most complex model does not win overall. That is a strength for an undergraduate dissertation.

### Use A Defensible Title And Contribution Statement

The existing title can remain as an internal project label, but the dissertation title should not imply that spatial and temporal branches are separate data modalities. They are two views of the same monitoring data. Two defensible alternatives are:

- **Physics Guided Lag Aware Dynamic Graph Transformer for Multi Site PM2.5 Forecasting**
- **Evaluating Wind Directed Dynamic Graphs and Cross View Attention for Multi Site PM2.5 Forecasting**

A defensible contribution statement is:

> This project develops and evaluates a lag-aware wind-directed graph prior for multi-site PM2.5 forecasting and tests its interaction with patch-based temporal encoding and cross-view attention under a leakage-controlled, reproducible ablation protocol.

Do not claim that the component combination is fundamentally novel. If a novelty statement is required, restrict it to the exact lag-aware graph definition, the local experimental protocol, or a confirmed implementation detail after a final literature search.

## Literature And Citation Audit

### The Novelty Claim Is Not Supported

Several primary sources overlap substantially with the proposal:

- [DP DDGCN](https://doi.org/10.1016/j.scitotenv.2022.154298) used distance and wind to construct dynamic directed graphs for air-quality prediction, modeled incoming and outgoing transport paths, and incorporated future meteorology.
- [MSTGAN](https://doi.org/10.1016/j.ins.2024.121072) combined a multistation Transformer, dynamic spatiotemporal attention, and graph-coupled recurrent units for air-quality forecasting.
- [DSGT](https://doi.org/10.1016/j.neucom.2024.128924) presented a dynamic synchronous graph Transformer for regional air-quality forecasting.
- [MAGICFormer](https://doi.org/10.1016/j.asoc.2025.113033) combined a multidimensional graph attention module, an Informer temporal encoder, and a Cross Decoder that fuses graph and time-series representations through cross-attention.
- [AST GT](https://doi.org/10.1016/j.jenvman.2026.130515), published in July 2026, combines multi-site PM2.5 forecasting, Transformer temporal encoding, multi-source context, a wind-driven dynamic attention bias, and gated temporal-spatial cooperative attention. This is the closest direct challenge to the proposal's novelty claim.
- [AirFlow](https://arxiv.org/abs/2608.09775), an August 2026 preprint, uses a dual-stream air-quality model with gated bidirectional cross-attention, although it does not use graph propagation.

This does not make the project obsolete. It changes the scientific burden. The report must compare its exact graph, tokenization, fusion, data availability, and evaluation design with these works. A project may be valuable because it is reproducible, physically tested, and carefully ablated even when its broad architecture class already exists.

### Several Proposal References Need Correction

| Proposal entry | Finding | Required action |
|---|---|---|
| AGATNet as Wang et al in IEEE TKDE 2022 | The identifiable AGATNet is Dimri, Choi, Salman, Park, and Singh, 2024, in Journal of Geophysical Research Machine Learning and Computation. It bias-corrects CMAQ forecasts rather than solving the same observation-only task. | Replace the citation and describe the task difference. Use [DOI 10.1029 2024JH000244](https://doi.org/10.1029/2024JH000244). |
| AirQFormer as volume 106 in 2024 | Article 106113 is in Sustainable Cities and Society volume 119, February 2025. It is a hybrid regional forecast and bias-correction system, not a direct architectural predecessor for the same sensor-only task. | Correct year and volume and narrow the comparison. Use [the publisher record](https://www.sciencedirect.com/science/article/pii/S2210670724009351). |
| TransNet in Pattern Recognition 2023 | No reliable exact match for the supplied title and bibliographic record was identified. A 2026 transport-informed PM2.5 model called TransNet is a different work. | Remove the entry unless the team can supply a DOI or full paper matching the claimed reference. |
| DSGT as Liu et al in Expert Systems with Applications 2024 | The matching work is by Hanzhong Xia, Xiaoxia Chen, Binjie Chen, and Yue Hu in Neurocomputing volume 616, article 128924, published in 2025 with a 2024 DOI. | Replace the reference with [DOI 10.1016 j.neucom.2024.128924](https://doi.org/10.1016/j.neucom.2024.128924). |
| AirFormer described as a static or learned GCN followed by temporal processing | AirFormer uses efficient spatial and temporal self-attention components within stacked deterministic blocks, not the simplified static sequential GCN characterization in the survey table. | Rewrite the comparison from the [AirFormer paper](https://arxiv.org/abs/2211.15979) and [AAAI publication](https://ojs.aaai.org/index.php/AAAI/article/view/26676). |
| PM2.5 GNN described mainly as a wind-informed static graph | The model combines a knowledge graph with geographic and meteorological information and recurrent temporal modeling. The current one-line comparison loses important detail. | Recheck the claims against the [primary paper](https://arxiv.org/abs/2002.12898). |

The literature table should distinguish four dimensions rather than assign one vague label to each paper: graph construction, temporal operator, spatial-temporal interaction, and forecast information available at prediction time. This prevents unlike systems from being compared as if they solved the same task.

### Baselines Required By The Literature

The baseline suite should include models that can invalidate the need for a complex architecture:

- Persistence and 24-hour and 168-hour seasonal naive forecasts.
- A historical-average baseline by station, hour of day, and month or season.
- Ridge regression or a regularized direct linear forecaster.
- DLinear or NLinear. The [AAAI 2023 DLinear study](https://doi.org/10.1609/aaai.v37i9.26317) demonstrates why a simple direct linear model must be tested before attributing gains to a Transformer.
- A temporal-only LSTM or temporal convolutional network.
- A static graph model such as STGCN and, if feasible, an adaptive graph model such as [Graph WaveNet](https://www.ijcai.org/proceedings/2019/264). The original [STGCN paper](https://www.ijcai.org/Proceedings/2018/0505) is a useful conventional reference, but its traffic setting should be acknowledged.
- The proposed dynamic graph with simple fusion, before adding cross-view attention.

AirFormer reproduction is optional because its national-scale data assumptions and architecture make a faithful comparison more expensive. It is better to implement fewer baselines correctly than many approximate versions.

## Architecture Audit

### PatchTST Is Misidentified

The original [PatchTST paper](https://arxiv.org/abs/2211.14730) defines two core ideas: subseries patches as tokens and channel independence, where each channel is one univariate time series processed with shared embedding and Transformer weights. The roadmap instead proposes, for each station, to concatenate all variables inside a patch and project that multivariate vector. It then shares the encoder across stations and calls stations the channels.

That is a legitimate multivariate patch encoder, but it is not PatchTST channel independence. There are two clean options:

1. **Recommended:** implement the intended tensor operation and call it a multivariate patch transformer. Preserve pollutant and meteorological interactions inside each station patch.
2. Implement true PatchTST channel independence for every scalar feature, then add an explicit feature-mixing stage before or after the shared temporal encoder.

The first option is simpler, matches the current intent, and avoids an unnecessary architecture rewrite. The report and module name should change accordingly.

### The Fusion Layer Loses Time Before Attention

The roadmap reduces the temporal encoder to one vector per station and uses only the last-hour spatial vector. Both inputs to the fusion layer therefore have shape batch by station by embedding. Cross-attention between these tensors can only attend across station tokens. It cannot make the spatial branch attend to a particular historical event, patch, or hour because no temporal token axis remains.

This is the most important tensor-design defect. A corrected design must retain the patch axis:

- Temporal branch output: batch by station by patch by embedding.
- Spatial branch output: batch by station by patch by embedding, produced by aggregating graph information within each patch or at each hour and pooling only inside the patch.
- Cross-view fusion: for each station, temporal patch queries attend to spatial-context patch keys and values, with a mirrored direction only if justified by ablation.
- Fused history: batch by station by patch by embedding.
- Decoder: horizon queries attend to the fused patch history or a carefully pooled representation.

If the patch axis is collapsed first, rename the module cross-branch station attention and remove claims about temporally significant events.

### Cross Modal Is The Wrong Technical Label

Spatial and temporal encoders operate on the same tabular monitoring records. They are different representations or views, not separate modalities such as imagery, text, radar, or satellite observations. Use **cross-view attention**, **spatial-temporal co-attention**, or **cross-branch attention**. If future work adds satellite imagery, CTM fields, or textual incident reports, cross-modal would then be appropriate.

### The Spatial Branch Discards Most Of Its Computation

The roadmap applies the dynamic GAT at every one of 168 hours but retains only the final spatial state. Without recurrent state transfer or temporal aggregation inside the spatial branch, the first 167 graph computations do not influence the spatial representation passed to fusion. This is equivalent to using only the final observed graph for that branch and is both wasteful and inconsistent with the claim of modeling graph evolution.

Retain graph outputs for each patch or hour. A practical implementation is to run the graph layer per hour, pool the graph-informed states within each patch, and keep all patch tokens. A cheaper alternative is to construct one graph prior per patch from aggregated wind vectors and meteorology.

### The Proposed Wind Graph Has A Direction Reversal Risk

The UCI data store wind direction as categorical compass values. The [World Meteorological Organization](https://etrp.wmo.int/pluginfile.php/97653/mod_folder/content/0/Training%20Material/7_1_Wind%20Part%201%20Introduction.pdf) defines wind direction as the direction from which the wind blows. Pollutant transport follows the opposite, wind-to direction.

The graph implementation should use the convention **row equals target and column equals source**. For a source station j and target station i:

- Convert the categorical wind-from direction at source j to degrees.
- Add 180 degrees modulo 360 to obtain wind-to direction.
- Compute the bearing from source j to target i.
- Positive alignment means the source wind points toward the target.
- Store the resulting prior in target row i and source column j, because graph aggregation for target i sums messages from sources j.

The current roadmap defines an edge as source i influencing target j but later aggregates row i from columns j. These two conventions are transposes of each other. A unit test must catch this: under wind from west, a western source should influence an eastern target, not the reverse.

### Travel Time Cannot Be Ignored

Distance and wind speed imply transport delay. For stations separated by tens of kilometres, even a few metres per second corresponds to travel times of hours. A same-time edge is at best a rough association, not a transport model.

Use a lag-aware physical prior. Estimate travel time from source-target distance and source wind speed, clip it to a plausible range, and align the source pollutant state from the corresponding earlier hour or patch. For numerical stability, interpolate between neighbouring discrete lags or use a small lag kernel. If the wind is calm or direction is missing, fall back to the static geographic or learned graph rather than creating an arbitrary transport edge.

This lag-aware design is the most promising place for a project-specific contribution. It must remain a soft inductive bias, not be described as a full atmospheric dispersion model.

### Graph Weights Need Constraints And Normalization

The roadmap uses an exponential distance term with a learnable coefficient and adds a raw wind-speed-weighted prior to attention logits. This creates several risks:

- An unconstrained distance coefficient can become negative, making weights increase exponentially with distance.
- Raw wind speed and learned attention logits are on unrelated scales.
- A positive cosine rule still leaves approximately half of all directed station pairs connected.
- Adding the identity matrix may make diagonal weights exceed one if the diagonal already has a value.
- Bearing on a self-edge is undefined.

Use a nonnegative parameterization such as softplus for distance decay, normalize distance to kilometres divided by a fixed scale, normalize each prior row, and apply a radius or top-k support. Set self-edges explicitly rather than adding to an unknown diagonal. Introduce the prior as a bounded gate or as a logarithmic attention bias with a constrained coefficient. Report the learned gate and graph sparsity.

### The Claimed Correlation Graph Is Missing

The synopsis says graph connectivity uses real-time wind, meteorology, and historical pollutant correlations. The roadmap formula includes only distance, wind alignment, and wind speed. Either remove the correlation claim or implement it explicitly.

If included, calculate lagged PM2.5 correlation using training data only, shrink estimates toward zero, and exclude target-period information. Correlation is not causation; it should be one prior among several, not evidence of physical transport.

### Recommended Model Specification

The corrected model has five layers of responsibility.

1. **Input contract.** Historical pollutant and meteorological values, observation masks, time since last observation, cyclical calendar variables, and static station coordinates. Known future calendar variables are permitted. Future observed meteorology is forbidden.
2. **Physical graph prior.** A target-by-source sparse matrix built from distance, wind-to alignment, lag, and optional training-only lagged correlation. A learned residual adjacency may complement but not silently replace the physical prior.
3. **Spatial token encoder.** One or two shared graph-attention layers per hour or patch, producing one spatial-context token per station and patch. Residual connections, normalization, and edge dropout are required.
4. **Temporal token encoder.** A multivariate patch transformer shared across stations. Positional or relative-time information must remain visible. Patch length and stride are tuned only on validation data.
5. **Cross-view fusion and decoder.** Parameter-matched concatenation and cross-view attention variants receive the same tokens. A direct decoder outputs all 24 future hours; metrics are reported at 1, 6, 12, and 24 hours.

The output target is PM2.5 at every station. If all pollutants are later predicted, that is a separate multi-task experiment and should not be mixed into the core result.

## Data And Preprocessing Audit

### The Local Beijing Dataset Is Complete In Time But Not In Values

The local files contain 420,768 rows, equal to 12 stations by 35,064 hourly timestamps from 1 March 2013 through 28 February 2017. This matches the [UCI repository](https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata), which states that the air-quality measurements come from 12 nationally controlled stations and meteorology is matched from the nearest weather station. The latter point limits how strongly station-level wind can be interpreted as plume transport.

The local audit found the following missingness and observed maxima:

| Variable | Missing rows | Missing percent | Observed maximum |
|---|---:|---:|---:|
| PM2.5 | 8,739 | 2.08 | 999 |
| PM10 | 6,449 | 1.53 | 999 |
| SO2 | 9,021 | 2.14 | 500 |
| NO2 | 12,116 | 2.88 | 290 |
| CO | 20,701 | 4.92 | 10,000 |
| O3 | 13,277 | 3.16 | 1,071 |
| Wind direction | 1,822 | 0.43 | Categorical |

Numeric meteorological variables have roughly 0.08 to 0.10 percent missingness. The extreme pollutant maxima are rare, but rarity alone does not prove error.

### Do Not Delete Pollution Episodes With A Generic Z Score

The roadmap proposes Z-score outlier removal. This risks deleting the episodes the forecast is most valuable for. Air-quality quality assurance relies on calibration, validation flags, instrument behaviour, temporal context, and corroboration rather than a single global statistical threshold. The [US EPA ambient monitoring quality assurance resources](https://www.epa.gov/amtic/ambient-air-monitoring-quality-assurance) are useful methodological guidance even though this dataset is not an EPA network.

Use a rule hierarchy:

1. Reject impossible values and documented missing codes.
2. Flag long constant runs, impossible step changes, and values unsupported by neighbouring times and stations.
3. Preserve plausible high pollution episodes and include a quality flag as an input when appropriate.
4. Report results on all accepted observations and separately on high-pollution regimes.

### Imputation Must Be Causal And Masked

Do not interpolate across a validation or test target using future observations. Do not impute target labels and score the imputed labels as truth.

Recommended policy:

- Split by target time before fitting any transformer, scaler, climatology, or correlation graph.
- For inputs, use forward fill up to a small maximum gap, then training-derived station-hour climatology or a causal spatial estimate.
- Add an observation mask and time-since-observed feature for every imputed channel.
- Keep a target-validity mask and calculate the loss only where the true target exists.
- Run a sensitivity analysis with no imputation beyond the short forward-fill window.

Short linear interpolation may be used only in a retrospective sensitivity experiment and must never use information after the forecast origin in the operational benchmark.

### Scaling And Target Transformation Need Validation

Min-max scaling is sensitive to rare peaks and distribution drift. Fit every transformation on training data only. Compare standard scaling, robust scaling, and a log1p transformation for PM2.5. Per-station scaling can improve optimization but may hide absolute cross-station differences; if used, the station scale parameters must be saved and predictions converted back before metrics are calculated.

Reversible instance normalization may be tested as an ablation, not assumed beneficial. Recent closest work such as AST GT already treats nonstationary normalization as a major component, so the report must distinguish any similar choice.

### Recommended Chronological Split

Use complete seasonal blocks:

- Training targets: 1 March 2013 through 28 February 2015.
- Validation targets: 1 March 2015 through 29 February 2016.
- Test targets: 1 March 2016 through 28 February 2017.

For the first validation or test origin, historical lookback may come from the immediately preceding period, but every forecast target must remain inside its assigned split. Hyperparameters are selected on validation only. The test period is evaluated once after the full protocol is frozen.

Add monthly or seasonal blocked results within the test year. If compute permits, use rolling-origin refits as a robustness analysis. Because adjacent sliding windows overlap heavily, uncertainty intervals should resample whole weekly time blocks containing all stations rather than treating every window as independent.

### Forecast Information Must Be Explicit

The UCI weather columns are observations matched to nearby weather stations. Using their future values at prediction time would create an oracle experiment, not an operational forecast. Define two distinct settings if both are studied:

- **History-only setting:** data through forecast origin plus known future calendar variables. This is the core reproducible benchmark.
- **Forecast-meteorology setting:** archived numerical weather forecasts available at the historical issue time. This requires a separate dataset and provenance record.

Future observed meteorology may be reported only as an explicitly labelled upper-bound oracle. Do not use it in the headline result.

### Delhi Requires A Separate Data Contract

The [CPCB air-quality portal](https://cpcb.gov.in/air-quality-management-portals/) and [continuous monitoring portal](https://app.cpcbccr.com/ccr) are living operational systems. Station availability, units, flags, and coverage can change. Before adding Delhi, freeze:

- Station identifiers and coordinates.
- Exact date range and download timestamp.
- Pollutant and meteorological units.
- Quality-control flags and missing-data codes.
- Licensing or reuse conditions.
- The overlap of features with Beijing.

Train and evaluate Beijing and Delhi separately first. Cross-city transfer, zero-shot evaluation, and fine-tuning are different research questions and should be optional extensions, not assumed generalization evidence.

## Experimental Design

### Use A Full Factorial Core

The roadmap's three models cannot identify the claimed component effects. It omits dynamic graph plus concatenation, and none of its three models removes the Transformer. Use the following core matrix with the same temporal encoder and a matched parameter budget:

| ID | Spatial component | Fusion | Purpose |
|---|---|---|---|
| T0 | None | None | Temporal-only reference |
| S0 | Static distance graph | Concatenation | Conventional graph baseline |
| D0 | Lag-aware wind graph | Concatenation | Isolates the dynamic graph effect |
| S1 | Static distance graph | Cross-view attention | Isolates fusion on a static graph |
| D1 | Lag-aware wind graph | Cross-view attention | Full proposed model |

Add one temporal-backbone ablation, replacing the patch transformer with a TCN or LSTM while retaining the same spatial graph. This tests whether the Transformer is actually necessary.

The closest fair comparison for the cross-view module is D0 versus D1. The closest fair comparison for the dynamic graph is S0 versus D0 and S1 versus D1. Do not compare only S0 with D1 and attribute the combined difference to one component.

### Minimum Credible Baseline Set

At minimum report:

1. Persistence.
2. Seasonal naive at 24 hours and 168 hours, using the applicable lag for each forecast target.
3. Station-hour historical average.
4. Ridge direct multi-output forecast.
5. DLinear or NLinear.
6. Temporal-only TCN or LSTM.
7. T0, S0, D0, S1, and D1.

Graph WaveNet or a faithful STGCN can be added if time permits. Never omit naive baselines; on highly autocorrelated air-quality data they are often more informative than another complex neural comparator.

### Training Protocol

- Predict all 24 hours directly and evaluate hours 1, 6, 12, and 24.
- Use the same data split, input availability, preprocessing, and target masks for every model.
- Tune a small common search space on validation data. Avoid giving the proposed model a much larger search budget.
- Use early stopping on validation MAE averaged across the four reporting horizons.
- Run at least three random seeds; five is preferable for the final core matrix.
- Report mean, standard deviation, parameter count, training time, peak memory, and inference time.
- Save the exact configuration, seed, code version or source hash, dependency lock, scaler state, station ordering, and split manifest with every checkpoint.
- Overfit a tiny batch before full training. Failure to overfit is a model or data bug, not an optimization result.

### Metrics And Statistical Reporting

Primary regression metrics should be MAE and RMSE in micrograms per cubic metre, reported by horizon and station as well as overall. Add WAPE or MASE for scale-aware comparison. Report R squared only as a secondary descriptive statistic.

MAPE is unstable when actual concentrations are near zero and can dominate an average for the wrong reason. The classic forecast-accuracy analysis by [Hyndman and Koehler](https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/) explains these limitations. Prefer MASE or WAPE. If MAPE is retained to match prior work, state the zero-handling rule and a minimum denominator threshold.

Add decision-relevant slices:

- Top decile PM2.5 MAE and RMSE.
- Event precision, recall, and F1 for a predeclared threshold or empirically defined severe regime.
- Performance by season, wind-speed bin, and station.
- Improvement over persistence and daily seasonal naive by horizon.
- Weekly block-bootstrap 95 percent confidence intervals for paired model differences.

Do not compare an hourly value directly with a 24-hour regulatory standard. If a health-standard analysis is included, first aggregate predictions and observations to the standard's averaging period.

### Interpretability Must Not Rely On Attention Alone

Attention maps can be useful diagnostics but are not automatically causal explanations. The primary caution is documented by [Jain and Wallace](https://aclanthology.org/N19-1357/), with an important counterpoint from [Wiegreffe and Pinter](https://aclanthology.org/D19-1002/). Use several tests:

- Remove or permute wind edges and measure performance change.
- Reverse wind directions as a negative control.
- Compare learned high-weight edges with physically aligned source-target pairs.
- Occlude each station or feature group.
- Plot graph sparsity and edge stability by wind regime.
- Test whether gains are larger when advection should matter.

Describe attention visualizations as association diagnostics unless these interventions support a stronger interpretation.

## Repository Readiness Audit

### Current State

The repository is a Phase 0 and partial Phase 1 skeleton, not an implementation-ready research codebase. The spatial, temporal, fusion, full-model, training, and evaluation modules are placeholders. This is useful because the architecture can still be corrected without expensive rework.

The current working copy also has reproducibility and execution gaps:

- It is not currently a Git worktree, so there is no commit history or immutable source identifier.
- `requirements.txt` is unpinned and `pyproject.toml` declares no dependencies or build backend.
- Documentation tells the user to run `.venv`, while the present directory is named `venv`; `.gitignore` ignores `.venv` but not `venv`.
- The recorded Phase 0 verification says three tests passed with a future-looking Torch 2.13 CPU build, while the current suite contains six tests and the present interpreters lack required packages.
- No Phase 1 output directory or manifest is present.
- The coordinate CSV is encoded as Windows 1252 and contains degree symbols, while the loader uses pandas' default UTF-8 decoding. In the current copy, this can fail before the header-renaming logic runs.
- Tests verify row count, schema, and random-number repetition, but not missingness, duplicates, causal windows, station order, graph direction, masks, leakage, or end-to-end trainability.

### Required Phase Zero Repair

Before model development:

1. Initialize Git or place the project in its intended repository, commit the audited starting point, and record a source hash in experiment metadata.
2. Choose one environment convention. Use `.venv` consistently and ignore both `.venv` and `venv` during transition.
3. Move runtime and development dependencies into `pyproject.toml`, pin them through a lock file, and document the supported Python and PyTorch versions. Official [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) notes that complete reproducibility is not guaranteed across releases or platforms, so environment capture is part of the result.
4. Convert `stations.csv` to UTF-8 with ASCII-safe headers such as `station,lon,lat`, or open it with an explicit encoding once and immediately write a normalized UTF-8 artifact.
5. Create a data manifest containing file names, sizes, checksums, source URLs, download dates, row counts, station order, and date ranges.
6. Add a command that builds data from raw inputs and a separate command that validates existing artifacts without rewriting them.
7. Make the test suite runnable from a clean clone with one documented install command.

### Reproducibility Utility Gaps

The current seed helper covers Python, NumPy, PyTorch CPU and CUDA seeds and cuDNN deterministic mode. Extend it to:

- Call `torch.use_deterministic_algorithms` when strict mode is requested.
- Configure deterministic DataLoader generators and worker initialization.
- Record the seed per run rather than hard-code one value in entry points.
- Set CUDA workspace configuration where required and document any operation that cannot be deterministic.
- Capture Python, OS, CUDA, cuDNN, GPU, PyTorch, NumPy, pandas, and package-lock information.

Setting `PYTHONHASHSEED` inside a running Python process does not retroactively change the interpreter's hash seed. Set it before process startup in the run launcher or document that limitation.

### Minimum Test Gates

| Gate | Required test |
|---|---|
| Data integrity | Exact station and timestamp grid, duplicates, units, encoding, checksums, missingness report |
| Leakage | Scalers, imputers, correlation graph, and climatology fit only on training targets |
| Windowing | First and last origin in each split, horizon containment, valid target masks |
| Graph physics | West-to-east synthetic wind test, transpose test, calm-wind fallback, missing-wind fallback, lag and distance limits |
| Tensor semantics | Shape checks preserve station and patch axes through fusion |
| Numerical safety | No NaN or infinity under extreme distance, wind, and missing inputs |
| Training | Tiny-batch overfit, checkpoint round trip, resumed run equivalence where feasible |
| Evaluation | Hand-calculated metric fixtures including zero targets and missing labels |

The roadmap's proposed test that zeroes one branch and expects an untrained gate to shift toward the other branch is invalid. An untrained gate has no reason to behave semantically. Test algebraic limits by explicitly setting gate logits or weights.

## Revised Implementation Roadmap

The order below is designed to fail cheap and preserve scientific validity.

| Phase | Deliverable | Exit condition |
|---|---|---|
| 0 Scientific specification | Approved task definition, information set, tensor shapes, graph direction, hypotheses, and claim language | Every tensor and forecast-time input is unambiguous |
| 1 Repository repair | Reproducible environment, Git history, locked dependencies, data manifest, UTF-8 coordinates | Clean installation and tests pass on a fresh environment |
| 2 Data contract | Causal windows, masks, split manifest, EDA, imputation and scaling artifacts | Leakage tests and data-integrity report pass |
| 3 Naive and simple baselines | Persistence, seasonal naive, historical average, ridge, DLinear or NLinear | Scores reproduced from saved configs and unit-tested metrics |
| 4 Temporal backbone | TCN or LSTM plus multivariate patch transformer | Tiny-batch overfit and temporal-only T0 results |
| 5 Static graph | Distance graph and static graph encoder | Direction and shape tests pass; S0 benchmark complete |
| 6 Dynamic graph | Lag-aware wind prior, learned residual option, graph diagnostics | Synthetic physics tests pass; D0 benchmark complete |
| 7 Fusion | Concatenation and cross-view attention over retained patch tokens | Parameter-matched S1 and D1 runs complete |
| 8 Controlled experiments | Core factorial across at least three seeds | Mean, spread, compute, and paired intervals reported |
| 9 Robustness and explanation | Regime slices, negative controls, edge and feature interventions | Claims supported or narrowed based on evidence |
| 10 Delhi extension | Frozen Delhi data contract and separate benchmark | Only starts if Beijing core and provenance gates pass |
| 11 Release and dissertation | Reproduction instructions, model cards, results tables, limitations | A clean clone reproduces one headline table |

### Suggested Semester Scope

For a Bachelor of Technology timeline, make Beijing the committed deliverable. The minimum thesis-worthy result is phases 0 through 8 with strong documentation. Phase 9 adds depth. Delhi and cross-city transfer are stretch goals. A smaller experiment grid with trustworthy causal preprocessing is more valuable than a broad two-city claim built on inconsistent inputs.

### Go And No Go Gates

- **Go to model work** only after the environment, encoding, split, masks, and baseline metrics are reproducible.
- **Go to dynamic graph work** only after the temporal-only and static-graph models can overfit a tiny batch and beat at least one naive baseline on validation.
- **Go to cross-view attention** only after D0 establishes that the corrected dynamic graph is numerically stable.
- **Go to Delhi** only after the Beijing core matrix is complete and the Delhi data contract is frozen.
- **No claim of improvement** unless it holds across seeds and paired blocked uncertainty intervals, or is explicitly described as exploratory.
- **No claim of physical interpretability** unless direction-reversal and edge-intervention controls behave as predicted.

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Novelty claim challenged by very recent work | High | High | Reframe as lag-aware graph and rigorous ablation study; cite AST GT and adjacent systems |
| Future weather leakage | Medium | High | Freeze history-only information set; label any future-observation experiment as oracle |
| Wind graph points in the wrong direction | Medium | High | Use wind-from to wind-to conversion and synthetic west-to-east unit tests |
| Cross-attention does not use temporal tokens | High in current design | High | Preserve patch axis through both branches and document every tensor shape |
| Complex model loses to seasonal naive or DLinear | Medium | Medium | Treat as a valid result; study regime-specific value and simplify claims |
| Missing data inflate performance | Medium | High | Causal imputation, input masks, target masks, and sensitivity runs |
| Extreme pollution episodes removed as outliers | Medium | High | Quality-control rules and flagged retention rather than blanket Z-score deletion |
| Delhi data instability expands scope | High | Medium | Make Delhi a gated extension with frozen files and provenance |
| Environment cannot reproduce recorded results | High today | High | Lock dependencies, capture hardware and versions, use clean-install CI |
| Attention maps overinterpreted | Medium | Medium | Add edge reversal, permutation, occlusion, and regime tests |

## Recommended Thesis Claims

The final thesis should make only claims supported by the completed experiments. Plausible claim forms are:

- A lag-aware wind prior improved short-horizon forecasts under directional high-wind conditions but provided little benefit in calm conditions.
- Cross-view attention improved or failed to improve over parameter-matched concatenation after both methods received the same patch-level tokens.
- The graph component helped particular stations or seasons rather than uniformly improving the network average.
- A simpler temporal model matched the Transformer overall, while the graph prior added value during transport-sensitive episodes.
- The physically constrained graph was more stable or interpretable than an unconstrained learned adjacency even when average error was similar.

Avoid claims that attention proves causal pollutant transfer, that one city establishes broad generalization, or that lower error alone makes the system an operational early-warning service.

## Immediate Action List

1. Freeze implementation beyond the current data-loading skeleton.
2. Approve the revised research questions, title language, information set, and tensor shapes.
3. Correct the bibliography and rewrite the literature comparison table.
4. Repair the repository environment and coordinate-file encoding.
5. Implement target-time splits, causal masks, data manifests, and naive baselines.
6. Rename the intended temporal encoder to multivariate patch transformer.
7. Specify the dynamic graph with target-by-source indexing, wind-to direction, lag, sparsity, and bounded priors.
8. Preserve patch tokens through the spatial branch and fusion.
9. Run the full factorial model matrix with identical preprocessing and tuning budgets.
10. Add Delhi only after the Beijing evidence package is complete.

## Source Ledger

| Claim supported | Primary or authoritative source | Evidence strength |
|---|---|---|
| Beijing dataset scope, dates, station count, variables, missing-value convention, and nearest-weather-station matching | [UCI Beijing Multi Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata) | Direct dataset record |
| Wind direction is reported as the direction from which wind blows | [World Meteorological Organization wind training material](https://etrp.wmo.int/pluginfile.php/97653/mod_folder/content/0/Training%20Material/7_1_Wind%20Part%201%20Introduction.pdf) | Official meteorological guidance |
| PatchTST channel independence means one univariate series per channel with shared weights | [PatchTST primary paper](https://arxiv.org/abs/2211.14730) | Direct architecture definition |
| Dynamic wind and distance graphs predate the proposal | [DP DDGCN](https://doi.org/10.1016/j.scitotenv.2022.154298) | Direct primary paper |
| Dynamic graph and Transformer air-quality models predate the proposal | [MSTGAN](https://doi.org/10.1016/j.ins.2024.121072) and [DSGT](https://doi.org/10.1016/j.neucom.2024.128924) | Direct primary papers |
| Cross-attention fusion of graph and time-series features predates the proposal | [MAGICFormer](https://doi.org/10.1016/j.asoc.2025.113033) | Direct primary paper |
| A nearly matching wind-aware spatiotemporal Transformer exists in 2026 | [AST GT](https://doi.org/10.1016/j.jenvman.2026.130515) | Direct publisher and PubMed record |
| Simple direct linear models are necessary forecasting baselines | [DLinear AAAI paper](https://doi.org/10.1609/aaai.v37i9.26317) | Direct primary paper |
| MAPE has known weaknesses and MASE is a robust alternative | [Hyndman and Koehler](https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/) | Primary methodological paper |
| Attention weights are contested as explanations | [Jain and Wallace](https://aclanthology.org/N19-1357/) and [Wiegreffe and Pinter](https://aclanthology.org/D19-1002/) | Primary paired methodological papers |
| Reproducibility depends on releases, platforms, and deterministic settings | [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Official framework documentation |
| CPCB is the authoritative Delhi monitoring portal | [CPCB portal index](https://cpcb.gov.in/air-quality-management-portals/) and [continuous monitoring portal](https://app.cpcbccr.com/ccr) | Official government sources |
| Air-monitoring outlier decisions require quality-assurance context | [US EPA ambient monitoring quality assurance](https://www.epa.gov/amtic/ambient-air-monitoring-quality-assurance) | Official quality-assurance guidance |

## Final Assessment

SDGT CMA should proceed, but under a corrected scientific contract. The strongest version of the project is not a race to assemble more modules. It is a disciplined investigation of physical graph priors and spatial-temporal fusion under honest information constraints. The current repository is early enough that this pivot is inexpensive. Once the specification, data contract, and baseline gates are fixed, implementation can begin with far less risk of producing a polished but uninterpretable comparison.
