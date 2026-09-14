# SDGT-CMA — Run 2 findings

**The revised config fixed both Run 1 defects. The grid is now interpretable, and what it says is a null result.**

| | |
|---|---|
| **Date** | 9 September 2026 |
| **Run** | 5 configurations × 3 seeds + 7 baselines, Beijing, test year 2016-03 → 2017-02 |
| **Artifacts** | `sdgt_results_run_2/experiments/{runs,figures}` |
| **Cost** | **49 minutes** of Kaggle GPU for the whole grid |
| **Headline** | **Both fixes worked. No component of the architecture produces a robust improvement.** |
| **Superseded by** | nothing yet — but note these numbers were computed on the **27-feature** dataset, before ERA5 boundary layer height was added on 9 Sept 2026. The next run is on 28 features and is not directly comparable. |

---

## 1. The two Run 1 defects are fixed

Both changes did exactly what they were predicted to do.

| | Run 1 | Run 2 | target |
|---|---:|---:|---|
| Train / val loss ratio | 3.0 – 3.9 | **1.13 – 1.57** | ~1.0 |
| Best epoch | 0 – 5 of 11–15 | **0 – 13 of 13–26** | inside the run |
| Parameters | 340,572 | **62,776 – 75,676** | ≤ ~100k |
| MAE h=1, grid range | 16.68 – 21.43 | **10.51 – 10.66** | ≤ 10.30 (persistence) |

The persistence anchor closed the h=1 collapse completely: the grid went from
62–108 % worse than persistence to within 2–3 % of it. The capacity and schedule
changes closed the overfitting gap.

**Gate 1 passes with one asterisk.** Two runs — `s0` and `d0` at seed 43 — still
have their best epoch at 0. They are not undertrained in the Run 1 sense (they
ran 13 epochs and early-stopped on a flat validation curve); they simply never
improved on their first epoch. Section 4 explains why that is a property of the
validation signal rather than of those two runs.

---

## 2. Results

Test MAE in µg/m³, mean ± spread across seeds 42/43/44.

| model | h=1 | h=6 | h=12 | h=24 | params |
|---|---:|---:|---:|---:|---:|
| persistence | 10.30 | 32.03 | 44.34 | 58.13 | — |
| climatology | 60.29 | 60.34 | 60.40 | 60.51 | — |
| seasonal naive 24 h | 58.20 | 58.18 | 58.16 | 58.13 | — |
| ridge | 10.56 | 31.52 | 42.32 | 51.97 | — |
| DLinear | 10.26 | 32.58 | 44.16 | 53.97 | — |
| **LightGBM** | **9.35** | **28.19** | **39.02** | **50.82** | — |
| **t0** temporal only | **10.51 ±0.13** | **30.83 ±0.09** | **42.19 ±0.43** | **53.09 ±0.70** | 62,776 |
| s0 static + concat | 10.65 ±0.02 | 31.49 ±0.54 | 42.70 ±1.40 | 54.11 ±2.12 | 74,424 |
| d0 wind + concat | 10.66 ±0.07 | 31.37 ±0.38 | 42.74 ±0.94 | 54.24 ±1.17 | 74,492 |
| s1 static + cross-view | 10.60 ±0.10 | 31.43 ±0.47 | 43.51 ±0.63 | 53.88 ±1.12 | 75,608 |
| d1 wind + cross-view | 10.56 ±0.06 | 31.38 ±0.50 | 43.11 ±0.19 | 53.58 ±0.49 | 75,676 |

Three things to read off this table.

**`t0` — the model with no graph at all — is the best neural configuration at
every horizon.** It is also the smallest and the most seed-stable. Every spatial
component added on top of it costs mean MAE and roughly doubles-to-triples the
seed spread (t0 ±0.70 at h=24; s0 ±2.12).

**LightGBM still wins at every horizon**, by 1.2 (h=1) to 2.6 (h=6) µg/m³. This
was the expected outcome from the published work on this dataset and it has now
survived a converged grid, so it is a real result rather than an artifact of
undertrained baselines.

**At h=1 every neural model is now a slightly worse persistence.** The anchor
made persistence the zero-output, and the learned correction on top of it is
net-harmful by 0.2–0.4 µg/m³. That is honest and easy to explain: at a 1-hour
lead on a signal with 0.969 autocorrelation there is almost nothing left to add.

---

## 3. Paired comparisons — nothing is robust, in either direction

| comparison | h=1 | h=6 | h=12 | h=24 |
|---|---|---|---|---|
| s0 → d0 *(the graph, alone)* | comparable | comparable | seed-dependent | seed-dependent |
| s0 → s1 *(the fusion, alone)* | seed-dependent | comparable | seed-dependent | seed-dependent |
| d0 → d1 *(fusion on the wind graph)* | seed-dependent | seed-dependent | comparable | comparable |
| t0 → s0 *(adding a graph at all)* | seed-dependent | seed-dependent | seed-dependent | seed-dependent |

Every mean difference is between −1.03 and +0.67 µg/m³, and every one of them is
smaller than the seed spread it sits inside.

Note carefully that this **includes the one negative result Run 1 appeared to
show.** In Run 1, "adding a graph makes h=1 worse by 4.75 µg/m³" was robust. In
Run 2 that comparison is −0.13 and seed-dependent: the effect was overfitting,
not the graph. The Run 1 caution — *"reporting 'the graph does not help' from Run
1 would be wrong"* — was correct, and the corrected number is now in hand.

The honest statement from Run 2 is symmetric and weaker: **no pairwise
comparison in this grid resolves, in either direction.** The `t0 → s0` mean is
negative at all four horizons, but the per-seed differences at h=24 are −3.35,
+1.52, −1.26 — the signs disagree, and the mean is carried by a single bad run
(`s0` seed 42, 57.01).

This is exactly what the pre-registered power analysis said would happen. The
evidence review measured the wind-graph's headroom over a plain network average
at **+0.26 to +0.65 % MAE** before a line of the model was written. Run 2
measures it at approximately zero. Two independent methods, same answer.

---

## 4. New finding — validation cannot rank these models

This is the most important thing Run 2 adds, and it was not visible in Run 1.

| | spread across all 15 runs |
|---|---:|
| `best_val_mae` | 31.60 → 32.51 — **0.91 µg/m³** |
| test MAE at h=24 | 51.97 → 57.01 — **5.04 µg/m³** |

The validation split holds ~52 independent weeks, so by the same weekly block
bootstrap used everywhere else it carries a **±12.6 % interval — about ±4.0
µg/m³** on a value near 32. The entire grid's validation spread is 0.91. The
selection signal is roughly **four times coarser than the differences it is being
asked to resolve.**

The consequence is measurable, not theoretical:

```
corr(best_epoch, test MAE at h=24)  =  +0.656
```

Runs that trained longer scored *worse* on test, while validation reported them
as fine. Four of the five configurations have their worst h=24 result at seed 42,
which is the seed that trained longest in every case (best epoch 13).

This is not a distribution-shift problem — the splits are comparable:

| split | hours | mean PM2.5 | median | p95 | frac > 150 |
|---|---:|---:|---:|---:|---:|
| train | 17,520 | 84.0 | 61.0 | 244 | 17.2 % |
| val | 8,784 | 73.1 | 46.0 | 236 | 12.7 % |
| test | 8,760 | 78.1 | 52.0 | 244 | 14.5 % |

It is a resolution problem. One year of hourly data from a 12-station network
that behaves like a single signal does not contain enough independent
information to separate models this similar.

---

## 5. The components *did* learn this time — and still did not help

This is the finding that makes the null result strong rather than merely
inconclusive, and it comes from `d1_wind_crossview_seed42`'s diagnostics.

In Run 1 the graph's learnable physics and the fusion gate were indistinguishable
from their initial values, which is why Run 1 licensed no conclusion about
either. In Run 2 both moved substantially:

| diagnostic | init | Run 1 | **Run 2** |
|---|---:|---:|---:|
| `prior_strength` | 1.000 | 0.983 | **0.610** |
| `decay_rate` | 1.000 | 0.980 | **0.597** |
| fusion gate, mean | 0.500 | 0.502 | 0.521 |
| fusion gate, **std** | — | 0.088 | **0.194** |
| fusion gate, range | — | narrow | **0.037 → 0.980** |

The gate is the clearest signal. In Run 1 it sat flat at its 0.5 initialisation.
In Run 2 its spread more than doubled and individual channels ran the full range
— per-channel means from 0.28 to 0.70, with extremes at 0.04 and 0.98. The
cross-view fusion is genuinely choosing between the spatial and temporal views
per channel, which is precisely what it was designed to do.

The graph physics moved in an interpretable direction: the model **discounted the
physical prior to 61 % of its initial strength and roughly halved the decay
length**. Told to trust the wind-transport prior and free to disagree, it
partially disagreed.

Meanwhile the geometry stayed physically correct — mean degree 4.63 (top-k 4 plus
self), calm fallback rate 4.0 %, and a **learned mean transport lag of 2.91 h**
against the 2 h advection peak measured independently in the evidence review.

So the null result in §2 and §3 is not "the spatial machinery never trained."
It is: **the machinery trained, behaved physically, developed real internal
structure — and produced no measurable improvement in forecast accuracy.** That
is a far more defensible statement, and it is only available because Run 2
converged.

---

## 6. Should the capacity sweep run next? — **No**

The Run 1 document recommended `configs/ablation/capacity_d{24,32,48}.yaml`
(~2 GPU-hours) to "settle the capacity empirically rather than by judgement."
**Run 2's data withdraws that recommendation.** Three reasons:

1. **The question it answers is no longer live.** The sweep was proposed when the
   train/val ratio was 3.0–3.9× and capacity was the dominant defect. It is now
   1.13–1.57 with early stopping firing at epochs 13–26 of an allowed 60. Nothing
   in Run 2 is capacity-limited.

2. **It cannot produce a trustworthy answer.** The sweep ranks d=24/32/48 by
   validation MAE. §4 shows validation MAE has a ±4.0 µg/m³ resolution against a
   grid-wide spread of 0.91. It would return three numbers inside each other's
   confidence intervals, and any winner picked from them would be noise dressed
   as evidence — the precise failure mode the paired-bootstrap machinery exists
   to prevent.

3. **The prize is smaller than the noise.** Even a perfectly-tuned d would be
   chasing the ~1 µg/m³ that separates the grid configurations, against a seed
   spread of ±0.7 to ±2.6 at h=24 and a 2.3 µg/m³ deficit to LightGBM.

Two GPU-hours spent there buys a table nobody can defend in a viva.

---

## 7. What to run instead

### 7.1 Negative controls — Phase 9 *(do this first; ~20 min GPU)*

This is now the highest-value experiment in the project, and Run 2 is what
promoted it.

The grid's result is "the wind graph does not measurably help on this network."
As it stands that is a *failure to find an effect*, which is the weakest form of
a negative claim. The controls convert it into a *positive finding about the
data*:

- **Wind reversal** — negate `u, v`, re-evaluate `d1`. If performance does not
  degrade, the graph demonstrably was not using wind.
- **Edge permutation, station occlusion, identity adjacency** — the same delta,
  reported through the paired bootstrap.

If reversal changes nothing, the thesis can state, with evidence: *the graph is
inert on this network, and here is why* — 12 stations with mean pairwise
separation 27 km, cross-station PM2.5 correlation 0.887, transport signal peaking
at a 2-hour lag, and 52 % of hours below 1.5 m/s. That is a defensible
contribution. "We tried it and it didn't improve MAE" is not.

**This is now built and tested** — `src/controls.py`, `scripts/run_controls.py`,
25 tests, and both invocations wired into the Kaggle notebook. It carries its own
power references so a row reading "no effect" can be told apart from a test with
no power; `docs/NEGATIVE_CONTROLS.md` explains how to read it. It needs one GPU
session, well under an hour.

§5 is what makes this worth doing now rather than earlier. In Run 1 a null
control would have proved nothing, because nothing had trained. In Run 2 the
graph physics moved and the gate developed structure, so a wind-reversal control
now genuinely discriminates: if a model whose `prior_strength` and `decay_rate`
demonstrably adapted is indifferent to having its wind field negated, the graph
is inert and we can say so.

### 7.2 Delhi acquisition — Phase 8 *(deferred, 9 Sept 2026)*

**Decision: on hold.** Beijing-only for now — run the controls, write up the null
result, and revisit Delhi before the write-up is final. The synopsis still names
CPCB Delhi-NCR, so this is a deferral rather than a scope change, and it should
be reopened deliberately rather than by default.

The case for it, recorded so the decision can be re-taken on the same evidence:

- It is the largest gap between the locked synopsis and what exists.
- It is the only remaining task with an **external dependency that can fail in
  ways we cannot fix** — the CPCB scraper tracks a portal that changes, and ERA5
  needs a Copernicus CDS account that must be registered before any data moves.
  The implementation plan flags this: *"Start Delhi acquisition in week 3, not
  week 10."*
- It is the only remaining experiment that could turn the null result positive.
  Beijing's failure is a scale mismatch, not an implementation defect: 27 km mean
  separation and 0.887 correlation leave almost no station-to-station structure
  to model. Delhi-NCR is materially larger and more heterogeneous, so the
  hypothesis has a chance there it never had in Beijing.

**One thing that must not be deferred silently.** ERA5 boundary layer height was
approved for *Beijing* as well as Delhi (change spec §3.9) and has never been
fetched. Adding it takes the feature count from 27 to 28, which changes the
embedding's input width — every existing checkpoint then refuses to load, and the
grid, baselines and controls all have to be recomputed. That is only about 1.5
GPU-hours, so it is cheap to absorb, but **only if the decision is taken before
the next full grid run.** Taken afterwards, it invalidates the very results being
written up.

### 7.3 Not now

- **Capacity sweep** — §6.
- **Lookback sweep, TCN control** — same resolution ceiling as the capacity
  sweep. They will return "comparable" at every horizon.
- **Chasing LightGBM** — a 2.3 µg/m³ gap on a dataset where gradient boosting is
  the known strong baseline is a finding to report, not a bug to fix. Reporting
  it plainly is worth more than a tuning campaign that closes half of it.

---

## 8. What can now be said

**Established:**

- The pipeline runs end to end on free Kaggle GPU: full grid, 15 runs, 49
  minutes, no leakage, 135 tests passing.
- Both Run 1 defects are diagnosed and fixed, with before/after numbers.
- On Beijing's 12-station network, **no component of the proposed architecture —
  the wind graph, the lag-awareness, or the cross-view fusion — produces a
  measurable improvement over a purely temporal model**, and the temporal model
  does not beat LightGBM.
- This is not a training failure. The graph's learnable physics adapted
  (`prior_strength` 1.00 → 0.61, `decay_rate` 1.00 → 0.60), the fusion gate
  developed real per-channel structure (std 0.088 → 0.194, range 0.04 → 0.98),
  and the learned mean transport lag of 2.91 h matches the independently measured
  2 h advection peak. **The machinery works; the signal is not there.**
- The reason is measured, not guessed: the network is spatially too compact for
  station-to-station advection to carry signal. Predicted before implementation
  (+0.26 to +0.65 % headroom), confirmed after (≈ 0 %).
- **One year of test data cannot resolve differences of this size.** This is a
  property of the dataset, and it bounds what any amount of further tuning on
  Beijing can establish.

**Still open:**

- Whether the graph is inert or merely unhelpful — the controls (§7.1) settle it.
- Whether the architecture behaves differently on a larger, more heterogeneous
  network — Delhi (§7.2) settles it.

---

## 9. Practical note — checkpoints now come back from Kaggle

Run 2's bundle excluded `*.pt`, so `sdgt_results_run_2/` holds predictions,
diagnostics, metrics, curves and configs but **no trained weights**. Everything
in this document was derived without them, and the interpretability diagnostics
did come back as intended (`d1_wind_crossview_seed42/diagnostics.npz`, 250
stratified windows).

The cost was that Run 2's models cannot be controlled after the fact: they no
longer exist outside the Kaggle session that produced them.

**Fixed.** The notebook's packaging step no longer filters checkpoints. At the
Run 2 capacity a checkpoint is about 0.3 MB (63k–76k parameters), so all fifteen
add roughly 5 MB to the archive — there was never a real size argument, and the
exclusion cost an entire experiment. `tests/test_repository.py` now gates against
it coming back, and against the notebook dropping the control step.

So the controls run in the next full Kaggle session alongside the grid, and every
later re-evaluation can happen locally.
