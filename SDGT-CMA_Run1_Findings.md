# SDGT-CMA — Run 1 findings

**The first full grid on Kaggle GPU: what it shows, what it does not, and what changes before Run 2.**

| | |
|---|---|
| **Date** | 8 September 2026 |
| **Run** | 5 configurations × 3 seeds + 7 baselines, Beijing, test year 2016-03 → 2017-02 |
| **Artifacts** | `sdgt_results/experiments/{runs,figures}` |
| **Headline** | **The grid is not yet interpretable. Every run overfitted; none converged.** |

---

## 1. What actually happened

The pipeline ran end to end and produced everything it was supposed to: 15 grid
runs, 7 baselines, diagnostics and 14 figures. Nothing crashed, no leakage
surfaced, and the graph diagnostics are physically sensible.

But **all 15 runs overfitted**, and the effect is not marginal:

| | |
|---|---:|
| Train loss at the final epoch | 0.063 – 0.088 |
| Validation loss at the final epoch | 0.241 – 0.275 |
| **Ratio** | **3.0× – 3.9×, in every single run** |
| Best epoch | **0–5** out of 11–15 run |

Training loss fell by roughly 70 % while validation loss *rose from epoch 1
onward*. The checkpoints that were saved and scored are therefore models that had
trained for between one and five epochs.

This was predicted. The evidence review measured ~243 independent 3-day episodes
in the training split and warned that "capacity is the enemy here". The default
carried **340,572 parameters — 1,402 per independent episode**. The warning was
written down and then not acted on when the defaults were set.

---

## 2. Results as measured (read with the caveat above)

Test MAE in µg/m³, mean ± spread across 3 seeds.

| model | h=1 | h=6 | h=12 | h=24 |
|---|---:|---:|---:|---:|
| persistence | 10.30 | 32.03 | 44.34 | 58.13 |
| ridge | 10.56 | 31.52 | 42.32 | 51.97 |
| DLinear | 10.26 | 32.58 | 44.16 | 53.97 |
| **LightGBM** | **9.35** | **28.19** | **39.02** | **50.82** |
| t0 temporal only | 16.68 ±0.16 | 32.13 ±1.09 | 42.90 ±1.17 | 52.49 ±0.30 |
| s0 static + concat | 21.43 ±1.48 | 32.32 ±0.46 | **40.87 ±0.27** | **50.73 ±0.26** |
| d0 wind + concat | 20.69 ±0.21 | **31.83 ±0.37** | 41.29 ±0.79 | 51.41 ±1.01 |
| s1 static + cross-view | 20.31 ±0.57 | 32.05 ±0.53 | 41.94 ±1.29 | 51.89 ±1.45 |
| d1 wind + cross-view | 19.92 ±0.68 | 32.04 ±0.41 | 41.91 ±0.73 | 51.75 ±0.85 |

**LightGBM still wins at every horizon except h=24**, where `s0` edges it by 0.09
µg/m³ — well inside the seed spread, so a tie.

### Paired comparisons

Only one result is robust, and it is negative.

| comparison | h=1 | h=6 | h=12 | h=24 |
|---|---|---|---|---|
| s0 → d0 *(the graph)* | seed-dependent | seed-dependent | seed-dependent | comparable |
| s0 → s1 *(the fusion)* | seed-dependent | seed-dependent | seed-dependent | comparable |
| d0 → d1 *(fusion on the wind graph)* | seed-dependent | comparable | seed-dependent | seed-dependent |
| **t0 → s0** *(adding a graph at all)* | **−4.75, robust** | comparable | seed-dependent | seed-dependent |

The one thing that survives every seed and every bootstrap interval is that
**adding a graph makes h=1 worse by 4.75 µg/m³**. Everything else sits inside the
noise, exactly as the power analysis predicted it would.

---

## 3. Two defects, diagnosed

### 3.1 Overfitting — the dominant problem

Capacity against the data:

| configuration | parameters | per independent episode |
|---|---:|---:|
| **Run 1 default** — d=64, 3 layers, head 256 | **340,572** | **1,402** |
| d=48, 2 layers, head 192 | 165,724 | 682 |
| **Run 2** — d=32, 2 layers, head 128 | **75,676** | **311** |
| d=24, 2 layers, head 96 | 43,708 | 180 |

A second, independent contributor: **the learning-rate warmup was actively
harmful.** It ramps the rate up over five epochs, and the best epoch was 0–2 —
that is, the best checkpoint was always the one taken while the rate was still
smallest. The model was unstable at the full 1e-3.

Corroborating evidence that nothing trained:

- The fusion gate finished at **0.502 ± 0.088**, flat across all patch positions
  — indistinguishable from its 0.5 initialisation. The fusion learned nothing.
- The graph's learnable physics barely moved: `prior_strength` 1.000 → **0.983**,
  `decay_rate` 1.000 → **0.980**.

### 3.2 The h=1 collapse — partly architectural

Every neural model was roughly twice as bad as persistence at one hour ahead.
Measured on the test split:

| | MAE vs truth at h=1 |
|---|---:|
| persistence — the last observed value `y_t` | **10.18** |
| the model (t0) | 16.67 |
| an 8-hour mean of recent history | 23.07 |

And how close the model's forecast sits to each:

| | |
|---|---:|
| model vs `y_t` | 14.03 |
| model vs the 8-hour mean | 18.32 |

The model lands *between* the last hour and the patch average, and closer to the
last hour — so it is extracting some fine-grained recent information, but not
enough. Patch pooling dilutes exactly the quantity that dominates a 1-hour
forecast: PM2.5 autocorrelation at one hour is **0.969**.

---

## 4. What has changed for Run 2

All committed and tested; 135 tests pass.

### Capacity and schedule (`configs/base.yaml`)

| setting | Run 1 | Run 2 | why |
|---|---|---|---|
| `d_model` | 64 | **32** | 340k → 76k parameters |
| `temporal.n_layers` | 3 | **2** | " |
| `head.hidden` | 256 | **128** | " |
| `dropout` | 0.1 | **0.25** | the train/val gap was 3–4× |
| `weight_decay` | 1e-4 | **1e-2** | " |
| `edge_dropout` | 0.1 | **0.15** | " |
| `lr` | 1e-3 | **3e-4** | unstable at 1e-3 |
| `warmup_epochs` | 5 | **0** | the ramp made the best epoch the lowest-rate one |
| `epochs` | 100 | **60** | it never needed 100 |
| `patience` | 10 | **12** | |

### Persistence-anchored head (the h=1 fix)

The head now predicts the **change from the last observed value** rather than the
level:

```
forecast = y_t + head(fused)
```

Persistence becomes the model's zero-output, so it starts from the strongest
available prior and only has to learn the deviation — which is the part that is
actually hard. Metrics are unchanged: the level is reconstructed before anything
is scored, and no baseline is affected.

It is a config flag (`model.head.persistence_anchor`, default on) so it can be
ablated, and two tests pin the arithmetic: a zeroed head must reproduce
persistence exactly, and shifting the anchor must shift the forecast one for one.

### Verified

One real run of `t0` on the revised config, seed 42, CPU, 12 epochs:

| | Run 1 config | Run 2 config | reference |
|---|---:|---:|---|
| parameters | 340,572 | **62,776** | −82 % |
| train / val loss ratio | 3.2 | **1.01** | 1.0 = no overfitting |
| MAE h=1 | 16.68 | **10.51** | persistence 10.30 |
| MAE h=6 | 32.13 | **31.00** | persistence 32.03 |
| MAE h=12 | 42.90 | **41.53** | persistence 44.34 |
| MAE h=24 | 52.49 | **51.88** | persistence 58.13 |

Both defects are fixed. Validation loss is now flat rather than rising, h=1 has
come back to persistence from 62 % worse than it, and every horizon improved —
from a model 5.4× smaller.

**One caveat worth acting on.** Training loss (0.2149) finished *above*
validation loss (0.2123). Some of that is dropout being active during training
and not during validation, but it is a hint that the pendulum may have swung
from over-capacity to slightly over-regularised. The right size is likely
between the two settings, which is what the capacity sweep in §6 is for.

---

## 5. What Run 1 does and does not license us to say

**Can be said now:**

- The pipeline works end to end on free Kaggle GPU. The full grid cost roughly
  **40 minutes** of GPU time — iterating is cheap.
- The graph behaves physically. On real Beijing wind the learned mean transport
  lag is **2.85 h**, matching the 2 h advection peak measured independently in
  the evidence review. Mean degree 4.63 = top-k 4 plus self.
- LightGBM remains the bar, as the published work on this dataset predicted.

**Cannot be said yet:**

- Anything about whether the wind graph helps. Every config was undertrained and
  the graph models overfitted *fastest* (best epoch 0–2 versus 4–5 for the
  temporal-only model), which is the more likely explanation of the h=1 penalty
  than any property of the graph itself.
- Anything about the cross-view fusion. Its gate never left initialisation.

Reporting "the graph does not help" from Run 1 would be wrong.

---

## 6. Next steps

1. **Re-run the grid with the revised config** — `python scripts/run_grid.py
   --seeds 42 43 44`. About 40 minutes of GPU. The gate is that the train/val
   loss ratio drops below roughly 1.5 and the best epoch lands well inside the
   run rather than at 0.
2. **Read the h=1 row first.** If the anchor works, all five configs should be at
   or below persistence's 10.30. If they are not, the problem is not the anchor
   and the patch schedule needs revisiting.
3. **Only then** read the graph and fusion comparisons. They are meaningful only
   once the models converge.
4. **Settle the capacity empirically rather than by judgement.** Run 1 was
   clearly too large; the verification above hints Run 2 may be slightly too
   small. `configs/ablation/capacity_*.yaml` sweeps d=24 / 32 / 48 on the full
   model so the choice is evidence-backed in the thesis instead of asserted.
   Roughly 2 GPU-hours.
5. If overfitting somehow persists, shorten the lookback to 24 h, which the
   evidence review found optimal for the linear probe anyway.

The lookback sweep, the TCN control and the remaining ablations are worth running
after the core grid is trustworthy, not before.
