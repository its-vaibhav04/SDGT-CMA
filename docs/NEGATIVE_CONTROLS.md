# Negative controls

**What the suite does, how to run it, and what each row licenses you to write.**

| | |
|---|---|
| **Code** | `src/controls.py`, `scripts/run_controls.py` |
| **Tests** | `tests/test_controls.py` (22) |
| **Cost** | one test-split pass per control; ~7 passes per run, plus 13 more with `--occlusion` |
| **Needs** | `checkpoint.pt` in the run directory |

---

## 1. Why this exists

Run 2 found that no spatial component — the wind graph, the lag awareness, or
the cross-view fusion — improved MAE over a purely temporal model. Taken alone
that is a **failure to find an effect**, which is the weakest form of a negative
claim. It is equally consistent with two very different worlds:

- the graph is **inert**: it computes something physical and the model ignores it;
- the graph **works** but the benefit is smaller than one test year can resolve.

Run 2 also showed the components are not simply untrained — `prior_strength`
moved 1.00 → 0.61, `decay_rate` 1.00 → 0.60, and the fusion gate developed real
per-channel structure. So something *is* happening. The controls are what
separate those worlds.

The sharp question a control asks is not "did it help?" but **"does the model
depend on this at all?"** If the wind field can be reversed — every transport
direction turned through 180° — and the forecast does not move, then the graph
was never using wind, and no attention heatmap rescues the claim.

---

## 2. The power problem, and how the suite solves it

This is the part that is easy to get wrong, so it is built into the tool rather
than left to the reader.

The head is persistence-anchored:

```
forecast = y_t + head(fused)
```

**No control perturbs the anchor.** So the most any input perturbation can
possibly move is the *learned correction* sitting on top of persistence. If that
correction is worth 4 µg/m³ and wind reversal moves it by 0.01, that is a strong
statement. If the correction were worth 0.02 in the first place, the same 0.01
would mean nothing at all.

A control reporting "no effect" is therefore **uninterpretable in isolation**.
Two reference rows supply the missing scale, and the report prints every other
control as a percentage of them:

| reference | what it measures |
|---|---|
| `zero_correction` | zero the head entirely. The gap to the unperturbed model is **what the learned correction is worth in total** — the denominator. |
| `history_shuffle` | scramble the lookback hours. The largest damage any *input* perturbation can realistically do — the practical ceiling. |

If a graph control reads zero *and* `history_shuffle` also reads zero, you have
learned nothing about the model and everything about the test having no power.
Read the references first, every time.

---

## 3. The controls

| control | what it breaks | what a null result means |
|---|---|---|
| `wind_reversal` | negates `wind_uv`, the field the **graph** reads | the graph does not use wind direction |
| `edge_permutation` | relabels each edge's source by a fixed derangement | it does not matter *which* stations a target attends to |
| `identity_adjacency` | cuts every cross-station edge | cross-station information contributes nothing |
| `wind_reversal_full` | negates wind in the graph input **and** the feature columns | the whole model, not just the graph, ignores wind |
| `station_occlusion` | zeroes one station's inputs at a time | that station's observations are worth nothing to the others |

### The isolation that makes `wind_reversal` a *graph* test

`wind_u` and `wind_v` are model features, and `wind_uv` is passed separately to
the graph. In `src/models/sdgt.py`, `wind_uv` reaches `self.graph(wind_uv)` and
nothing else. Negating only `wind_uv` therefore perturbs the graph while the
temporal branch keeps seeing the true wind — a single-component test.
`wind_reversal_full` negates both and asks the whole-model question instead.

That isolation is pinned by a test: a **static** distance graph ignores wind, so
`wind_reversal` on an S0 model must produce a *bitwise identical* forecast. If
anyone later routes `wind_uv` into the temporal branch, that test fails
immediately instead of quietly turning every D1 control into a whole-model
measurement.

### Design details that matter

- **Derangement, not shuffle.** `edge_permutation` uses a permutation with no
  fixed point, so no station is accidentally left attending to itself — which
  would weaken the control by exactly the fraction it failed to move.
- **Row sums are preserved.** Permuting entries within a row cannot change their
  total, so the adjacency stays a distribution over sources. The control tests
  "right weights, wrong stations", not "malformed graph".
- **Reflection, not negation.** Features are standardised, so negating the
  *stored* value reverses the wind **and** shifts its mean. The correct column
  for `-x` is `-z - 2μ/σ`. Dropping that shift would confound the control with an
  offset; a test pins the formula against a round trip through the real `Scaler`.
- **The occluded station is excluded from its own scoring.** Its forecast
  collapses towards the anchor whatever the graph does, so including it would
  measure the anchor and hide the quantity of interest — what station *j* is
  worth to everybody else. The reference is scored on the same reduced mask.
- **Nothing trains, nothing mutates.** Controls swap modules inside a
  `try/finally` and perturb inputs; a test asserts the weights are unchanged
  afterwards, and another asserts the swaps are undone even when the body raises.

---

## 4. Running it

```bash
# the full model, all three seeds, plus per-station occlusion
python scripts/run_controls.py --occlusion --runs \
    experiments/runs/d1_wind_crossview_seed42 \
    experiments/runs/d1_wind_crossview_seed43 \
    experiments/runs/d1_wind_crossview_seed44

# the temporal-only contrast: no graph, so only whole-model controls apply
python scripts/run_controls.py --runs experiments/runs/t0_temporal_only_seed42

# wiring check in seconds, never a reported result
python scripts/run_controls.py --limit 400 --runs <run_dir>
```

Each run writes `controls.json` beside its checkpoint (`controls_limited.json`
under `--limit`, so a smoke test can never overwrite a real result). Both
invocations are already in `notebooks/kaggle_run_grid.py`.

**Checkpoints are required.** Bundles produced before September 2026 excluded
`*.pt`, which is why Run 2 could not be controlled after the fact; the notebook
now keeps them (~0.3 MB each at the current capacity, ~5 MB for all fifteen), and
`tests/test_repository.py` gates against the exclusion coming back.

---

## 5. Reading the output

```
      control                     h=1              h=6             h=12             h=24
  ------------------------------------------------------------------------------------
      wind_reversal         +0.002  [-0.01,+0.02]  ...
  REF zero_correction       +0.140 *[+0.11,+0.17]  ...
```

- **Positive = the perturbation made the forecast worse**, which is what a real
  dependency looks like. The sign is flipped from `paired_interval`'s convention
  so the table reads naturally.
- `*` marks a 95 % paired weekly-block-bootstrap interval that excludes zero.
- `REF` marks a reference row — a scale, not evidence of a dependency.

Then the summary states each control as a share of the `zero_correction`
denominator at h=24, which is the sentence that goes in the write-up.

**Across seeds.** With more than one run the script prints a final table with the
mean, standard deviation and detection count at h=24. A control that fires for
one seed out of three has not found anything — the same standard the grid's
paired comparisons are held to.

---

## 6. What the results will license

Write the conclusion from the pattern, not from any single row.

**If the graph controls are null and the references are not** — the expected
outcome given Run 2 — the claim is strong and specific:

> The wind graph is inert on this network. Reversing every transport direction,
> relabelling every edge, and cutting every cross-station edge each changed test
> MAE by under X µg/m³, against a learned correction worth Y µg/m³ that the same
> apparatus detects at Z µg/m³ when temporal order is destroyed. This is
> consistent with the network geometry measured before implementation: 12
> stations, 27 km mean separation, 0.887 cross-station PM2.5 correlation, a
> transport signal peaking at a 2 h lag, and 52 % of hours below 1.5 m/s.

That is a contribution — a measured, bounded, mechanistically explained negative
result. "We tried it and it didn't improve MAE" is not.

**If a graph control does fire**, the null result in `SDGT-CMA_Run2_Findings.md`
needs revisiting: the graph would be carrying signal that the MAE comparison is
too coarse to resolve, which points at the resolution ceiling in §4 of that
document rather than at the architecture.

**If the references are also null**, stop and fix the apparatus before writing
anything. That would mean the learned correction is negligible and the model is
effectively persistence — a finding about the training setup, not the graph.
