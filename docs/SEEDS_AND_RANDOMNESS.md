# Seeds and randomness

**What the seed controls, what it deliberately does not, and how many you need.**

The short version: **a seed varies the model, never the data.** Every run — every
configuration, every seed — is trained, validated and tested on exactly the same
rows in exactly the same order of time. That is what makes the paired bootstrap
in `src/bootstrap.py` valid, and it is why seed spread measures *optimisation*
variance rather than getting tangled up with data variance.

---

## 1. What a seed controls

`seed_everything(seed)` in `src/utils/seeding.py` is called once at the top of
every run, before anything else is constructed. It seeds:

| RNG | Reached by | Affects |
|---|---|---|
| Python `random` | `random.seed` | nothing in the current code path; seeded for completeness |
| NumPy global | `np.random.seed` | nothing in training; used by some analysis paths |
| PyTorch CPU | `torch.manual_seed` | weight initialisation, dropout masks |
| PyTorch CUDA | `torch.cuda.manual_seed_all` | the same, on GPU |
| cuDNN | `deterministic=True`, `benchmark=False` | removes kernel-selection nondeterminism |

Downstream, the seed therefore determines exactly four things:

1. **Weight initialisation** — every `nn.Linear`, the attention vectors, the
   station embedding, the graph's `decay_logit` and `self_logit` starting points.
2. **Dropout masks** — the `p=0.1` dropout inside the temporal encoder, spatial
   layers, fusion and head, resampled every forward pass in training.
3. **Edge dropout** — the `p=0.1` Bernoulli mask over non-self graph edges in
   `WindGATLayer`.
4. **Batch shuffling** — `WindowBatcher` holds its own
   `torch.Generator().manual_seed(seed)`, so the permutation of training origins
   each epoch is reproducible and differs between seeds.

## 2. What a seed does *not* control

This list matters more than the one above, because it is what makes the
comparisons valid.

| Fixed regardless of seed | Where it is decided |
|---|---|
| Split boundaries | `SPLIT_DATES` in `build_beijing.py`, by calendar date |
| Which origins are valid | `valid_origins()` — a deterministic function of split, lookback, horizon and the target mask |
| The order of stations | `stations.csv`, sorted once at build time |
| Feature scaling | Fitted at build time on the training slice, saved to `scaler.json` |
| Imputation and climatology | Fitted at build time, on training data only |
| Which targets are masked | The raw observation record |
| Test set contents | Identical rows, identical order, for every model and every seed |

Concretely: at `lookback=48, horizon=24` the origin counts are **train 17,449 /
val 8,761 / test 8,737** for every run in the project. Two `predictions.npz`
files from any two runs are row-aligned by construction, which is exactly what
`paired_interval` checks and refuses to proceed without.

## 3. The seeds themselves

`42, 43, 44` — arbitrary consecutive integers, fixed in `scripts/run_grid.py`.
There is nothing special about them and nothing is drawn *from* a distribution of
seeds. A seed is just an index into PyTorch's RNG stream.

What *does* have a distribution is the **outcome** across seeds, and it comes
from these initialisers:

| Source | Distribution |
|---|---|
| `nn.Linear` weights | Kaiming-uniform, `U(-1/sqrt(fan_in), +1/sqrt(fan_in))` (PyTorch default) |
| `nn.Linear` biases | `U(-1/sqrt(fan_in), +1/sqrt(fan_in))` |
| GAT attention vectors | Xavier-uniform, `U(-sqrt(6/(fan_in+fan_out)), +...)` |
| Station embedding | `N(0, 0.02^2)` |
| `decay_logit`, `self_logit`, `prior_logit` | **Not random** — fixed at 0.5413 so `softplus` starts at exactly 1.0 |
| Dropout / edge dropout masks | `Bernoulli(0.1)`, rescaled by `1/(1-p)` |
| Epoch shuffling | Uniform random permutation of the training origins |

The graph's physical parameters starting at a fixed, meaningful value rather than
a random one is deliberate: the prior begins at full strength with a 20 km decay
scale, so the model starts from the physics and learns how far to depart from it.

## 4. Why three seeds, and what three buys you

Three seeds give a mean and a sample standard deviation (`ddof=1`). That is
enough to answer *"is this difference larger than the run-to-run noise?"* and not
enough for a formal test of the mean. Five would be better for the final headline
table; three is the compromise the change specification settles on, given that
this project is judged on a working system and figures rather than on statistical
depth.

**Report the spread, always.** If two configurations differ by less than the seed
spread, the honest word is *comparable* — do not bold the smaller number.

## 5. Two variances, and why a claim must clear both

This is the part that is easy to get wrong, and `scripts/run_grid.py` now
enforces it.

| Source | Question it answers | Measured by |
|---|---|---|
| **Test-set sampling** | Would this hold on a different year? | Weekly block bootstrap over the 53 independent test weeks |
| **Optimisation** | Would this hold if the optimiser had landed elsewhere? | Spread of the point difference across seeds |

They are independent, and a difference that survives only one of them is not a
result. A model can look significantly better on seed 42 and worse on seed 43;
the bootstrap on seed 42 alone would happily report "significant".

`run_grid.py` therefore runs the paired bootstrap **once per seed** and grades
each comparison:

| Verdict | Condition |
|---|---|
| **robust** | every seed agrees in sign *and* every seed's interval excludes zero |
| **seed-dependent** | some seeds significant, others not, or signs disagree |
| **comparable** | no seed's interval excludes zero |

Only *robust* belongs in the thesis as an improvement. *Seed-dependent* is worth
reporting honestly and is itself informative — it usually means the effect is
smaller than the optimisation noise, which on this dataset is the expected
outcome for the graph component.

## 6. Reproducibility, honestly stated

Two runs with the same seed on the same machine and the same package versions
produce the same numbers. Across machines, PyTorch versions or GPU
architectures, they may not — this is documented PyTorch behaviour, not a bug
here. That is why `env.json` (Python, OS, CUDA, GPU, package versions, git
revision) is written next to every run's metrics.

One known limitation, stated rather than papered over: **`PYTHONHASHSEED` only
takes effect at interpreter startup.** Setting it inside a running process does
nothing, so `seed_everything` warns when it is unset rather than pretending. It
affects only Python's string hashing, which nothing in the training path depends
on. To silence it properly, export it before launching:

```bash
PYTHONHASHSEED=42 python scripts/run_grid.py --seeds 42 43 44
```

For strict determinism at some cost in speed, `seed_everything(seed,
strict=True)` additionally requests deterministic CUDA kernels and sets
`CUBLAS_WORKSPACE_CONFIG`. It is not the default because a handful of operations
have no deterministic implementation and would raise.

## 7. Where seeds are set, in one place

| File | What it seeds |
|---|---|
| `src/utils/seeding.py` | the RNGs themselves |
| `src/train.py` | calls `seed_everything(seed)` first thing in `train()` |
| `src/data/windows.py` | `WindowBatcher` generator, for shuffling |
| `scripts/run_grid.py` | the seed list, `--seeds` |
| `scripts/run_baselines.py` | seeds LightGBM; the other baselines are deterministic |
| `src/bootstrap.py` | its own resampling seed, separate from the model seed |

Note the last row: the bootstrap's resampling RNG is deliberately independent of
the model seed, so the same resampling pattern is applied when comparing
different seeds' predictions.
