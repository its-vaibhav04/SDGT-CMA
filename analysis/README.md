# Pre-implementation analysis

The scripts that produced every number in `SDGT-CMA_Evidence_Review.md`. They read
the raw PRSA CSVs directly and depend only on NumPy, so they run before any of the
project pipeline exists. They are frozen: treat them as the evidence record, not as
project code.

```bash
PYTHONPATH=analysis python analysis/profile_dataset.py
PYTHONPATH=analysis python analysis/wind_transport.py
PYTHONPATH=analysis python analysis/spatial_value_probe.py   # slowest, ~10 min
PYTHONPATH=analysis python analysis/find_episodes.py
```

| Script | Reproduces |
|---|---|
| `common.py` | Shared loader, geometry, ridge probe helper |
| `profile_dataset.py` | Evidence review section 7 (dataset facts) and 9 (persistence baselines) |
| `wind_transport.py` | Section 3: wind convention test, advection signature, transport physics |
| `spatial_value_probe.py` | Section 4 (spatial value, lookback sweep, regime slices) and 6 (statistical power) |
| `find_episodes.py` | Section 8: the concrete case-study dates for the figure set |

The wind convention test in `wind_transport.py` is worth reading first. It
establishes from the data alone that `wd` is a direction-*from*, which is the
single assumption the whole graph module rests on.
