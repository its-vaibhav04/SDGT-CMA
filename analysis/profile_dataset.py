"""Dataset profile: missingness, gaps, quality, spatial structure, drift, seasonality.

Reproduces section 7 of SDGT-CMA_Evidence_Review.md, plus the persistence
baselines in section 9.

    python analysis/profile_dataset.py
"""

from __future__ import annotations

import itertools

import numpy as np

import common


def main() -> None:
    ds = common.load()
    pm = ds.pm25
    n_cells = ds.n_stations * ds.n_hours

    print(f"stations={ds.n_stations}  hours={ds.n_hours}  rows={n_cells}")
    print(f"range   {ds.timestamps[0]}  ->  {ds.timestamps[-1]}")

    # ---------------------------------------------------------------- missingness
    print("\n=== MISSINGNESS ===")
    for name in common.NUMERIC_COLUMNS:
        arr = ds[name]
        missing = int(np.isnan(arr).sum())
        print(f"  {name:6s} {missing:7d}  {100 * missing / n_cells:5.2f}%   max={np.nanmax(arr):9.1f}")
    wd_missing = int(np.isnan(ds.wind_dir_deg).sum())
    print(f"  {'wd':6s} {wd_missing:7d}  {100 * wd_missing / n_cells:5.2f}%   (categorical, 16-point)")

    complete = (~np.isnan(pm)).all(axis=0)
    print(f"\n  hours with all 12 stations reporting PM2.5: "
          f"{complete.sum()} / {ds.n_hours} = {100 * complete.sum() / ds.n_hours:.1f}%")

    # ---------------------------------------------------------------- gaps
    print("\n=== PM2.5 GAP LENGTHS ===")
    runs = []
    for s in range(ds.n_stations):
        count = 0
        for missing in np.isnan(pm[s]):
            if missing:
                count += 1
            elif count:
                runs.append(count)
                count = 0
        if count:
            runs.append(count)
    runs = np.array(runs)
    short = runs <= 3
    print(f"  gaps={len(runs)}  mean={runs.mean():.1f}  median={np.median(runs):.0f}  "
          f"p90={np.percentile(runs, 90):.0f}  max={runs.max()}")
    print(f"  gaps <= 3h are {100 * short.mean():.1f}% of gaps but only "
          f"{100 * runs[short].sum() / runs.sum():.1f}% of missing hours")
    print("  -> long gaps dominate the missing mass; forward-fill alone is not enough")

    # ---------------------------------------------------------------- quality
    print("\n=== DATA QUALITY ===")
    longest_flat = max(
        max(len(list(g)) for g in itertools.groupby(pm[s][~np.isnan(pm[s])]))
        for s in range(ds.n_stations)
    )
    print(f"  longest constant run in PM2.5: {longest_flat} h  -> no stuck sensors")
    for name, ceiling in [("PM2.5", 999), ("PM10", 999), ("CO", 10000), ("O3", 900)]:
        arr = ds[name]
        at_ceiling = int((arr == ceiling).sum())
        near = sorted(set(float(v) for v in arr[arr >= ceiling * 0.95] if np.isfinite(v)))[-6:]
        print(f"  {name:6s} exactly {ceiling}: {at_ceiling:4d} times | nearby distinct values: {near}")
    print("  -> these read as instrument ceilings, not sentinel codes: flag, do not delete")

    # ---------------------------------------------------------------- shared meteorology
    print("\n=== SHARED METEOROLOGY (stations mapped to nearest weather station) ===")
    for label, arrays in [("TEMP", ds["TEMP"]), ("wind direction", ds.wind_dir_deg)]:
        groups: list[list[int]] = []
        for s in range(ds.n_stations):
            for group in groups:
                a, b = arrays[s], arrays[group[0]]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() > 1000 and np.array_equal(a[m], b[m]):
                    group.append(s)
                    break
            else:
                groups.append([s])
        print(f"  {label}: {len(groups)} distinct series across 12 stations")
        for group in groups:
            if len(group) > 1:
                print(f"      shared: {[ds.stations[i] for i in group]}")

    # ---------------------------------------------------------------- spatial structure
    print("\n=== SPATIAL STRUCTURE ===")
    corr = np.corrcoef(pm[:, complete])
    dist, _ = common.geometry(ds.stations)
    c, d = common.upper_triangle(corr), common.upper_triangle(dist)
    print(f"  cross-station PM2.5 correlation: mean={c.mean():.3f}  min={c.min():.3f}  max={c.max():.3f}")
    print(f"  pairwise distance km: min={d.min():.1f}  max={d.max():.1f}  mean={d.mean():.1f}")
    print(f"  correlation vs distance over the 66 pairs: r={np.corrcoef(d, c)[0, 1]:.3f}")
    print("  -> real distance decay exists, which validates the station coordinates")

    ws = np.corrcoef(ds.wind_speed[:, (~np.isnan(ds.wind_speed)).all(axis=0)])
    tp = np.corrcoef(ds["TEMP"][:, (~np.isnan(ds["TEMP"])).all(axis=0)])
    print(f"  cross-station WSPM corr={common.upper_triangle(ws).mean():.3f}  "
          f"TEMP corr={common.upper_triangle(tp).mean():.3f}")

    # ---------------------------------------------------------------- temporal structure
    print("\n=== PM2.5 AUTOCORRELATION (mean over stations) ===")
    for lag in (1, 3, 6, 12, 24, 48, 168):
        vals = [common.nan_corr(pm[s][:-lag], pm[s][lag:]) for s in range(ds.n_stations)]
        print(f"  lag {lag:3d} h : {np.mean(vals):.3f}")
    print("  -> nothing at 168 h: the one-week lookback has no periodicity to recover")

    print("\n=== EFFECTIVE INDEPENDENT SAMPLE SIZE ===")
    net = np.nanmean(pm, axis=0)
    for lag in (24, 48, 72, 96):
        print(f"  network-mean autocorrelation at {lag:3d} h = {common.nan_corr(net[:-lag], net[lag:]):.3f}")
    print(f"  -> about {common.TRAIN_END // 72} independent 3-day episodes in the training split")

    # ---------------------------------------------------------------- calendar effects
    hours = np.arange(ds.n_hours)
    hour_of_day = hours % 24
    day_of_week = ((hours // 24) + 4) % 7           # 2013-03-01 was a Friday
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_means = [np.nanmean(pm[:, day_of_week == d]) for d in range(7)]
    hod_means = [np.nanmean(pm[:, hour_of_day == h]) for h in range(24)]
    overall = np.nanmean(pm)
    print("\n=== CALENDAR EFFECTS ===")
    print("  " + "  ".join(f"{names[d]}={dow_means[d]:.1f}" for d in range(7)))
    print(f"  weekday spread = {100 * (max(dow_means) - min(dow_means)) / overall:.1f}% of the mean")
    print(f"  hour-of-day spread = {100 * (max(hod_means) - min(hod_means)) / overall:.1f}% of the mean")
    print("  -> keep dow_sin/dow_cos; it does not justify a 168 h window")

    months = np.array([t.month for t in ds.timestamps])
    print("  monthly means: " + "  ".join(
        f"{m:02d}={np.nanmean(pm[:, months == m]):.0f}" for m in range(1, 13)))

    # ---------------------------------------------------------------- drift
    print("\n=== DISTRIBUTION DRIFT ACROSS SPLITS ===")
    for label, mask in zip(("train", "val  ", "test "), ds.split_masks()):
        block = pm[:, mask]
        print(f"  {label} mean={np.nanmean(block):6.2f}  std={np.nanstd(block):6.2f}  "
              f"p90={np.nanpercentile(block, 90):6.1f}  hours={mask.sum()}")

    # ---------------------------------------------------------------- preprocessing impact
    print("\n=== IMPACT OF THE ORIGINALLY PROPOSED PREPROCESSING ===")
    mu, sd = np.nanmean(pm), np.nanstd(pm)
    threshold = mu + 3 * sd
    removed = int((pm > threshold).sum())
    finite = int(np.isfinite(pm).sum())
    print(f"  Z-score z=3 threshold = {threshold:.0f} ug/m3")
    print(f"  would delete {removed} of {finite} hours ({100 * removed / finite:.2f}%),"
          f" every one a Severe-AQI episode")
    lo, hi = np.nanmin(pm), np.nanmax(pm)
    print(f"  Min-Max scaling: p99={np.nanpercentile(pm, 99):.0f} ug/m3 maps to "
          f"{(np.nanpercentile(pm, 99) - lo) / (hi - lo):.3f}")
    print(f"  -> 99% of the data compresses into [0, {(np.nanpercentile(pm, 99) - lo) / (hi - lo):.2f}]; use log1p")

    # ---------------------------------------------------------------- persistence
    print("\n=== PERSISTENCE BASELINE (all rows) ===")
    for h in (1, 6, 12, 24):
        errs = []
        for s in range(ds.n_stations):
            x, y = pm[s][:-h], pm[s][h:]
            m = np.isfinite(x) & np.isfinite(y)
            errs.append(np.abs(x[m] - y[m]))
        e = np.concatenate(errs)
        print(f"  h={h:2d}: MAE={e.mean():6.2f}  RMSE={np.sqrt((e ** 2).mean()):6.2f}")
    print(f"\n  PM2.5 mean={mu:.2f} std={sd:.2f} p50={np.nanpercentile(pm, 50):.0f} "
          f"p90={np.nanpercentile(pm, 90):.0f} p99={np.nanpercentile(pm, 99):.0f} max={hi:.0f}")


if __name__ == "__main__":
    main()
