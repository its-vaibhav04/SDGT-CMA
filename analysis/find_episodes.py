"""Locate the concrete case-study dates used by the figure set.

Reproduces section 8 of SDGT-CMA_Evidence_Review.md. Everything reported here is
inside the test split, so using these dates for figures does not touch training
data.

    python analysis/find_episodes.py
"""

from __future__ import annotations

import numpy as np

import common

SEPARATION_HOURS = 96   # keep selected events from being the same episode twice
N_EVENTS = 6


def main() -> None:
    ds = common.load()
    pm, speed = ds.pm25, ds.wind_speed
    net = np.nanmean(pm, axis=0)
    net_speed = np.nanmean(speed, axis=0)
    n = ds.n_hours

    print(f"TEST SPLIT: {ds.timestamps[common.VAL_END]}  ->  {ds.timestamps[-1]}")

    # ------------------------------------------------------------ severe episodes
    filled = np.where(np.isfinite(net), net, np.nanmean(net))
    rolling = np.convolve(filled, np.ones(24) / 24, mode="same")

    candidates = sorted(range(common.VAL_END, n), key=lambda t: -rolling[t])
    picked: list[int] = []
    for t in candidates:
        if all(abs(t - q) > SEPARATION_HOURS for q in picked):
            picked.append(t)
        if len(picked) == N_EVENTS:
            break

    print("\n=== SEVERE EPISODES (24 h rolling mean of the network) ===")
    print("  best case studies for a forecast-vs-actual figure\n")
    for t in sorted(picked):
        lo, hi = max(0, t - 36), min(n, t + 36)
        print(f"  {ds.timestamps[t]:%Y-%m-%d %H:%M}  24h-mean={rolling[t]:6.1f}  "
              f"peak_hour={np.nanmax(net[lo:hi]):6.1f}  "
              f"mean_wind={np.nanmean(speed[:, lo:hi]):.2f} m/s")

    # ------------------------------------------------------------ clear-out events
    drop = np.full(n, np.nan)
    for t in range(common.VAL_END, n - 12):
        if np.isfinite(net[t]) and np.isfinite(net[t + 12]):
            drop[t] = net[t] - net[t + 12]

    ranked = sorted((t for t in range(common.VAL_END, n - 12) if np.isfinite(drop[t])),
                    key=lambda t: -drop[t])
    chosen: list[int] = []
    for t in ranked:
        if all(abs(t - q) > SEPARATION_HOURS for q in chosen):
            chosen.append(t)
        if len(chosen) == 5:
            break

    print("\n=== CLEAR-OUT EVENTS (largest 12 h drops) ===")
    print("  best case studies for the dynamic-graph map animation\n")
    for t in sorted(chosen):
        print(f"  {ds.timestamps[t]:%Y-%m-%d %H:%M}  PM2.5 {net[t]:6.1f} -> {net[t + 12]:5.1f} "
              f"in 12 h   mean wind over the window = {np.nanmean(net_speed[t:t + 12]):.2f} m/s")

    print("\n=== RECOMMENDED FIGURE TARGETS ===")
    print("  2016-03-04  strongest clear-out at 3.04 m/s -> dynamic-graph animation.")
    print("              If the graph does not visibly swing downwind here, there is a")
    print("              direction bug; this catches it faster than a unit test.")
    print("  2017-01-01  worst episode of the test year (24 h mean 443, peak 522)")
    print("              -> forecast-vs-actual case study.")
    print("  2017-01-28  Chinese New Year: fireworks-driven peak of 607 ug/m3, the")
    print("              highest single hour in the test set and unpredictable from the")
    print("              feature set -> the honest failure case, and the concrete")
    print("              justification for choosing Huber loss.")


if __name__ == "__main__":
    main()
