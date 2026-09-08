"""Wind convention verification and the inter-station advection signature.

Reproduces section 3 of SDGT-CMA_Evidence_Review.md. This is the script that
establishes, from the data alone, that `wd` is a direction-from and that
station-to-station transport peaks at a 2-hour lag.

    python analysis/wind_transport.py
"""

from __future__ import annotations

import numpy as np

import common

ORDER = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
MIN_SPEED = 2.0        # only look at hours where transport is plausible
ALIGN_THRESHOLD = 0.7  # cos(angle) above/below this counts as up/downwind
LAGS = (0, 1, 2, 3, 4, 6, 8, 12)


def convention_test(ds) -> None:
    """Beijing geography settles the wind convention without a standards citation.

    Clean air arrives from the north-west (Mongolian plateau); loaded air from
    the south-east (Hebei industrial corridor). If `wd` labelled the direction
    the wind blows *toward*, this table would be inverted.
    """
    print("=== WIND-DIRECTION CONVENTION TEST ===")
    print(f"mean PM2.5 by recorded wd, hours with speed >= {MIN_SPEED} m/s\n")

    pm, speed = ds.pm25, ds.wind_speed
    results = []
    for label in ORDER:
        deg = common.COMPASS[label]
        m = (ds.wind_dir_deg == deg) & np.isfinite(pm) & (speed >= MIN_SPEED)
        results.append((label, float(np.nanmean(pm[m])), int(m.sum())))
        print(f"  wd={label:4s} mean PM2.5 = {results[-1][1]:6.1f}   (n={results[-1][2]})")

    cleanest = min(results, key=lambda r: r[1])
    dirtiest = max(results, key=lambda r: r[1])
    print(f"\n  cleanest: {cleanest[0]} at {cleanest[1]:.1f} | "
          f"dirtiest: {dirtiest[0]} at {dirtiest[1]:.1f} "
          f"({dirtiest[1] / cleanest[1]:.1f}x swing)")
    if cleanest[0] in ("N", "NNW", "NW", "NNE"):
        print("  CONFIRMED: 'wd' is the direction the wind blows FROM.")
        print("  Transport bearing = wd + 180 deg. Adjacency must be [target, source].")
    else:
        print("  UNEXPECTED result -- re-check the convention before building the graph.")


def advection_signature(ds, residualise: bool) -> None:
    """Does an upwind source lead a downwind target, and by how much?

    For each ordered pair (source j -> target i) and each lag, compare how well
    the source predicts the target when j is upwind of i versus downwind of it.
    A positive difference that peaks at a physically plausible lag is the
    fingerprint of advection.

    With ``residualise=True`` the city-wide common factor is regressed out
    first, which is what reveals that almost all of the apparent structure is a
    single shared regime rather than pairwise transport.
    """
    pm, speed = ds.pm25, ds.wind_speed
    _, bearing = common.geometry(ds.stations)
    n, t_len = ds.n_stations, ds.n_hours

    series = pm
    if residualise:
        net = np.nanmean(pm, axis=0)
        series = np.full_like(pm, np.nan)
        for i in range(n):
            m = np.isfinite(pm[i]) & np.isfinite(net)
            design = np.stack([net[m], np.ones(m.sum())], axis=1)
            coef, *_ = np.linalg.lstsq(design, pm[i][m], rcond=None)
            series[i][m] = pm[i][m] - design @ coef

    header = "AFTER REMOVING THE CITY-WIDE COMMON FACTOR" if residualise else "RAW PM2.5"
    print(f"\n=== ADVECTION SIGNATURE -- {header} ===")

    for lag in LAGS:
        aligned, opposed = [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # transport bearing at the source, versus the bearing source->target
                transport = (ds.wind_dir_deg[j] + 180.0) % 360.0
                align = np.cos(np.radians(transport - bearing[j, i]))

                src = series[j][: t_len - lag] if lag else series[j]
                tgt = series[i][lag:] if lag else series[i]
                a = align[: t_len - lag] if lag else align
                s = speed[j][: t_len - lag] if lag else speed[j]

                valid = np.isfinite(src) & np.isfinite(tgt) & np.isfinite(a) & (s >= MIN_SPEED)
                up, down = valid & (a > ALIGN_THRESHOLD), valid & (a < -ALIGN_THRESHOLD)
                if up.sum() > 300:
                    aligned.append(np.corrcoef(src[up], tgt[up])[0, 1])
                if down.sum() > 300:
                    opposed.append(np.corrcoef(src[down], tgt[down])[0, 1])

        diff = np.mean(aligned) - np.mean(opposed)
        print(f"  lag {lag:2d} h  aligned={np.mean(aligned):+.4f}  "
              f"anti-aligned={np.mean(opposed):+.4f}  difference={diff:+.4f}")


def travel_times(ds) -> None:
    print("\n=== IMPLIED TRANSPORT PHYSICS ===")
    dist, _ = common.geometry(ds.stations)
    median_km = np.median(common.upper_triangle(dist))
    closest_km = common.upper_triangle(dist).min()
    speeds = ds.wind_speed[np.isfinite(ds.wind_speed)]

    for q in (25, 50, 75, 90):
        v = np.percentile(speeds, q)
        print(f"  wind p{q:<2d} = {v:.1f} m/s -> {median_km * 1000 / v / 3600:5.1f} h across "
              f"the median {median_km:.0f} km, {closest_km * 1000 / v / 3600:.1f} h across "
              f"the closest {closest_km:.1f} km")

    calm = 100 * (speeds < 1.5).mean()
    windy = 100 * (speeds >= 3.0).mean()
    print(f"\n  hours below 1.5 m/s (near calm): {calm:.1f}%")
    print(f"  hours at or above 3.0 m/s:       {windy:.1f}%")
    print(f"  -> ReLU(align * speed) zeroes the graph for {calm:.0f}% of the dataset;")
    print("     a static-graph fallback below a speed threshold is required.")


def main() -> None:
    ds = common.load()
    convention_test(ds)
    advection_signature(ds, residualise=False)
    advection_signature(ds, residualise=True)
    travel_times(ds)
    print("\nReading: the raw signature peaks at a 2 h lag (+0.045) and is negative at")
    print("lag 0 -- the fingerprint of real advection. It vanishes once the common")
    print("factor is removed, so nearly all inter-station structure is one city-scale")
    print("regime rather than pairwise transport. Use lags of 1-6 h, not same-hour edges.")


if __name__ == "__main__":
    main()
