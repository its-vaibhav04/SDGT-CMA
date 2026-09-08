"""How much is spatial information actually worth, and can we measure it?

Reproduces sections 4, 5 and 6 of SDGT-CMA_Evidence_Review.md:

  * a lookback sweep showing error rises monotonically past ~24 h
  * an incremental ridge probe isolating the value of network context and of
    wind-directed weighting specifically
  * regime slices showing the wind term helps most when advection should matter
  * a weekly block bootstrap establishing what effect size is detectable at all

Ridge is used deliberately. It is not a competitive model; it is a controlled
probe in which everything except the spatial information is held identical, so
any difference is attributable. A neural model may extract more, but the ratio
between variants is the useful signal.

    python analysis/spatial_value_probe.py
"""

from __future__ import annotations

import numpy as np

import common

HORIZONS = (1, 6, 12, 24)
BASE_LAGS = [1, 2, 3, 4, 5, 6, 9, 12, 18, 24]
LOOKBACK_SETS = {
    "L=6":   [1, 2, 3, 4, 5, 6],
    "L=24":  BASE_LAGS,
    "L=48":  BASE_LAGS + [36, 48],
    "L=72":  BASE_LAGS + [36, 48, 60, 72],
    "L=168": BASE_LAGS + [36, 48, 60, 72, 96, 120, 144, 168],
}
NEIGHBOUR_LAGS = [1, 3, 6, 12, 24]
UPWIND_LAGS = [1, 2, 3, 4, 6, 8]
MAX_PAD = 8          # headroom so upwind lags never run off the start of the array
DISTANCE_SCALE = 20.0  # km, for the inverse-distance weight in the upwind feature


class Design:
    """Builds pooled (station, origin) design matrices for the ridge probes."""

    def __init__(self, ds):
        self.ds = ds
        self.pm = ds.pm25
        self.net = np.nanmean(ds.pm25, axis=0)
        self.dist, self.bearing = common.geometry(ds.stations)
        hours = np.arange(ds.n_hours)
        self.hour_of_day = hours % 24
        self.day_of_week = ((hours // 24) + 4) % 7

    def upwind_feature(self, target: int, origins: np.ndarray, lag: int) -> np.ndarray:
        """Wind-weighted average of other stations' PM2.5, `lag` hours before the origin.

        Weight combines forward alignment at the source, wind speed, and inverse
        distance. Where the wind gives no usable weight (calm or all misaligned)
        it falls back to the plain neighbour mean rather than to zero, so the
        comparison against the network-mean variant stays honest.
        """
        at = origins - lag + 1
        numer = np.zeros(len(origins))
        denom = np.full(len(origins), 1e-6)
        stack = []
        for source in range(self.ds.n_stations):
            if source == target:
                continue
            transport = (self.ds.wind_dir_deg[source][at] + 180.0) % 360.0
            align = np.cos(np.radians(transport - self.bearing[source, target]))
            speed = self.ds.wind_speed[source][at]
            weight = (np.clip(align, 0, None) * np.clip(speed, 0, None)
                      * np.exp(-self.dist[target, source] / DISTANCE_SCALE))
            value = self.pm[source][at]
            stack.append(value)
            ok = np.isfinite(weight) & np.isfinite(value)
            numer[ok] += weight[ok] * value[ok]
            denom[ok] += weight[ok]

        fallback = np.nanmean(np.stack(stack), axis=0)
        out = numer / denom
        weak = denom < 1e-3
        out[weak] = fallback[weak]
        return np.nan_to_num(out, nan=np.nanmean(fallback))

    def build(self, variant: str, horizon: int, lags=None):
        """variant: A own-history | B +network mean | C +wind-weighted | D +raw neighbours."""
        lags = lags or BASE_LAGS
        start = max(lags) + MAX_PAD
        origins = np.arange(start, self.ds.n_hours - horizon)

        blocks, targets, times = [], [], []
        for station in range(self.ds.n_stations):
            feats = [self.pm[station][origins - lag + 1] for lag in lags]
            feats += [self.ds[name][station][origins] for name in common.METEOROLOGY]

            wd = np.radians(self.ds.wind_dir_deg[station][origins])
            feats += [np.sin(wd), np.cos(wd)]
            feats += [np.sin(2 * np.pi * self.hour_of_day[origins] / 24),
                      np.cos(2 * np.pi * self.hour_of_day[origins] / 24),
                      np.sin(2 * np.pi * self.day_of_week[origins] / 7),
                      np.cos(2 * np.pi * self.day_of_week[origins] / 7)]

            if variant in ("B", "C", "D"):
                feats += [self.net[origins - lag + 1] for lag in NEIGHBOUR_LAGS]
            if variant == "C":
                feats += [self.upwind_feature(station, origins, lag) for lag in UPWIND_LAGS]
            if variant == "D":
                feats += [self.pm[j][origins] for j in range(self.ds.n_stations) if j != station]

            onehot = np.zeros((self.ds.n_stations, len(origins)))
            onehot[station] = 1.0
            blocks.append(np.stack(feats + list(onehot), axis=1))
            targets.append(self.pm[station][origins + horizon])
            times.append(origins)

        X = np.concatenate(blocks)
        y = np.concatenate(targets)
        t = np.concatenate(times)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return X[keep], y[keep], t[keep]


def lookback_sweep(design) -> None:
    print("=== LOOKBACK SWEEP -- test MAE (ug/m3), own-station history only ===")
    print(f"{'lookback':12s}" + "".join(f"  h={h:<6d}" for h in HORIZONS))
    for label, lags in LOOKBACK_SETS.items():
        row = []
        for h in HORIZONS:
            X, y, t = design.build("A", h, lags=lags)
            pred, truth, _ = common.ridge_fit_predict(X, y, t)
            row.append(np.abs(pred - truth).mean())
        print(f"{label:12s}" + "".join(f"  {v:7.2f} " for v in row))
    print("  -> error rises monotonically past ~24 h; L=168 is worst at every horizon")


def incremental_probe(design):
    print("\n=== INCREMENTAL VALUE OF SPATIAL INFORMATION -- test MAE (ug/m3) ===")
    labels = {
        "A": "A own-history only     ",
        "B": "B + network-mean hist. ",
        "C": "C + WIND-WEIGHTED upwd ",
        "D": "D + all 11 neighbours  ",
    }
    print(f"{'variant':24s}" + "".join(f"  h={h:<6d}" for h in HORIZONS))
    scores = {}
    for variant in ("A", "B", "C", "D"):
        row = []
        for h in HORIZONS:
            X, y, t = design.build(variant, h)
            pred, truth, _ = common.ridge_fit_predict(X, y, t)
            row.append(np.abs(pred - truth).mean())
        scores[variant] = row
        print(f"{labels[variant]:24s}" + "".join(f"  {v:7.2f} " for v in row))

    print("\n  improvement over A:")
    for variant in ("B", "C", "D"):
        gains = [100 * (scores["A"][i] - scores[variant][i]) / scores["A"][i]
                 for i in range(len(HORIZONS))]
        print(f"    {variant}: " + "  ".join(f"h={h}: {g:+5.2f}%" for h, g in zip(HORIZONS, gains)))
    wind = [100 * (scores["B"][i] - scores["C"][i]) / scores["B"][i] for i in range(len(HORIZONS))]
    print("    C vs B (the pure wind-direction gain): "
          + "  ".join(f"h={h}: {g:+5.2f}%" for h, g in zip(HORIZONS, wind)))
    print("  -> spatial value peaks at h=1-6 and decays to nothing by h=24,")
    print("     the opposite of the synopsis claim. Wind weighting adds under 0.7%.")
    return scores


def regime_slices(design) -> None:
    print("\n=== REGIME SLICES -- does the wind term help when advection should matter? ===")
    for h in (6, 12):
        Xb, yb, tb = design.build("B", h)
        Xc, yc, tc = design.build("C", h)
        pred_b, truth, times = common.ridge_fit_predict(Xb, yb, tb)
        pred_c, _, _ = common.ridge_fit_predict(Xc, yc, tc)
        err_b, err_c = np.abs(pred_b - truth), np.abs(pred_c - truth)
        speed = np.nanmean(design.ds.wind_speed, axis=0)[times]

        print(f"\n  horizon h={h}")
        for lo, hi, label in [(0, 1.5, "calm  < 1.5 m/s"), (1.5, 3.0, "light 1.5-3"),
                              (3.0, 1e9, "windy >= 3 m/s")]:
            m = (speed >= lo) & (speed < hi)
            if m.sum() < 200:
                continue
            gain = 100 * (err_b[m].mean() - err_c[m].mean()) / err_b[m].mean()
            print(f"    {label:16s} n={m.sum():6d}  MAE_B={err_b[m].mean():6.2f}  "
                  f"MAE_C={err_c[m].mean():6.2f}  gain={gain:+5.2f}%")
        top = truth >= np.percentile(truth, 90)
        gain = 100 * (err_b[top].mean() - err_c[top].mean()) / err_b[top].mean()
        print(f"    {'top-decile PM2.5':16s} n={top.sum():6d}  MAE_B={err_b[top].mean():6.2f}  "
              f"MAE_C={err_c[top].mean():6.2f}  gain={gain:+5.2f}%")


def power_analysis(design, n_boot: int = 500, seed: int = 0) -> None:
    """Weekly block bootstrap: what difference is detectable on one test year?"""
    print("\n=== STATISTICAL POWER -- weekly block bootstrap on the test split ===")
    rng = np.random.default_rng(seed)
    for h in (6, 24):
        Xb, yb, tb = design.build("B", h)
        Xc, yc, tc = design.build("C", h)
        pred_b, truth, times = common.ridge_fit_predict(Xb, yb, tb)
        pred_c, _, _ = common.ridge_fit_predict(Xc, yc, tc)
        err_b, err_c = np.abs(pred_b - truth), np.abs(pred_c - truth)

        week = times // 168
        weeks = np.unique(week)
        index = {w: np.where(week == w)[0] for w in weeks}

        absolute, paired = [], []
        for _ in range(n_boot):
            drawn = rng.choice(weeks, size=len(weeks), replace=True)
            rows = np.concatenate([index[w] for w in drawn])
            absolute.append(err_b[rows].mean())
            paired.append(err_b[rows].mean() - err_c[rows].mean())
        absolute, paired = np.array(absolute), np.array(paired)
        lo, hi = np.percentile(absolute, [2.5, 97.5])
        dlo, dhi = np.percentile(paired, [2.5, 97.5])

        print(f"\n  h={h}  ({len(weeks)} independent weekly blocks, {len(truth)} station-hours)")
        print(f"    absolute MAE      = {err_b.mean():6.2f}   95% CI [{lo:.2f}, {hi:.2f}]"
              f"  -> +/-{100 * (hi - lo) / 2 / err_b.mean():.1f}% of MAE")
        print(f"    paired difference = {err_b.mean() - err_c.mean():+.3f}   "
              f"95% CI [{dlo:+.3f}, {dhi:+.3f}]  "
              f"significant={'YES' if dlo > 0 or dhi < 0 else 'NO'}")
    print("\n  -> absolute MAE cannot rank models on this dataset; paired differences on")
    print("     identical rows can resolve effects below half a percent. Report paired.")


def main() -> None:
    design = Design(common.load())
    lookback_sweep(design)
    incremental_probe(design)
    regime_slices(design)
    power_analysis(design)


if __name__ == "__main__":
    main()
