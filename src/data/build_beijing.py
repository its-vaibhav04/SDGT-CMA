"""Build the Beijing processed dataset from the raw PRSA CSVs.

    python -m src.data.build_beijing

Produces every artifact in the data contract (see ``src/data/contract.py``).
Delhi builds to the same contract, so nothing downstream needs to know which
city it is looking at.

The order of operations matters and is deliberate:

1. Load the raw hourly grid.
2. Reject physically impossible values; flag instrument ceilings but *keep* them.
3. Record observation masks and time-since-observed **before** any filling.
4. Derive wind u/v from raw direction and speed, then impute u/v as ordinary
   continuous channels -- imputing a circular variable like bearing directly
   would produce nonsense at the 0/360 wrap.
5. Impute inputs causally: forward-fill up to 3 h, then training-split
   climatology. Targets are never filled.
6. Fit scalers on the training slice of the **pre-imputation** array, so
   climatology fill cannot drag the statistics around.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from src.data import era5
from src.data.transforms import Scaler, TargetScaler, default_kinds
from src.utils.manifest import build_data_manifest, write_json

RAW_DIR = Path("data/raw/PRSA_Data_20130301-20170228")
# Hand-compiled reference data, not a download, so it lives in the repo rather
# than under the gitignored data/raw tree -- a fresh clone must be able to build.
# The legacy location is still accepted so snapshots taken before the move keep
# working; the canonical path is first.
STATIONS_CSV = Path("configs/stations_beijing.csv")
LEGACY_STATIONS_CSV = Path("data/raw/stations.csv")
OUT_DIR = Path("data/processed/beijing")

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEOROLOGY = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
TARGET = "PM2.5"

# Generous physical bounds. These reject impossible readings only -- they are not
# an outlier filter. A z>3 rule would delete 1.78% of hours, every one of them a
# Severe-AQI episode, which is exactly what an early-warning model exists to predict.
PHYSICAL_BOUNDS = {
    "PM2.5": (0.0, 2000.0),
    "PM10": (0.0, 3000.0),
    "SO2": (0.0, 1000.0),
    "NO2": (0.0, 1000.0),
    "CO": (0.0, 30000.0),
    "O3": (0.0, 1500.0),
    "TEMP": (-50.0, 60.0),
    "PRES": (800.0, 1200.0),
    "DEWP": (-60.0, 50.0),
    "RAIN": (0.0, 500.0),
    "WSPM": (0.0, 60.0),
}

# Values at these levels are instrument saturation, not measurements. Verified in
# the evidence review: PM2.5=999 occurs once alongside genuine 941 and 957
# readings; CO=10000 occurs 56 times against 9900 and 9800.
CEILINGS = {"PM2.5": 999.0, "PM10": 999.0, "CO": 10000.0}

COMPASS = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

SPLIT_DATES = {
    "train": ("2013-03-01", "2015-02-28"),
    "val": ("2015-03-01", "2016-02-29"),
    "test": ("2016-03-01", "2017-02-28"),
}

FORWARD_FILL_LIMIT = 3
TIME_SINCE_CAP = 24


# ---------------------------------------------------------------------- loading
def resolve_stations_path(path: Path) -> Path:
    """Accept the canonical path, or fall back to the legacy one if it is there."""
    if path.exists():
        return path
    if path == STATIONS_CSV and LEGACY_STATIONS_CSV.exists():
        print(f"  [stations] using legacy location {LEGACY_STATIONS_CSV}")
        return LEGACY_STATIONS_CSV
    raise FileNotFoundError(
        f"station table not found at {path} (or {LEGACY_STATIONS_CSV}). It is tracked "
        f"in the repository as {STATIONS_CSV}; if it is missing, the copy of the "
        f"project is incomplete -- re-clone or re-upload it."
    )


def load_stations(path: Path) -> tuple[list[str], np.ndarray]:
    path = resolve_stations_path(path)
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row["station"])
    names = [row["station"].strip() for row in rows]
    coords = np.array([[float(r["lat"]), float(r["lon"])] for r in rows], dtype=np.float64)
    if len(set(names)) != len(names):
        raise ValueError("duplicate station names in stations.csv")
    return names, coords


def load_raw(raw_dir: Path, stations: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[Path]]:
    """Return per-column ``[T, N]`` arrays, wind direction, timestamps, source paths."""
    paths = sorted(raw_dir.glob("*.csv"))
    if len(paths) != len(stations):
        raise FileNotFoundError(f"expected {len(stations)} CSVs in {raw_dir}, found {len(paths)}")

    by_station: dict[str, dict[str, np.ndarray]] = {}
    wind_by_station: dict[str, np.ndarray] = {}
    timestamps: np.ndarray | None = None

    for path in paths:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        name = rows[0]["station"].strip()

        columns: dict[str, np.ndarray] = {}
        for column in POLLUTANTS + METEOROLOGY:
            values = np.full(len(rows), np.nan)
            for index, row in enumerate(rows):
                text = row[column]
                if text not in ("NA", ""):
                    values[index] = float(text)
            columns[column] = values
        by_station[name] = columns

        wind = np.full(len(rows), np.nan)
        for index, row in enumerate(rows):
            wind[index] = COMPASS.get(row["wd"].strip(), np.nan)
        wind_by_station[name] = wind

        if timestamps is None:
            timestamps = np.array(
                [
                    np.datetime64(
                        f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d}"
                        f"T{int(r['hour']):02d}",
                        "h",
                    )
                    for r in rows
                ]
            )

    missing = set(stations) - set(by_station)
    if missing:
        raise ValueError(f"stations.csv lists stations absent from the raw data: {sorted(missing)}")

    values = {
        column: np.stack([by_station[name][column] for name in stations], axis=1)
        for column in POLLUTANTS + METEOROLOGY
    }
    wind_dir = np.stack([wind_by_station[name] for name in stations], axis=1)
    assert timestamps is not None
    return values, wind_dir, timestamps, paths


# ------------------------------------------------------------------ quality
def apply_quality_rules(values: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    """Null out impossible readings; flag saturation without deleting it."""
    report: dict[str, int] = {}
    ceiling_flag = np.zeros_like(next(iter(values.values())), dtype=bool)

    for column, array in values.items():
        low, high = PHYSICAL_BOUNDS[column]
        impossible = np.isfinite(array) & ((array < low) | (array > high))
        report[f"rejected_{column}"] = int(impossible.sum())
        array[impossible] = np.nan

        ceiling = CEILINGS.get(column)
        if ceiling is not None:
            hit = np.isfinite(array) & (array >= ceiling)
            report[f"at_ceiling_{column}"] = int(hit.sum())
            ceiling_flag |= hit

    report["at_ceiling_any"] = int(ceiling_flag.sum())
    return ceiling_flag, report


# --------------------------------------------------------------- imputation
def forward_fill(array: np.ndarray, limit: int) -> np.ndarray:
    """Causal fill: carry the last observation forward at most ``limit`` hours.

    Looks only backwards, which is what makes it usable at a forecast origin.
    Operates on the time axis of a ``[T, N]`` array.
    """
    out = array.copy()
    n_hours = out.shape[0]
    for station in range(out.shape[1]):
        column = out[:, station]
        last_value = np.nan
        age = limit + 1
        for t in range(n_hours):
            if np.isfinite(column[t]):
                last_value = column[t]
                age = 0
            else:
                age += 1
                if age <= limit and np.isfinite(last_value):
                    column[t] = last_value
    return out


def hours_since_observed(array: np.ndarray, cap: int) -> np.ndarray:
    """Hours since the channel was last genuinely observed, clipped at ``cap``."""
    out = np.full(array.shape, float(cap), dtype=np.float64)
    for station in range(array.shape[1]):
        age = cap
        for t in range(array.shape[0]):
            age = 0 if np.isfinite(array[t, station]) else min(age + 1, cap)
            out[t, station] = age
    return out


def climatology_fill(
    array: np.ndarray,
    timestamps: np.ndarray,
    train_slice: slice,
) -> np.ndarray:
    """Fill remaining gaps with a training-split station/month/hour climatology.

    Falls back to station-month, then station, then global -- so a station with
    no training observation in some month still gets a sensible value rather
    than a NaN that would propagate into the features.
    """
    months = timestamps.astype("datetime64[M]").astype(int) % 12
    hours = (timestamps.astype("datetime64[h]").astype(np.int64)) % 24

    out = array.copy()
    train_values = array[train_slice]
    train_months = months[train_slice]
    train_hours = hours[train_slice]

    for station in range(array.shape[1]):
        gaps = ~np.isfinite(out[:, station])
        if not gaps.any():
            continue

        column = train_values[:, station]
        observed = np.isfinite(column)
        global_mean = float(np.nanmean(column)) if observed.any() else 0.0

        by_month_hour: dict[tuple[int, int], float] = {}
        by_month: dict[int, float] = {}
        for month in range(12):
            in_month = observed & (train_months == month)
            if in_month.any():
                by_month[month] = float(column[in_month].mean())
            for hour in range(24):
                cell = in_month & (train_hours == hour)
                if cell.any():
                    by_month_hour[(month, hour)] = float(column[cell].mean())

        for t in np.flatnonzero(gaps):
            key = (int(months[t]), int(hours[t]))
            out[t, station] = by_month_hour.get(
                key, by_month.get(int(months[t]), global_mean)
            )
    return out


# ---------------------------------------------------------------------- build
def compute_split_indices(timestamps: np.ndarray) -> dict[str, dict[str, object]]:
    """Turn the split dates into half-open hour indices on the target axis."""
    days = timestamps.astype("datetime64[D]")
    splits: dict[str, dict[str, object]] = {}
    for name, (start_date, end_date) in SPLIT_DATES.items():
        start = int(np.searchsorted(days, np.datetime64(start_date), side="left"))
        end = int(np.searchsorted(days, np.datetime64(end_date), side="right"))
        splits[name] = {
            "start_index": start,
            "end_index": end,
            "start_date": str(timestamps[start]),
            "end_date": str(timestamps[end - 1]),
        }
    return splits


def build(
    raw_dir: Path = RAW_DIR,
    stations_csv: Path = STATIONS_CSV,
    out_dir: Path = OUT_DIR,
    use_era5: bool = True,
) -> None:
    print(f"Building Beijing dataset -> {out_dir}")
    stations, coords = load_stations(stations_csv)
    values, wind_dir, timestamps, source_paths = load_raw(raw_dir, stations)
    n_hours, n_stations = timestamps.shape[0], len(stations)
    print(f"  loaded {n_hours} hours x {n_stations} stations")

    ceiling_flag, quality_report = apply_quality_rules(values)
    rejected = sum(v for k, v in quality_report.items() if k.startswith("rejected_"))
    print(f"  quality: {rejected} impossible values nulled, "
          f"{quality_report['at_ceiling_any']} station-hours at an instrument ceiling (kept)")

    splits = compute_split_indices(timestamps)
    train_slice = slice(splits["train"]["start_index"], splits["train"]["end_index"])
    print(f"  splits: " + "  ".join(
        f"{name}=[{spec['start_index']},{spec['end_index']})" for name, spec in splits.items()))

    # --- wind vector, derived before imputation so a missing bearing stays missing
    speed = values["WSPM"]
    radians = np.radians(wind_dir)
    wind_u = -speed * np.sin(radians)   # eastward
    wind_v = -speed * np.cos(radians)   # northward
    wind_observed = np.isfinite(wind_u) & np.isfinite(wind_v)

    # --- masks and staleness, recorded before any filling
    raw_channels = {name: values[name].copy() for name in POLLUTANTS}
    obs_masks = {name: np.isfinite(values[name]).astype(np.float32) for name in POLLUTANTS}
    obs_masks["wind"] = wind_observed.astype(np.float32)
    since_observed = {
        name: hours_since_observed(values[name], TIME_SINCE_CAP) for name in ("PM2.5", "PM10")
    }

    # --- target, taken from the pre-imputation array. Targets are never filled.
    target_raw = raw_channels[TARGET].astype(np.float32)
    target_mask = np.isfinite(target_raw)

    # --- causal imputation of the inputs
    channels: dict[str, np.ndarray] = {}
    for name in POLLUTANTS + METEOROLOGY:
        filled = forward_fill(values[name], FORWARD_FILL_LIMIT)
        channels[name] = climatology_fill(filled, timestamps, train_slice)
    for name, array in (("wind_u", wind_u), ("wind_v", wind_v)):
        filled = forward_fill(array, FORWARD_FILL_LIMIT)
        channels[name] = climatology_fill(filled, timestamps, train_slice)

    wind_uv = np.stack([channels["wind_u"], channels["wind_v"]], axis=-1).astype(np.float32)

    # --- calendar
    hours = timestamps.astype("datetime64[h]").astype(np.int64)
    hour_of_day = hours % 24
    day_of_week = ((hours // 24) + 4) % 7      # 2013-03-01 was a Friday
    broadcast = np.ones((1, n_stations))
    calendar = {
        "hour_sin": np.sin(2 * np.pi * hour_of_day / 24)[:, None] * broadcast,
        "hour_cos": np.cos(2 * np.pi * hour_of_day / 24)[:, None] * broadcast,
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7)[:, None] * broadcast,
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7)[:, None] * broadcast,
    }

    # --- optional boundary layer height
    blh = era5.resolve(
        "beijing",
        coords=coords,
        stations=stations,
        n_hours=n_hours,
        start_date=SPLIT_DATES["train"][0],
        end_date=SPLIT_DATES["test"][1],
        enabled=use_era5,
    )

    # --- assemble the feature stack, order fixed by this list
    ordered: list[tuple[str, np.ndarray]] = []
    ordered += [(name, channels[name]) for name in POLLUTANTS]
    ordered += [(name, channels[name]) for name in ("TEMP", "PRES", "DEWP", "WSPM", "RAIN")]
    ordered += [("wind_u", channels["wind_u"]), ("wind_v", channels["wind_v"])]
    if blh is not None:
        ordered.append(("blh", blh.astype(np.float64)))
    ordered += list(calendar.items())
    ordered += [(f"obs_{name}", obs_masks[name]) for name in POLLUTANTS]
    ordered.append(("obs_wind", obs_masks["wind"]))
    ordered += [(f"since_obs_{name}", since_observed[name]) for name in ("PM2.5", "PM10")]
    ordered.append(("at_ceiling", ceiling_flag.astype(np.float64)))

    feature_names = [name for name, _ in ordered]
    stacked = np.stack([array for _, array in ordered], axis=-1)   # [T, N, F]

    # --- scaling, fitted on the training slice of the pre-imputation values.
    # Using the raw channels here means climatology fill cannot shift the statistics.
    fit_source = stacked.copy()
    for index, name in enumerate(feature_names):
        if name in raw_channels:
            fit_source[..., index] = raw_channels[name]
    fit_source[..., feature_names.index("wind_u")] = wind_u
    fit_source[..., feature_names.index("wind_v")] = wind_v

    passthrough = [n for n in feature_names if n.startswith(("obs_", "hour_", "dow_"))]
    passthrough.append("at_ceiling")
    kinds = default_kinds(
        pollutants=POLLUTANTS,
        log_extra=["RAIN", "blh", "since_obs_PM2.5", "since_obs_PM10"],
        passthrough=passthrough,
    )
    scaler = Scaler.fit(fit_source, feature_names, kinds, train_slice)
    features = scaler.transform(stacked)

    target_scaler = TargetScaler.fit(target_raw, train_slice)
    target_scaled = target_scaler.transform(np.where(target_mask, target_raw, 0.0))
    target_scaled = np.where(target_mask, target_scaled, 0.0).astype(np.float32)

    if not np.isfinite(features).all():
        bad = [feature_names[i] for i in np.unique(np.argwhere(~np.isfinite(features))[:, 2])]
        raise ValueError(f"non-finite values remain in features: {bad}")

    # --- write
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", features.astype(np.float32))
    np.save(out_dir / "target_raw.npy", target_raw)
    np.save(out_dir / "target_scaled.npy", target_scaled)
    np.save(out_dir / "target_mask.npy", target_mask)
    np.save(out_dir / "wind_uv.npy", wind_uv)
    np.save(out_dir / "timestamps.npy", timestamps)
    shutil.copyfile(stations_csv, out_dir / "stations.csv")

    write_json(out_dir / "feature_names.json", {"features": feature_names})
    write_json(
        out_dir / "scaler.json",
        {"features": scaler.to_dict(), "target": target_scaler.to_dict()},
    )
    write_json(out_dir / "split.json", splits)
    write_json(
        out_dir / "manifest.json",
        build_data_manifest(
            city="beijing",
            source_files=[*source_paths, stations_csv],
            stations=stations,
            n_hours=n_hours,
            date_range=(str(timestamps[0]), str(timestamps[-1])),
            feature_names=feature_names,
            extra={
                "quality_report": quality_report,
                "target_observed_fraction": float(target_mask.mean()),
                "forward_fill_limit_hours": FORWARD_FILL_LIMIT,
                "era5_boundary_layer_height": blh is not None,
                "splits": splits,
            },
        ),
    )

    print(f"  features {features.shape} ({len(feature_names)} columns)")
    print(f"  target observed: {target_mask.mean() * 100:.2f}% of station-hours")
    print(f"  boundary layer height: {'included' if blh is not None else 'not available'}")
    print(f"  wrote {len(list(out_dir.iterdir()))} artifacts to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--stations", type=Path, default=STATIONS_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-era5", action="store_true", help="skip boundary layer height")
    args = parser.parse_args()
    build(args.raw_dir, args.stations, args.out_dir, use_era5=not args.no_era5)


if __name__ == "__main__":
    main()
