"""Shared loading and geometry helpers for the pre-implementation analysis scripts.

These scripts back every number in ``SDGT-CMA_Evidence_Review.md``. They read the
raw PRSA CSVs directly and depend only on NumPy, so they can be run before any of
the project pipeline exists.

Conventions used throughout (and inherited by the production code):

* ``wd`` in the raw data is the direction the wind blows **from** (standard
  meteorological convention). This is verified empirically in
  ``wind_transport.py`` rather than assumed.
* Transport bearing (the direction air actually moves) is ``wd + 180 deg``.
* Bearings are compass degrees, clockwise from north.
* Adjacency is indexed ``[target, source]`` so that aggregation for a target
  sums messages from its sources.
"""

from __future__ import annotations

import csv
import glob
import math
import os

import numpy as np

RAW_DIR = os.path.join("data", "raw", "PRSA_Data_20130301-20170228")

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEOROLOGY = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC_COLUMNS = POLLUTANTS + METEOROLOGY

# 16-point compass -> degrees clockwise from north
COMPASS = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

# Station coordinates. Independently validated: correlation between PM2.5
# similarity and inter-station distance is r = -0.921 across the 66 pairs,
# which would not hold if these were wrong.
STATION_COORDS = {
    "Aotizhongxin":  (39.982, 116.397),
    "Changping":     (40.217, 116.230),
    "Dingling":      (40.292, 116.220),
    "Dongsi":        (39.929, 116.417),
    "Guanyuan":      (39.929, 116.339),
    "Gucheng":       (39.914, 116.184),
    "Huairou":       (40.328, 116.628),
    "Nongzhanguan":  (39.937, 116.461),
    "Shunyi":        (40.127, 116.655),
    "Tiantan":       (39.886, 116.407),
    "Wanliu":        (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}

# Chronological split by target time, in hours from 2013-03-01 00:00.
# train 2013-03-01 -> 2015-02-28 (730 days) | val -> 2016-02-29 (366) | test -> 2017-02-28 (365)
# These match the date-derived boundaries the production builder computes, so the
# probes here and the trained models are evaluated on exactly the same test year.
TRAIN_END = 730 * 24   # 17520
VAL_END = TRAIN_END + 366 * 24   # 26304


class Dataset:
    """All 12 stations loaded into dense ``[n_stations, n_hours]`` arrays."""

    def __init__(self, stations, values, wind_dir_deg, timestamps):
        self.stations = stations
        self.values = values                # dict: column name -> [S, T]
        self.wind_dir_deg = wind_dir_deg    # [S, T], NaN where 'NA'
        self.timestamps = timestamps        # list[datetime], length T
        self.n_stations = len(stations)
        self.n_hours = len(timestamps)

    def __getitem__(self, column):
        return self.values[column]

    @property
    def pm25(self):
        return self.values["PM2.5"]

    @property
    def wind_speed(self):
        return self.values["WSPM"]

    def wind_uv(self):
        """Wind vector components in the direction the air is *moving*.

        Returns ``(u, v)`` where u is eastward and v is northward, in m/s.
        A wind recorded as coming from the north (wd=0) moves southward, so
        v is negative -- hence the leading minus signs.
        """
        rad = np.radians(self.wind_dir_deg)
        speed = self.values["WSPM"]
        return -speed * np.sin(rad), -speed * np.cos(rad)

    def split_masks(self):
        idx = np.arange(self.n_hours)
        return idx < TRAIN_END, (idx >= TRAIN_END) & (idx < VAL_END), idx >= VAL_END


def load(raw_dir: str = RAW_DIR) -> Dataset:
    """Read the 12 station CSVs into dense arrays, preserving the hourly grid."""
    import datetime

    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if len(paths) != 12:
        raise FileNotFoundError(f"expected 12 station CSVs in {raw_dir}, found {len(paths)}")

    stations, per_station, wind_rows, timestamps = [], [], [], None
    for path in paths:
        rows = list(csv.DictReader(open(path, newline="")))
        stations.append(rows[0]["station"])

        columns = {}
        for name in NUMERIC_COLUMNS:
            arr = np.full(len(rows), np.nan)
            for i, row in enumerate(rows):
                text = row[name]
                if text not in ("NA", ""):
                    arr[i] = float(text)
            columns[name] = arr
        per_station.append(columns)

        wd = np.full(len(rows), np.nan)
        for i, row in enumerate(rows):
            wd[i] = COMPASS.get(row["wd"], np.nan)
        wind_rows.append(wd)

        if timestamps is None:
            timestamps = [
                datetime.datetime(int(r["year"]), int(r["month"]), int(r["day"]), int(r["hour"]))
                for r in rows
            ]

    values = {name: np.stack([c[name] for c in per_station]) for name in NUMERIC_COLUMNS}
    return Dataset(stations, values, np.stack(wind_rows), timestamps)


def haversine_km(a: str, b: str) -> float:
    lat1, lon1 = map(math.radians, STATION_COORDS[a])
    lat2, lon2 = map(math.radians, STATION_COORDS[b])
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def bearing_deg(a: str, b: str) -> float:
    """Compass bearing FROM station a TO station b, degrees clockwise from north."""
    lat1, lon1 = map(math.radians, STATION_COORDS[a])
    lat2, lon2 = map(math.radians, STATION_COORDS[b])
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def geometry(stations):
    """Return ``(distance_km, bearing_deg)`` matrices, both indexed ``[i, j]`` as i->j."""
    dist = np.array([[haversine_km(a, b) for b in stations] for a in stations])
    bear = np.array([[bearing_deg(a, b) for b in stations] for a in stations])
    return dist, bear


def upper_triangle(matrix):
    """Off-diagonal upper-triangle entries of a symmetric matrix, as a flat array."""
    iu = np.triu_indices(matrix.shape[0], 1)
    return matrix[iu]


def nan_corr(x, y):
    """Pearson correlation over the positions where both inputs are finite."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def ridge_fit_predict(X, y, time_index, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0)):
    """Standardise on train, pick alpha on validation MAE, return test predictions.

    Returns ``(predictions, truth, time_index)`` restricted to the test split.
    """
    train = time_index < TRAIN_END
    val = (time_index >= TRAIN_END) & (time_index < VAL_END)
    test = time_index >= VAL_END

    mu, sd = X[train].mean(0), X[train].std(0) + 1e-8
    Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], axis=1)

    gram = Xs[train].T @ Xs[train]
    rhs = Xs[train].T @ y[train]
    best = None
    for alpha in alphas:
        w = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
        mae = np.abs(Xs[val] @ w - y[val]).mean()
        if best is None or mae < best[0]:
            best = (mae, w)

    return Xs[test] @ best[1], y[test], time_index[test]
