"""The data contract: what a processed city dataset contains, and how to check it.

Beijing and Delhi both build to this same set of artifacts, so every downstream
module -- windowing, graphs, models, metrics, figures -- is city-agnostic and
never needs to know which city it is looking at.

The contract is deliberately explicit about three things that are easy to get
wrong and impossible to notice afterwards:

* ``target_raw`` is never scaled. Every reported metric is computed from it.
* ``target_mask`` marks genuinely observed PM2.5. Imputed values are never
  scored as truth.
* ``wind_uv`` is in raw m/s and points in the direction the air is *moving*.
  The graph module needs physical units, not standardised ones.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.transforms import Scaler, TargetScaler
from src.utils.manifest import read_json

PROCESSED_ROOT = Path("data/processed")

ARTIFACTS = (
    "features.npy",
    "target_raw.npy",
    "target_scaled.npy",
    "target_mask.npy",
    "wind_uv.npy",
    "timestamps.npy",
    "stations.csv",
    "feature_names.json",
    "scaler.json",
    "split.json",
    "manifest.json",
)


@dataclass(frozen=True)
class Split:
    """A contiguous block of *target* time, as half-open hour indices."""

    name: str
    start: int
    end: int
    start_date: str
    end_date: str

    def __len__(self) -> int:
        return self.end - self.start

    def contains(self, index: int) -> bool:
        return self.start <= index < self.end

    def as_slice(self) -> slice:
        return slice(self.start, self.end)


@dataclass
class ProcessedDataset:
    """A validated city dataset, fully resident in memory (~45 MB for Beijing)."""

    city: str
    features: np.ndarray        # [T, N, F] float32, scaled and imputed
    target_raw: np.ndarray      # [T, N]    float32, ug/m3, NaN where unobserved
    target_scaled: np.ndarray   # [T, N]    float32, standardised, 0 where masked
    target_mask: np.ndarray     # [T, N]    bool
    wind_uv: np.ndarray         # [T, N, 2] float32, raw m/s, direction of motion
    timestamps: np.ndarray      # [T]       datetime64[h]
    stations: list[str]
    coords: np.ndarray          # [N, 2]    (lat, lon) in station order
    feature_names: list[str]
    scaler: Scaler
    target_scaler: TargetScaler
    splits: dict[str, Split]
    manifest: dict[str, Any]

    @property
    def n_hours(self) -> int:
        return self.features.shape[0]

    @property
    def n_stations(self) -> int:
        return self.features.shape[1]

    @property
    def n_features(self) -> int:
        return self.features.shape[2]

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)

    def describe(self) -> str:
        lines = [
            f"city={self.city}  hours={self.n_hours}  stations={self.n_stations}  "
            f"features={self.n_features}",
            f"range {self.timestamps[0]} -> {self.timestamps[-1]}",
            f"target observed: {self.target_mask.mean() * 100:.2f}% of station-hours",
        ]
        for split in self.splits.values():
            lines.append(
                f"  {split.name:5s} {split.start_date} -> {split.end_date}  "
                f"({len(split)} hours, indices [{split.start}, {split.end}))"
            )
        return "\n".join(lines)


def city_dir(city: str, root: Path | str = PROCESSED_ROOT) -> Path:
    return Path(root) / city


def load(city: str, root: Path | str = PROCESSED_ROOT) -> ProcessedDataset:
    """Load a processed city dataset, checking that every artifact is present."""
    base = city_dir(city, root)
    missing = [name for name in ARTIFACTS if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{base} is missing {missing}. Build it first, e.g. "
            f"`python -m src.data.build_{city}`."
        )

    with open(base / "stations.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stations = [row["station"] for row in rows]
    coords = np.array([[float(row["lat"]), float(row["lon"])] for row in rows], dtype=np.float64)

    split_payload = read_json(base / "split.json")
    splits = {
        name: Split(
            name=name,
            start=int(spec["start_index"]),
            end=int(spec["end_index"]),
            start_date=spec["start_date"],
            end_date=spec["end_date"],
        )
        for name, spec in split_payload.items()
    }

    scaler_payload = read_json(base / "scaler.json")

    return ProcessedDataset(
        city=city,
        features=np.load(base / "features.npy"),
        target_raw=np.load(base / "target_raw.npy"),
        target_scaled=np.load(base / "target_scaled.npy"),
        target_mask=np.load(base / "target_mask.npy"),
        wind_uv=np.load(base / "wind_uv.npy"),
        timestamps=np.load(base / "timestamps.npy"),
        stations=stations,
        coords=coords,
        feature_names=read_json(base / "feature_names.json")["features"],
        scaler=Scaler.from_dict(scaler_payload["features"]),
        target_scaler=TargetScaler.from_dict(scaler_payload["target"]),
        splits=splits,
        manifest=read_json(base / "manifest.json"),
    )


def validate(city: str, root: Path | str = PROCESSED_ROOT) -> list[str]:
    """Check a built dataset without rewriting it. Returns a list of problems.

    Separate from the builder on purpose: this can be run against an artifact
    directory that was produced weeks earlier, on another machine, to confirm it
    still says what it said then.
    """
    problems: list[str] = []
    dataset = load(city, root)

    t, n, f = dataset.features.shape

    # ---------------------------------------------------------------- shapes
    expected = {
        "target_raw": (t, n),
        "target_scaled": (t, n),
        "target_mask": (t, n),
        "wind_uv": (t, n, 2),
        "timestamps": (t,),
    }
    for name, shape in expected.items():
        actual = getattr(dataset, name).shape
        if actual != shape:
            problems.append(f"{name} has shape {actual}, expected {shape}")

    if len(dataset.stations) != n:
        problems.append(f"stations.csv has {len(dataset.stations)} rows but features has {n}")
    if len(dataset.feature_names) != f:
        problems.append(f"feature_names has {len(dataset.feature_names)} entries, features has {f}")

    # ------------------------------------------------------------ time grid
    deltas = np.diff(dataset.timestamps.astype("datetime64[h]").astype(np.int64))
    if deltas.size and not np.all(deltas == 1):
        problems.append("timestamps are not a contiguous hourly grid")
    if len(np.unique(dataset.timestamps)) != t:
        problems.append("timestamps contain duplicates")

    # ------------------------------------------------------------ value sanity
    if not np.isfinite(dataset.features).all():
        problems.append("features contain NaN or Inf after imputation")
    if not np.isfinite(dataset.wind_uv).all():
        problems.append("wind_uv contains NaN or Inf")
    if np.isnan(dataset.target_raw[dataset.target_mask]).any():
        problems.append("target_raw is NaN somewhere target_mask says it was observed")
    if not np.isfinite(dataset.target_scaled).all():
        problems.append("target_scaled contains NaN or Inf")

    # ------------------------------------------------------------ splits
    ordered = [dataset.splits[name] for name in ("train", "val", "test") if name in dataset.splits]
    if len(ordered) != 3:
        problems.append("split.json must define train, val and test")
    else:
        for earlier, later in zip(ordered, ordered[1:]):
            if earlier.end != later.start:
                problems.append(
                    f"splits {earlier.name} and {later.name} are not contiguous "
                    f"({earlier.end} != {later.start})"
                )
        if ordered[-1].end != t:
            problems.append(f"test split ends at {ordered[-1].end}, expected {t}")

    # --------------------------------------------- target scaling round-trip
    observed = dataset.target_mask
    if observed.any():
        recovered = dataset.target_scaler.inverse(dataset.target_scaled[observed])
        if not np.allclose(recovered, dataset.target_raw[observed], atol=1e-3):
            problems.append("target_scaled does not invert back to target_raw")

    return problems
