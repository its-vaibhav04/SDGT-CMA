"""Invertible feature and target scaling, fitted on the training split alone.

Two rules govern everything here.

**Fit on train only.** Statistics are computed over the training slice of the
time axis and nothing else. Fitting on the full record leaks test-set
distribution into training and inflates reported accuracy in a way a reviewer
catches immediately.

**Fit on observed values only.** Statistics come from the pre-imputation array,
so climatology fill cannot drag the mean around.

Pollutant *inputs* get ``log1p`` before standardising, because PM2.5 has a 999
tail and Min-Max squeezes 99 % of the data into ``[0, 0.37]``. The PM2.5
*target* deliberately does not: it is standardised raw, so the training
objective stays aligned with the MAE and RMSE reported in ug/m3 and inversion is
exact. A log-target variant belongs in the ablation set, not the default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np

Kind = Literal["log_standard", "standard", "none"]

EPSILON = 1e-8


@dataclass
class ColumnSpec:
    """How one feature column is transformed, and the parameters to undo it."""

    name: str
    kind: Kind
    mean: float = 0.0
    std: float = 1.0

    def apply(self, values: np.ndarray) -> np.ndarray:
        if self.kind == "none":
            return values
        out = np.log1p(np.clip(values, 0.0, None)) if self.kind == "log_standard" else values
        return (out - self.mean) / self.std

    def invert(self, values: np.ndarray) -> np.ndarray:
        if self.kind == "none":
            return values
        out = values * self.std + self.mean
        return np.expm1(out) if self.kind == "log_standard" else out


class Scaler:
    """Per-column transforms over a ``[T, N, C]`` array, applied on the C axis."""

    def __init__(self, specs: Sequence[ColumnSpec]):
        self.specs = list(specs)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        names: Sequence[str],
        kinds: dict[str, Kind],
        train_slice: slice,
    ) -> "Scaler":
        """Fit on ``values[train_slice]``, ignoring NaN so imputation cannot bias it.

        Args:
            values: ``[T, N, C]`` pre-imputation array; NaN marks unobserved.
            names: Column names, length C.
            kinds: Column name -> transform kind. Missing names default to
                ``"standard"``.
            train_slice: Slice on the time axis defining the training split.
        """
        if values.shape[-1] != len(names):
            raise ValueError(f"values has {values.shape[-1]} columns but {len(names)} names given")

        specs: list[ColumnSpec] = []
        train = values[train_slice]
        for index, name in enumerate(names):
            kind: Kind = kinds.get(name, "standard")
            if kind == "none":
                specs.append(ColumnSpec(name, kind))
                continue

            column = train[..., index].astype(np.float64)
            if kind == "log_standard":
                column = np.log1p(np.clip(column, 0.0, None))

            if not np.isfinite(column).any():
                raise ValueError(f"column {name!r} has no finite values in the training split")

            mean = float(np.nanmean(column))
            std = float(np.nanstd(column))
            if std < EPSILON:
                # A constant column carries no information; centre it and leave the
                # scale at 1 rather than dividing by ~0.
                std = 1.0
            specs.append(ColumnSpec(name, kind, mean, std))
        return cls(specs)

    # ------------------------------------------------------------- applying
    def transform(self, values: np.ndarray) -> np.ndarray:
        out = np.empty_like(values, dtype=np.float32)
        for index, spec in enumerate(self.specs):
            out[..., index] = spec.apply(values[..., index].astype(np.float64))
        return out

    def inverse(self, values: np.ndarray) -> np.ndarray:
        out = np.empty_like(values, dtype=np.float64)
        for index, spec in enumerate(self.specs):
            out[..., index] = spec.invert(values[..., index].astype(np.float64))
        return out

    # ------------------------------------------------------------ persistence
    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def to_dict(self) -> dict[str, Any]:
        return {"columns": [asdict(spec) for spec in self.specs]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Scaler":
        return cls([ColumnSpec(**column) for column in payload["columns"]])


class TargetScaler:
    """Standardisation for PM2.5 targets. No log -- see the module docstring."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, target: np.ndarray, train_slice: slice) -> "TargetScaler":
        """Fit on observed training-split values only.

        Args:
            target: ``[T, N]`` PM2.5 in ug/m3, NaN where unobserved.
        """
        train = target[train_slice].astype(np.float64)
        if not np.isfinite(train).any():
            raise ValueError("target has no finite values in the training split")
        mean = float(np.nanmean(train))
        std = float(np.nanstd(train))
        return cls(mean, max(std, EPSILON))

    def transform(self, target: np.ndarray) -> np.ndarray:
        return ((target.astype(np.float64) - self.mean) / self.std).astype(np.float32)

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        """Back to ug/m3. Every reported metric passes through here."""
        return scaled.astype(np.float64) * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "standard", "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TargetScaler":
        return cls(payload["mean"], payload["std"])


def default_kinds(
    pollutants: Iterable[str],
    log_extra: Iterable[str] = (),
    passthrough: Iterable[str] = (),
) -> dict[str, Kind]:
    """Assemble the transform map for the standard feature set.

    Pollutants and a few skewed extras (RAIN, boundary layer height, the
    time-since-observed channels) get ``log1p``; binary masks and the cyclical
    calendar encodings pass through untouched, since they are already bounded.
    """
    kinds: dict[str, Kind] = {}
    for name in pollutants:
        kinds[name] = "log_standard"
    for name in log_extra:
        kinds[name] = "log_standard"
    for name in passthrough:
        kinds[name] = "none"
    return kinds
