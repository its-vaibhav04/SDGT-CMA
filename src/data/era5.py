"""ERA5 boundary layer height, sampled at station coordinates.

Boundary layer height is the strongest physical driver of winter PM2.5
accumulation -- it is the mechanism the synopsis introduction names ("winter
temperature inversions trapping particulate matter") and which the original
feature set did not model at all. A shallow boundary layer traps emissions in a
thin volume; a deep one dilutes them.

**This is a soft dependency by design.** The Copernicus Climate Data Store
queues requests and can be slow or unavailable, and the project must never be
blocked on an external service. Every function here returns ``None`` rather than
raising when data cannot be obtained, and the city builders drop the feature and
record its absence in the manifest.

Three ways to supply the data, in the order tried:

1. A cached array at ``data/raw/era5/<city>_blh.npy`` of shape ``[T, N]``.
2. An automated fetch through ``cdsapi`` (requires ``pip install cdsapi`` and a
   ``~/.cdsapirc`` token from https://cds.climate.copernicus.eu).
3. Nothing -- the pipeline continues without the feature.

To supply it by hand, write an ``[T, N]`` float32 array in hours-since-start
order matching ``timestamps.npy``, with columns in ``stations.csv`` order, then
run the builder again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

CACHE_ROOT = Path("data/raw/era5")
DATASET = "reanalysis-era5-single-levels-timeseries"
VARIABLE = "boundary_layer_height"


def cache_paths(city: str, root: Path | str = CACHE_ROOT) -> tuple[Path, Path]:
    base = Path(root)
    return base / f"{city}_blh.npy", base / f"{city}_blh_meta.json"


def load_cached(
    city: str,
    *,
    n_hours: int,
    n_stations: int,
    root: Path | str = CACHE_ROOT,
) -> np.ndarray | None:
    """Return the cached ``[T, N]`` array, or ``None`` if absent or the wrong shape."""
    array_path, _ = cache_paths(city, root)
    if not array_path.exists():
        return None

    values = np.load(array_path)
    if values.shape != (n_hours, n_stations):
        print(
            f"  [era5] cache at {array_path} has shape {values.shape}, expected "
            f"{(n_hours, n_stations)} -- ignoring it"
        )
        return None
    if not np.isfinite(values).all():
        print(f"  [era5] cache at {array_path} contains non-finite values -- ignoring it")
        return None
    return values.astype(np.float32)


def fetch(
    city: str,
    *,
    coords: np.ndarray,
    stations: Sequence[str],
    start_date: str,
    end_date: str,
    root: Path | str = CACHE_ROOT,
) -> np.ndarray | None:
    """Fetch boundary layer height for each station, caching the result.

    Args:
        coords: ``[N, 2]`` array of (lat, lon) in station order.
        start_date, end_date: ``YYYY-MM-DD`` bounds, inclusive.

    Returns:
        ``[T, N]`` float32 array, or ``None`` if the fetch could not be made.
    """
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        print(
            "  [era5] cdsapi is not installed -- skipping boundary layer height.\n"
            "         To enable it: pip install cdsapi, then create ~/.cdsapirc with a\n"
            "         token from https://cds.climate.copernicus.eu/how-to-api"
        )
        return None

    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)

    try:
        columns = [
            _fetch_point(
                city=city,
                station=station,
                lat=float(coords[index, 0]),
                lon=float(coords[index, 1]),
                start_date=start_date,
                end_date=end_date,
                root=base,
            )
            for index, station in enumerate(stations)
        ]
    except Exception as error:  # noqa: BLE001 - a soft dependency must never propagate
        print(f"  [era5] fetch failed ({type(error).__name__}: {error}) -- continuing without it")
        return None

    if any(column is None for column in columns):
        return None

    values = np.stack(columns, axis=1).astype(np.float32)
    array_path, meta_path = cache_paths(city, base)
    np.save(array_path, values)
    meta_path.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "variable": VARIABLE,
                "start_date": start_date,
                "end_date": end_date,
                "stations": list(stations),
                "shape": list(values.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  [era5] cached {values.shape} to {array_path}")
    return values


def _fetch_point(
    *,
    city: str,
    station: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    root: Path,
) -> np.ndarray | None:
    """One station's time series, cached per station so a partial run resumes."""
    import cdsapi

    target = root / f"{city}_{station}_blh.nc"
    if not target.exists():
        client = cdsapi.Client(quiet=True)
        client.retrieve(
            DATASET,
            {
                "variable": [VARIABLE],
                "location": {"latitude": lat, "longitude": lon},
                "date": [f"{start_date}/{end_date}"],
                "data_format": "netcdf",
            },
            str(target),
        )

    try:
        import xarray as xr
    except ImportError:
        print("  [era5] xarray is required to read the downloaded NetCDF -- skipping")
        return None

    with xr.open_dataset(target) as handle:
        name = VARIABLE if VARIABLE in handle else list(handle.data_vars)[0]
        return np.asarray(handle[name].values, dtype=np.float64).squeeze()


def resolve(
    city: str,
    *,
    coords: np.ndarray,
    stations: Sequence[str],
    n_hours: int,
    start_date: str,
    end_date: str,
    enabled: bool = True,
    root: Path | str = CACHE_ROOT,
) -> np.ndarray | None:
    """Cache first, then fetch, then give up quietly. The builder's entry point."""
    if not enabled:
        return None

    cached = load_cached(city, n_hours=n_hours, n_stations=len(stations), root=root)
    if cached is not None:
        print(f"  [era5] using cached boundary layer height {cached.shape}")
        return cached

    fetched = fetch(
        city,
        coords=coords,
        stations=stations,
        start_date=start_date,
        end_date=end_date,
        root=root,
    )
    if fetched is not None and fetched.shape == (n_hours, len(stations)):
        return fetched
    if fetched is not None:
        print(f"  [era5] fetched shape {fetched.shape} != expected {(n_hours, len(stations))}")
    return None
