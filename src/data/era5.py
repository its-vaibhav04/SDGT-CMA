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
# What the CDS actually names the variable inside the NetCDF it returns.
SHORT_NAME = "blh"


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
    utc_offset_hours: int = 0,
    root: Path | str = CACHE_ROOT,
) -> np.ndarray | None:
    """Fetch boundary layer height for each station, caching the result.

    Args:
        coords: ``[N, 2]`` array of (lat, lon) in station order.
        start_date, end_date: ``YYYY-MM-DD`` bounds, inclusive, in the dataset's
            local time.
        utc_offset_hours: the dataset's offset from UTC (Beijing is +8). ERA5 is
            always UTC, so this is what puts the two on the same clock.

    Returns:
        ``[T, N]`` float32 array on the local-time grid, or ``None``.
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

    # The hourly grid the builder expects, in the *dataset's* local time.
    # end_date is inclusive, so the last hour is its 23:00.
    expected_local = np.arange(
        np.datetime64(f"{start_date}T00", "h"),
        np.datetime64(f"{end_date}T00", "h") + np.timedelta64(24, "h"),
        np.timedelta64(1, "h"),
    )

    # ERA5 is UTC; the station records are not. Beijing PRSA timestamps are
    # China Standard Time (UTC+8), so local hour t is UTC hour t-8, and the
    # request has to reach back a day earlier to cover the shifted window.
    #
    # This was a real bug, and a silent one: unshifted, mean boundary layer
    # height peaked at "06:00" and bottomed at "18:00", which is the diurnal
    # cycle upside down. Shifted, it peaks at 15:00 local and bottoms at 04:00 --
    # textbook -- and the winter anticorrelation with PM2.5 strengthens from
    # -0.28 to -0.46. Nothing downstream could have caught it: a feature offset
    # in time is still a perfectly well-formed column.
    if int(utc_offset_hours) != utc_offset_hours:
        print(
            f"  [era5] utc_offset_hours={utc_offset_hours} is not a whole number of "
            "hours; sub-hourly alignment is not implemented -- skipping"
        )
        return None
    expected = expected_local - np.timedelta64(int(utc_offset_hours), "h")

    # Widen the request to whole days covering the shifted window.
    request_start = str(expected[0].astype("datetime64[D]"))
    request_end = str(expected[-1].astype("datetime64[D]"))

    columns: list[np.ndarray | None] = []
    try:
        for index, station in enumerate(stations):
            print(f"  [era5] {index + 1}/{len(stations)} {station}", flush=True)
            fetched = _fetch_point(
                city=city,
                station=station,
                lat=float(coords[index, 0]),
                lon=float(coords[index, 1]),
                start_date=request_start,
                end_date=request_end,
                root=base,
            )
            if fetched is None:
                columns.append(None)
                break
            columns.append(_align(*fetched, expected, station))
            if columns[-1] is None:
                break
    except Exception as error:  # noqa: BLE001 - a soft dependency must never propagate
        print(f"  [era5] fetch failed ({type(error).__name__}: {error}) -- continuing without it")
        return None

    if len(columns) != len(stations) or any(column is None for column in columns):
        return None

    values = np.stack(columns, axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        print("  [era5] fetched array contains non-finite values -- continuing without it")
        return None
    array_path, meta_path = cache_paths(city, base)
    np.save(array_path, values)
    meta_path.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "variable": VARIABLE,
                "start_date": start_date,
                "end_date": end_date,
                "utc_offset_hours": int(utc_offset_hours),
                "request_window_utc": [request_start, request_end],
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
) -> tuple[np.ndarray, np.ndarray] | None:
    """One station's series, cached per station so a partial run resumes.

    Returns ``(valid_time, values)`` rather than a bare array. The caller aligns
    those times onto the dataset's hourly grid: the request asks for a date
    range, and trusting the response to come back with exactly the right number
    of hours in exactly the right order is the kind of assumption that produces a
    silently shifted feature rather than an error.

    **The download is a ZIP, despite ``data_format: netcdf``.** The CDS wraps the
    NetCDF in an archive and ``cdsapi`` saves it under whatever filename we
    passed, so naming the target ``.nc`` produces a file that is not a NetCDF at
    all. This unwraps it, and keeps the extracted member so a rerun is free.
    """
    import cdsapi

    archive = root / f"{city}_{station}_blh.download"
    target = root / f"{city}_{station}_blh.nc"

    if not target.exists():
        if not archive.exists():
            client = cdsapi.Client(quiet=True)
            client.retrieve(
                DATASET,
                {
                    "variable": [VARIABLE],
                    "location": {"latitude": lat, "longitude": lon},
                    "date": [f"{start_date}/{end_date}"],
                    "data_format": "netcdf",
                },
                str(archive),
            )
        _extract(archive, target)

    try:
        import xarray as xr
    except ImportError:
        print("  [era5] xarray is required to read the downloaded NetCDF -- skipping")
        return None

    with xr.open_dataset(target) as handle:
        # The CDS returns the GRIB short name, "blh", not the request's long
        # name. Prefer the long name if it is ever used, then the short one,
        # then whatever single variable is present.
        for candidate in (VARIABLE, SHORT_NAME):
            if candidate in handle:
                name = candidate
                break
        else:
            name = list(handle.data_vars)[0]

        time_name = "valid_time" if "valid_time" in handle.coords else "time"
        times = np.asarray(handle[time_name].values, dtype="datetime64[h]")
        values = np.asarray(handle[name].values, dtype=np.float64).squeeze()

    if values.shape != times.shape:
        print(
            f"  [era5] {station}: {values.shape} values against {times.shape} "
            f"timestamps -- ignoring"
        )
        return None
    return times, values


def _extract(archive: Path, target: Path) -> None:
    """Unwrap the CDS response, whether or not it is actually an archive."""
    import zipfile

    if not zipfile.is_zipfile(archive):
        archive.replace(target)
        return

    with zipfile.ZipFile(archive) as bundle:
        members = [n for n in bundle.namelist() if n.endswith(".nc")]
        if len(members) != 1:
            raise ValueError(
                f"expected exactly one .nc inside {archive.name}, found {members}"
            )
        with bundle.open(members[0]) as source:
            target.write_bytes(source.read())
    archive.unlink()


def _align(
    times: np.ndarray, values: np.ndarray, expected: np.ndarray, station: str
) -> np.ndarray | None:
    """Place a station's series onto the dataset's hourly grid.

    Strict on purpose. A physical feature that is silently offset by a few hours
    is worse than no feature at all, and this is a soft dependency -- refusing is
    cheap. Duplicate or missing hours therefore return ``None`` with a diagnostic
    rather than being patched over.
    """
    if len(np.unique(times)) != len(times):
        print(f"  [era5] {station}: duplicate timestamps in the response -- ignoring")
        return None

    order = np.argsort(times)
    times, values = times[order], values[order]

    position = np.searchsorted(times, expected)
    position = np.clip(position, 0, len(times) - 1)
    matched = times[position] == expected
    if not matched.all():
        print(
            f"  [era5] {station}: response covers {times[0]}..{times[-1]}, missing "
            f"{int((~matched).sum())} of {len(expected)} required hours -- ignoring"
        )
        return None
    return values[position]


def resolve(
    city: str,
    *,
    coords: np.ndarray,
    stations: Sequence[str],
    n_hours: int,
    start_date: str,
    end_date: str,
    utc_offset_hours: int = 0,
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
        utc_offset_hours=utc_offset_hours,
        root=root,
    )
    if fetched is not None and fetched.shape == (n_hours, len(stations)):
        return fetched
    if fetched is not None:
        print(f"  [era5] fetched shape {fetched.shape} != expected {(n_hours, len(stations))}")
    return None
