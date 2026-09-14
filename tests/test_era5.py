"""Gates on the ERA5 fetch, minus the network.

Two real failures from the live Copernicus API motivated these, and both were
silent rather than loud:

- **The download is a ZIP**, despite the request asking for ``data_format:
  netcdf``. Saving it as ``<station>_blh.nc`` produces a file that is not a
  NetCDF, and the error only surfaces later, in xarray, as something unrelated.
- **The response's time axis is not guaranteed** to be exactly the requested
  hours in order. Squeezing the values into the feature array without checking
  would produce a boundary layer height column silently offset in time --
  which no test downstream could catch, because a shifted physical feature is
  still a perfectly well-formed array.

Both are cheap to get right and expensive to discover later, so they are pinned
here. Nothing in this file touches the network.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from src.data import era5


def _hours(start: str, n: int) -> np.ndarray:
    return np.arange(
        np.datetime64(start, "h"), np.datetime64(start, "h") + np.timedelta64(n, "h"),
        np.timedelta64(1, "h"),
    )


# ------------------------------------------------------------------- unwrapping
def test_extract_unwraps_a_zipped_netcdf(tmp_path):
    """The live API wraps the NetCDF in an archive; this is the real shape."""
    archive = tmp_path / "beijing_X_blh.download"
    target = tmp_path / "beijing_X_blh.nc"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("reanalysis-era5-single-levels-timeseries-abc.nc", b"NETCDF-BYTES")

    era5._extract(archive, target)

    assert target.read_bytes() == b"NETCDF-BYTES"
    assert not archive.exists(), "the archive should be cleaned up after extraction"


def test_extract_passes_through_a_bare_netcdf(tmp_path):
    """If the CDS ever stops zipping, the code must not break."""
    archive = tmp_path / "beijing_X_blh.download"
    target = tmp_path / "beijing_X_blh.nc"
    archive.write_bytes(b"CDF\x01raw-netcdf")

    era5._extract(archive, target)

    assert target.read_bytes() == b"CDF\x01raw-netcdf"
    assert not archive.exists()


@pytest.mark.parametrize("members", [[], ["a.nc", "b.nc"]])
def test_extract_refuses_an_ambiguous_archive(tmp_path, members):
    """Guessing which member is the data would be worse than failing."""
    archive = tmp_path / "beijing_X_blh.download"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in members:
            bundle.writestr(name, b"x")
        if not members:
            bundle.writestr("readme.txt", b"x")

    with pytest.raises(ValueError, match="exactly one"):
        era5._extract(archive, tmp_path / "out.nc")


# -------------------------------------------------------------------- alignment
def test_align_places_values_on_the_expected_grid():
    expected = _hours("2013-03-01T00", 6)
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    aligned = era5._align(expected, values, expected, "X")

    np.testing.assert_array_equal(aligned, values)


def test_align_sorts_an_out_of_order_response():
    """Order is not guaranteed, and a reversed response must not reverse the feature."""
    expected = _hours("2013-03-01T00", 4)
    shuffled = expected[::-1]
    values = np.array([40.0, 30.0, 20.0, 10.0])       # paired with shuffled times

    aligned = era5._align(shuffled, values, expected, "X")

    np.testing.assert_array_equal(aligned, [10.0, 20.0, 30.0, 40.0])


def test_align_selects_the_requested_subset_of_a_longer_response():
    """A response covering more than was asked for must still land correctly."""
    supplied = _hours("2013-02-28T00", 48)
    values = np.arange(48, dtype=float)
    expected = _hours("2013-03-01T00", 5)

    aligned = era5._align(supplied, values, expected, "X")

    # 2013-03-01T00 is the 25th supplied hour, i.e. index 24.
    np.testing.assert_array_equal(aligned, [24.0, 25.0, 26.0, 27.0, 28.0])


def test_align_refuses_a_response_with_missing_hours(capsys):
    """Silently filling a gap would put a wrong physical value in the feature."""
    supplied = np.concatenate([_hours("2013-03-01T00", 2), _hours("2013-03-01T04", 2)])
    values = np.array([1.0, 2.0, 5.0, 6.0])
    expected = _hours("2013-03-01T00", 6)

    assert era5._align(supplied, values, expected, "Wanliu") is None
    assert "missing" in capsys.readouterr().out


def test_align_refuses_duplicate_timestamps(capsys):
    expected = _hours("2013-03-01T00", 3)
    supplied = np.array(
        [expected[0], expected[0], expected[1]], dtype="datetime64[h]"
    )

    assert era5._align(supplied, np.array([1.0, 2.0, 3.0]), expected, "Wanliu") is None
    assert "duplicate" in capsys.readouterr().out


def test_align_refuses_a_response_that_misses_the_range_entirely():
    """The clip in searchsorted must not let an off-by-a-year response through."""
    supplied = _hours("2014-03-01T00", 6)
    expected = _hours("2013-03-01T00", 6)

    assert era5._align(supplied, np.arange(6.0), expected, "Wanliu") is None


# ------------------------------------------------------------------------ cache
def test_cached_array_of_the_wrong_shape_is_ignored(tmp_path, capsys):
    """A stale cache from a different station set must not be silently accepted."""
    np.save(tmp_path / "beijing_blh.npy", np.zeros((10, 3), dtype=np.float32))

    result = era5.load_cached("beijing", n_hours=35064, n_stations=12, root=tmp_path)

    assert result is None
    assert "expected" in capsys.readouterr().out


def test_cached_array_with_non_finite_values_is_ignored(tmp_path):
    values = np.zeros((10, 2), dtype=np.float32)
    values[3, 1] = np.nan
    np.save(tmp_path / "beijing_blh.npy", values)

    assert era5.load_cached("beijing", n_hours=10, n_stations=2, root=tmp_path) is None


def test_resolve_is_a_no_op_when_disabled(tmp_path):
    """The soft dependency must be switchable off without touching the network."""
    assert era5.resolve(
        "beijing", coords=np.zeros((2, 2)), stations=["A", "B"], n_hours=10,
        start_date="2013-03-01", end_date="2013-03-01", enabled=False, root=tmp_path,
    ) is None


def test_resolve_prefers_a_valid_cache_over_fetching(tmp_path, capsys):
    """If this regressed, every build would re-download four years of data."""
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    np.save(tmp_path / "beijing_blh.npy", values)

    result = era5.resolve(
        "beijing", coords=np.zeros((2, 2)), stations=["A", "B"], n_hours=10,
        start_date="2013-03-01", end_date="2013-03-01", root=tmp_path,
    )

    np.testing.assert_array_equal(result, values)
    assert "using cached" in capsys.readouterr().out


# ------------------------------------------------------------------- time zone
def _fake_point(request_start: str, request_end: str):
    """A response whose value at each UTC hour encodes that hour, for tracing."""
    times = np.arange(
        np.datetime64(request_start, "h"),
        np.datetime64(request_end, "h") + np.timedelta64(24, "h"),
        np.timedelta64(1, "h"),
    )
    epoch = times.astype("datetime64[h]").astype(np.int64).astype(float)
    return times, epoch


def test_fetch_shifts_era5_utc_onto_the_datasets_local_clock(tmp_path, monkeypatch):
    """ERA5 is UTC; Beijing PRSA timestamps are UTC+8.

    This was a real bug and a completely silent one. Unshifted, mean boundary
    layer height peaked at "06:00" and bottomed at "18:00" -- the diurnal cycle
    upside down -- while every array stayed the right shape and every value
    stayed physically plausible. Nothing downstream could catch a feature that is
    merely offset in time.

    The trace here is exact: each returned value encodes the UTC hour it came
    from, so local hour t must carry the encoding of UTC hour t-8.
    """
    pytest.importorskip("cdsapi")
    captured = {}

    def stub(*, city, station, lat, lon, start_date, end_date, root):
        captured["window"] = (start_date, end_date)
        return _fake_point(start_date, end_date)

    monkeypatch.setattr(era5, "_fetch_point", stub)

    values = era5.fetch(
        "beijing", coords=np.zeros((1, 2)), stations=["X"],
        start_date="2013-03-01", end_date="2013-03-01",
        utc_offset_hours=8, root=tmp_path,
    )

    assert values is not None and values.shape == (24, 1)

    local = np.arange(
        np.datetime64("2013-03-01T00", "h"), np.datetime64("2013-03-02T00", "h"),
        np.timedelta64(1, "h"),
    )
    expected_utc_hour = (local - np.timedelta64(8, "h")).astype(np.int64).astype(float)
    np.testing.assert_array_equal(values[:, 0], expected_utc_hour)

    # The request must reach back a day so the shifted window is real data, not
    # an edge-filled approximation.
    assert captured["window"][0] == "2013-02-28", captured["window"]


def test_fetch_with_zero_offset_is_unshifted(tmp_path, monkeypatch):
    pytest.importorskip("cdsapi")
    monkeypatch.setattr(
        era5, "_fetch_point",
        lambda **kw: _fake_point(kw["start_date"], kw["end_date"]),
    )
    values = era5.fetch(
        "beijing", coords=np.zeros((1, 2)), stations=["X"],
        start_date="2013-03-01", end_date="2013-03-01",
        utc_offset_hours=0, root=tmp_path,
    )
    local = np.arange(
        np.datetime64("2013-03-01T00", "h"), np.datetime64("2013-03-02T00", "h"),
        np.timedelta64(1, "h"),
    ).astype(np.int64).astype(float)
    np.testing.assert_array_equal(values[:, 0], local)


def test_fetch_refuses_a_sub_hourly_offset(tmp_path, capsys):
    """Delhi is UTC+5:30. Refuse rather than silently round to +5 or +6."""
    pytest.importorskip("cdsapi")
    result = era5.fetch(
        "delhi", coords=np.zeros((1, 2)), stations=["X"],
        start_date="2018-01-01", end_date="2018-01-01",
        utc_offset_hours=5.5, root=tmp_path,
    )
    assert result is None
    assert "not a whole number" in capsys.readouterr().out


def test_metadata_records_the_offset_and_request_window(tmp_path, monkeypatch):
    """Provenance: a cached array must say what clock it is on."""
    pytest.importorskip("cdsapi")
    monkeypatch.setattr(
        era5, "_fetch_point",
        lambda **kw: _fake_point(kw["start_date"], kw["end_date"]),
    )
    era5.fetch(
        "beijing", coords=np.zeros((1, 2)), stations=["X"],
        start_date="2013-03-01", end_date="2013-03-01",
        utc_offset_hours=8, root=tmp_path,
    )
    import json as _json
    meta = _json.loads((tmp_path / "beijing_blh_meta.json").read_text(encoding="utf-8"))
    assert meta["utc_offset_hours"] == 8
    assert meta["request_window_utc"] == ["2013-02-28", "2013-03-01"]
