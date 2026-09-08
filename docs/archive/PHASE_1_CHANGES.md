# Phase 1 Changes

## Objective

Load the Beijing PRSA multi-site air-quality dataset and station coordinates into reusable Phase 1 artifacts, matching the roadmap's required long-format schema.

## Changes Made

- Added `src/data/phase1_build_dataset.py`.
- Implemented loading for all 12 raw station CSVs in `data/raw/PRSA_Data_20130301-20170228`.
- Implemented station coordinate loading from `data/raw/stations.csv`.
- Normalized the coordinate table to `[station, lat, lon]`.
- Combined all raw station files into one long-format DataFrame with columns:
  `station, datetime, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, wd`.
- Added validation that the 12 stations in the raw data exactly match the 12 stations in `stations.csv`.
- Added Phase 1 tests for station coverage, schema, chronological ordering, and expected hourly coverage.

## Outputs

The Phase 1 build script writes:

- `data/interim/phase1/prsa_long.csv`
- `data/interim/phase1/prsa_long.parquet`
- `data/interim/phase1/stations.csv`

## Verification

Run from the repository root:

```powershell
.\.venv\Scripts\python -m src.data.phase1_build_dataset
.\.venv\Scripts\python -m pytest tests
```

