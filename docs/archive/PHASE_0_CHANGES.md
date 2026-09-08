# Phase 0 Changes

## Objective

Set up the SDGT-CMA repository environment, skeleton, reproducibility utility, starter configs, and verification tests according to `SDGT-CMA_Implementation_Roadmap.md`.

## Changes Made

- Created the roadmap-prescribed project structure under `data/`, `src/`, `configs/`, `notebooks/`, `experiments/`, and `tests/`.
- Added importable Python packages for future data, graph, spatial, temporal, fusion, and full-model modules.
- Added placeholder modules for `gcn.py`, `adaptive_gat.py`, `patchtst.py`, `cma.py`, and `full_model.py` without implementing later-phase architecture early.
- Added `src/utils/reproducibility.py` with `set_seed(42)` support for Python `random`, NumPy, and PyTorch.
- Added starter model configs for the three required comparison models:
  - Model 1: GCN + concat fusion + PatchTST
  - Model 2: GCN + CMA fusion + PatchTST
  - Model 3: adaptive wind-aware GAT + CMA fusion + PatchTST
- Added `requirements.txt`, `pyproject.toml`, `.gitignore`, and a minimal `README.md`.
- Added Phase 0 tests for importability and deterministic seeding.

## Verification

Completed from the repository root:

```powershell
.\.venv\Scripts\python -m pytest tests
.\.venv\Scripts\python -c "import torch; print(torch.__version__)"
```

Results:

- `pytest tests`: 3 passed
- `torch.__version__`: `2.13.0+cpu`
- `import src.models.spatial.gcn`: passed

## Notes

PyTorch Geometric is intentionally not included in Phase 0 because the roadmap marks it as optional, and the proposed wind-aware GAT is expected to be implemented transparently in later phases.
