"""Provenance capture: file checksums and environment snapshots.

Two jobs. ``file_digest`` and ``build_data_manifest`` record what went into a
processed dataset, so a result can always be traced back to the exact bytes it
came from. ``capture_environment`` records what a training run executed on,
because "we cannot reproduce our own numbers" is a failure mode that shows up
only at the end, when it is expensive.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file, streamed so large raw files do not load into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    """Current commit hash, with a dirty marker, or ``"unknown"`` outside a repo."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return f"{rev}-dirty" if dirty else rev
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "torch", "sklearn", "scipy", "lightgbm", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return versions


def capture_environment() -> dict[str, Any]:
    """Everything needed to reconstruct the runtime that produced a result."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": _package_versions(),
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        }
    except ImportError:
        info["torch"] = None
    return info


def build_data_manifest(
    *,
    city: str,
    source_files: list[str | Path],
    stations: list[str],
    n_hours: int,
    date_range: tuple[str, str],
    feature_names: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe a processed dataset completely enough to detect any silent change."""
    manifest: dict[str, Any] = {
        "city": city,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "n_stations": len(stations),
        "station_order": stations,
        "n_hours": n_hours,
        "date_range": {"start": date_range[0], "end": date_range[1]},
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "sources": [
            {"path": str(p), "sha256": file_digest(p), "bytes": Path(p).stat().st_size}
            for p in sorted(source_files, key=str)
        ],
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
