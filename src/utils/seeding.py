"""Deterministic seeding across Python, NumPy and PyTorch.

Complete reproducibility is not achievable across PyTorch releases, platforms or
GPU architectures, so the goal here is narrower and more useful: two runs of the
same code on the same machine with the same seed must produce the same numbers,
and the environment that produced any result must be recoverable from its
manifest.

One known limitation, stated rather than papered over: ``PYTHONHASHSEED`` only
takes effect at interpreter startup. Setting it from inside a running process
does nothing. :func:`seed_everything` therefore warns if it is unset, and the
launcher in ``scripts/`` sets it before spawning the training process.
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Any

import numpy as np

try:  # torch is optional so the data pipeline can run without it
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def seed_everything(seed: int, strict: bool = False) -> None:
    """Seed every RNG the project touches.

    Args:
        seed: The seed. Recorded per run rather than hard-coded at call sites.
        strict: Also request deterministic CUDA kernels. This is slower and makes
            a handful of operations raise rather than silently use a
            nondeterministic path, which is the point -- use it when a result
            must be exactly reproducible, not for routine training.
    """
    if "PYTHONHASHSEED" not in os.environ:
        warnings.warn(
            "PYTHONHASHSEED is not set. Python's string hashing stays randomised "
            "for this process; set it in the launcher before startup if exact "
            "reproducibility is required.",
            RuntimeWarning,
            stacklevel=2,
        )

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if strict:
        # cuBLAS needs this set before the first CUDA context or matmuls raise.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def dataloader_generator(seed: int) -> Any:
    """A generator for ``DataLoader(generator=...)`` so shuffling is reproducible."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for dataloader_generator")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible RNG stream.

    Without this, forked workers inherit the parent's NumPy state and every
    worker generates identical random numbers -- a classic silent bug in any
    pipeline that augments or samples inside ``__getitem__``.
    """
    if torch is None:  # pragma: no cover
        return
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
