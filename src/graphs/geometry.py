"""Station geometry: distances and bearings.

**Bearing convention, stated once and used everywhere.** ``bearing_matrix[i, j]``
is the compass bearing *from station i to station j*, in degrees clockwise from
north. Combined with the adjacency convention ``[target, source]``, the
alignment test for "does source j's air move toward target i" is

    cos(transport_bearing_at_j - bearing_matrix[j, i])

Note the index order: ``[j, i]``, source first. Getting this backwards produces a
model that learns anti-physics without raising a single error, which is why
``tests/test_graph_physics.py`` checks it against a synthetic westerly.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """Great-circle distances in km, ``[N, N]``, symmetric with a zero diagonal.

    Args:
        coords: ``[N, 2]`` array of (latitude, longitude) in degrees.
    """
    lat = np.radians(coords[:, 0])[:, None]
    lon = np.radians(coords[:, 1])[:, None]

    dlat = lat - lat.T
    dlon = lon - lon.T
    inner = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def bearing_matrix(coords: np.ndarray) -> np.ndarray:
    """Initial great-circle bearings ``i -> j`` in degrees clockwise from north."""
    lat = np.radians(coords[:, 0])[:, None]
    lon = np.radians(coords[:, 1])[:, None]

    dlon = lon.T - lon
    y = np.sin(dlon) * np.cos(lat.T)
    x = np.cos(lat) * np.sin(lat.T) - np.sin(lat) * np.cos(lat.T) * np.cos(dlon)
    bearings = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    np.fill_diagonal(bearings, 0.0)     # undefined for a self-pair; never used
    return bearings


def transport_bearing(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compass bearing of the direction air is *moving*, from wind components.

    ``u`` is eastward and ``v`` northward, so ``atan2(u, v)`` (east over north,
    not the usual north over east) gives degrees clockwise from north.
    """
    return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0


def alignment(
    transport_deg: np.ndarray,
    bearings: np.ndarray,
) -> np.ndarray:
    """Cosine alignment between transport direction and source-to-target bearing.

    Args:
        transport_deg: ``[..., N]`` transport bearing at each *source*.
        bearings: ``[N, N]`` bearing matrix, indexed ``[source, target]``.

    Returns:
        ``[..., N, N]`` indexed ``[target, source]`` -- transposed on the way out
        so it lines up with the adjacency convention. ``+1`` means the source's
        air moves straight at the target.
    """
    delta = transport_deg[..., :, None] - bearings          # [..., source, target]
    aligned = np.cos(np.radians(delta))
    return np.swapaxes(aligned, -1, -2)                     # -> [..., target, source]
