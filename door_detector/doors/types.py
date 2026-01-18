"""Shared door-type taxonomy for Door Detector (detection, UI, training).

Keep this centralized so adding a new door type is a single edit + plumbing,
not a string-spelunking exercise.
"""

from __future__ import annotations

from typing import Literal

# Canonical door types this project aims to support.
DoorType = Literal["swing", "double", "pocket", "bifold"]

DOOR_TYPES: tuple[str, ...] = ("swing", "double", "pocket", "bifold")
DOOR_TYPES_SET: set[str] = set(DOOR_TYPES)


def normalize_door_type(v: object, *, default: str = "swing") -> str:
    """Return a canonical door type string (or default)."""
    try:
        s = str(v).strip().lower()
    except Exception:
        return default
    return s if s in DOOR_TYPES_SET else default

