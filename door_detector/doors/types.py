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
        s_raw = str(v).strip().lower()
    except Exception:
        return default

    # Normalize common formatting variants (spaces/hyphens/underscores).
    s = s_raw.replace("-", "_").replace(" ", "_")

    # Map internal detector subtypes and legacy strings to canonical UI/training types.
    #
    # Note: the detector may emit fine-grained swing-related primitives like `swing_arc`
    # or `swing_leaf`. For labeling/training/UI, those should all be treated as `swing`.
    aliases: dict[str, str] = {
        # Swing subtypes / primitives.
        "swing_arc": "swing",
        "swing_leaf": "swing",
        "leaf_arc": "swing",
        # Common human-friendly variants.
        "bi_fold": "bifold",
        "bi_fold_door": "bifold",
        "bi_fold_doors": "bifold",
        "bifold_door": "bifold",
        "bifold_doors": "bifold",
        "double_door": "double",
        "double_doors": "double",
        "pocket_door": "pocket",
        "pocket_doors": "pocket",
    }

    if s in DOOR_TYPES_SET:
        return s
    if s in aliases:
        return aliases[s]
    return default

