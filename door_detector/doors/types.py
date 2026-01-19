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

# Door types exposed as explicit UI choices (review filter + "Label as" dropdowns).
#
# Note: The detector may still emit `pocket` and `bifold`, but the review UI treats those
# as non-separate categories (see `normalize_door_type` mapping + UI usage).
UI_DOOR_TYPES: tuple[str, ...] = ("swing", "double")
UI_DOOR_TYPES_SET: set[str] = set(UI_DOOR_TYPES)


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
        # Bifold doors are treated as double doors in the UI/training mapper.
        "bifold": "double",
        "bi_fold": "double",
        "bi_fold_door": "double",
        "bi_fold_doors": "double",
        "bifold_door": "double",
        "bifold_doors": "double",
        "double_door": "double",
        "double_doors": "double",
        "pocket_door": "pocket",
        "pocket_doors": "pocket",
    }

    if s in aliases:
        return aliases[s]
    if s in DOOR_TYPES_SET:
        return s
    return default

