"""Label schema + helpers for the Streamlit review UI."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict


LABELS_SCHEMA_VERSION = 2
_LEGACY_LABEL_KEYS = {"accepted_ids", "rejected_ids", "added_boxes", "notes"}
_LABELS_V2_REQUIRED_KEYS = {
    "schema_version",
    "reviewed_at",
    "confirmed_ids",
    "deleted_ids",
    "manual_additions",
    "unmatched_manual_boxes",
}


def labels_v2_default() -> Dict[str, Any]:
    """Return an empty schema v2 labels object (reviewed_at is null until first save)."""
    return {
        "schema_version": LABELS_SCHEMA_VERSION,
        "reviewed_at": None,
        "confirmed_ids": [],
        "deleted_ids": [],
        "manual_additions": [],
        "unmatched_manual_boxes": [],
    }


def validate_labels_v2_or_raise(labels_data: Dict[str, Any], *, labels_path: Path) -> None:
    """Validate schema v2 labels.json. Raise with a clear migration message on failure."""
    if not isinstance(labels_data, dict):
        raise ValueError(f"Invalid labels.json (expected object): {labels_path}")

    schema_version = labels_data.get("schema_version", None)
    if schema_version != LABELS_SCHEMA_VERSION:
        raise ValueError(
            "\n".join(
                [
                    f"Unsupported labels.json schema in {labels_path}.",
                    f"Expected schema_version={LABELS_SCHEMA_VERSION}, got {schema_version!r}.",
                    "",
                    "This UI no longer supports legacy label schemas.",
                    "Please delete this labels.json (or migrate it offline) and re-review the file.",
                ]
            )
        )

    legacy_present = sorted(k for k in _LEGACY_LABEL_KEYS if k in labels_data)
    if legacy_present:
        raise ValueError(
            "\n".join(
                [
                    f"labels.json in {labels_path} contains deprecated fields: {legacy_present}",
                    "Please delete this labels.json (or migrate it offline) and re-review the file.",
                ]
            )
        )

    missing = sorted(k for k in _LABELS_V2_REQUIRED_KEYS if k not in labels_data)
    if missing:
        raise ValueError(f"labels.json in {labels_path} is missing required keys: {missing}")

    # Lightweight type checks (don’t coerce silently; fail fast).
    for lk in ["confirmed_ids", "deleted_ids", "manual_additions", "unmatched_manual_boxes"]:
        if not isinstance(labels_data.get(lk), list):
            raise ValueError(f"labels.json field {lk!r} must be a list: {labels_path}")


def save_labels(dir_path: Path, labels_data: Dict[str, Any]) -> None:
    labels_path = dir_path / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)


def coerce_id_set(v: Any) -> set[str]:
    """Coerce a list/set of ids into a stable set[str]."""
    out: set[str] = set()
    if v is None:
        return out
    try:
        it = v if isinstance(v, (list, tuple, set)) else list(v)
    except Exception:
        it = []
    for x in it:
        if x is None:
            continue
        s = str(x)
        if s:
            out.add(s)
    return out


def snapshot_label_state(src: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "confirmed_ids": coerce_id_set(src.get("confirmed_ids", set())),
        "deleted_ids": coerce_id_set(src.get("deleted_ids", set())),
        "manual_additions": copy.deepcopy(list(src.get("manual_additions", []))),
        "unmatched_manual_boxes": copy.deepcopy(list(src.get("unmatched_manual_boxes", []))),
    }


def apply_label_state(dst: Dict[str, Any], state: Dict[str, Any]) -> None:
    dst["confirmed_ids"] = coerce_id_set(state.get("confirmed_ids", set()))
    dst["deleted_ids"] = coerce_id_set(state.get("deleted_ids", set()))
    dst["manual_additions"] = copy.deepcopy(list(state.get("manual_additions", [])))
    dst["unmatched_manual_boxes"] = copy.deepcopy(list(state.get("unmatched_manual_boxes", [])))


def get_working_label_state(fstate: Dict[str, Any]) -> Dict[str, Any]:
    """Return active label state dict (draft while editing, else committed fstate)."""
    if bool(fstate.get("edit_mode")) and isinstance(fstate.get("_edit_draft"), dict):
        return fstate["_edit_draft"]
    return fstate


def enter_edit_mode(fstate: Dict[str, Any]) -> None:
    if bool(fstate.get("edit_mode")) and isinstance(fstate.get("_edit_draft"), dict):
        return
    baseline = snapshot_label_state(fstate)
    draft = snapshot_label_state(fstate)
    fstate["edit_mode"] = True
    fstate["_edit_baseline"] = baseline
    fstate["_edit_draft"] = draft
    # Track which confirmations are attributable to manual additions so removing a
    # manual record can revert to undecided unless explicitly confirmed.
    manual_ids = set()
    for rec in draft.get("manual_additions", []):
        cid = rec.get("snapped_candidate_id")
        if cid:
            manual_ids.add(str(cid))
    fstate["_edit_manual_confirmed_ids"] = set(draft.get("confirmed_ids", set())) & manual_ids


def cancel_edit_mode(fstate: Dict[str, Any]) -> None:
    baseline = fstate.get("_edit_baseline")
    if isinstance(baseline, dict):
        apply_label_state(fstate, baseline)
    fstate["edit_mode"] = False
    fstate["_edit_baseline"] = None
    fstate["_edit_draft"] = None
    fstate["_edit_manual_confirmed_ids"] = set()


def save_edit_mode(fstate: Dict[str, Any]) -> None:
    draft = fstate.get("_edit_draft")
    if isinstance(draft, dict):
        apply_label_state(fstate, draft)
    fstate["edit_mode"] = False
    fstate["_edit_baseline"] = None
    fstate["_edit_draft"] = None
    fstate["_edit_manual_confirmed_ids"] = set()


def make_labels_payload_from_fstate(fstate: Dict[str, Any]) -> Dict[str, Any]:
    """Create schema v2 labels payload from an in-memory file state."""
    return {
        "schema_version": LABELS_SCHEMA_VERSION,
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "confirmed_ids": sorted(list(coerce_id_set(fstate.get("confirmed_ids", set())))),
        "deleted_ids": sorted(list(coerce_id_set(fstate.get("deleted_ids", set())))),
        "manual_additions": list(fstate.get("manual_additions", [])),
        "unmatched_manual_boxes": list(fstate.get("unmatched_manual_boxes", [])),
    }

