"""Right-side review panel + top controls for the Streamlit UI."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from door_detector.signatures import compute_analysis_signature
from door_detector.reweight_fit import fit_reweighter

from door_detector.ui.labels import (
    cancel_edit_mode as _cancel_edit_mode,
    enter_edit_mode as _enter_edit_mode,
    flatten_confirmed_ids,
    get_working_label_state as _get_working_label_state,
    make_labels_payload_from_fstate,
    save_edit_mode as _save_edit_mode,
    save_labels,
)
from door_detector.doors.types import DOOR_TYPES, normalize_door_type


logger = logging.getLogger("door_detector.review_app")


def _queue_pipeline_run(file_id: str, file_dir_str: str, config_path: str, label: str) -> None:
    # Queue work; it will execute on the next run so the viewer can paint a loader first.
    st.session_state.door_detector_pipeline_task = {
        "file_id": str(file_id),
        "file_dir": str(file_dir_str),
        "config_path": str(config_path),
        "label": str(label),
        "_started": False,
    }


def get_current_signature(config_path: str):
    try:
        return compute_analysis_signature(Path(config_path))
    except Exception:
        return None


def _get_lib():
    return st.session_state.library


def _delete_library_item_and_reset_ui(file_id: str) -> None:
    lib = _get_lib()
    lib.delete_item(file_id)
    # Clear selection + per-file UI state to avoid dangling widget keys.
    try:
        st.session_state.files.pop(file_id, None)
    except Exception:
        pass
    for k in [
        f"auto_focus_{file_id}",
        f"auto_focus_sink_{file_id}",
        f"jump_{file_id}",
        f"door_click_sink_{file_id}",
        f"selected_door_sink_{file_id}",
        f"focus_seq_sink_{file_id}",
        f"edit_mode_sink_{file_id}",
        f"draw_event_sink_{file_id}",
        f"manual_overlay_sink_{file_id}",
        f"door_state_sink_{file_id}",
        f"viewer_display_sink_{file_id}",
        f"confirm_delete_{file_id}",
    ]:
        try:
            st.session_state.pop(k, None)
        except Exception:
            pass
    st.session_state.selected_file_id = None

    st.cache_data.clear()
    try:
        st.cache_resource.clear()
    except Exception:
        pass

    # Labels schema changed (v2-only); avoid any lingering cached file artifacts.
    try:
        st.session_state.pop("_last_loaded_labels_schema_version", None)
    except Exception:
        pass


def save_current_labels(file_id: str, file_dir: Path) -> None:
    fstate = st.session_state.files[file_id]
    labels_to_save = make_labels_payload_from_fstate(fstate)
    save_labels(file_dir, labels_to_save)


def main_viewer_controls(
    item: Dict,
    *,
    full_dims: Optional[Tuple[int, int]],
    doors_data: Dict,
    fstate: Dict,
) -> None:
    lib = _get_lib()

    file_id = item["id"]
    file_dir = Path(item["path"])
    # Some discovered file ids can include characters like '(' which are not valid
    # HTML element ids; Streamlit may then omit the button id attribute. Use a
    # safe hashed suffix for widget keys that we want to style via CSS.
    key_suffix = hashlib.md5(str(file_id).encode("utf-8")).hexdigest()[:12]

    # Grid for main controls
    c1, c2, c_del = st.columns([2, 2, 1])
    with c1:
        status = item.get("status", "not_processed")
        if status == "processing":
            # Streamlit runs the pipeline synchronously; the UI won't render mid-run.
            # So a persisted "processing" status is stale and should not block running.
            doors_path = file_dir / "doors.json"
            if doors_path.exists():
                lib.update_status(file_id, "done")
                status = "done"
            else:
                lib.update_status(file_id, "not_processed")
                status = "not_processed"
            st.cache_data.clear()
            st.rerun()

        config_path = "configs/door_rules.json"
        current_sig = get_current_signature(config_path)
        stored_sig = doors_data.get("analysis_signature")
        is_out_of_date = stored_sig and current_sig and stored_sig != current_sig

        label = "Re-analyze" if status == "done" else "Analyze"
        if is_out_of_date:
            label = f"{label} (!)"

        task = st.session_state.get("door_detector_pipeline_task")
        is_running_for_file = bool(task and task.get("file_id") == str(file_id))
        analysis_label = f"Analyzing {item.get('original_name', '')}".strip() or "Analyzing…"

        st.button(
            label,
            type="primary" if not status == "done" else "secondary",
            use_container_width=True,
            disabled=is_running_for_file,
            on_click=_queue_pipeline_run,
            args=(str(file_id), str(file_dir), str(config_path), analysis_label),
        )
    with c2:
        modes = ["Highlight All", "Highlight Selected", "Off"]
        fstate["viewer_display_mode"] = st.selectbox(
            "Mode",
            modes,
            index=modes.index(fstate.get("viewer_display_mode"))
            if fstate.get("viewer_display_mode") in modes
            else 0,
            label_visibility="collapsed",
        )
    with c_del:
        confirm_key = f"confirm_delete_{file_id}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False

        if not st.session_state[confirm_key]:
            if st.button(
                "Delete",
                key=f"delete_btn_{key_suffix}",
                help="Remove this PDF from the library (deletes source.pdf and all artifacts).",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state[confirm_key] = True
                st.rerun()
        else:
            # Stack buttons vertically to avoid extreme wrapping in narrow layouts.
            if st.button(
                "Confirm",
                key=f"delete_confirm_btn_{key_suffix}",
                use_container_width=True,
                type="primary",
            ):
                _delete_library_item_and_reset_ui(file_id)
                st.rerun()
            if st.button(
                "Cancel",
                key=f"delete_cancel_btn_{key_suffix}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state[confirm_key] = False
                st.rerun()

    c3, c4 = st.columns(2)
    with c3:
        if not bool(fstate.get("edit_mode")):
            if st.button("Edit Doors", use_container_width=True, type="secondary"):
                _enter_edit_mode(fstate)
                st.rerun()
        else:
            col_save, col_cancel = st.columns(2)
            if col_save.button("Save", use_container_width=True, type="primary"):
                _save_edit_mode(fstate)
                save_current_labels(str(file_id), file_dir)
                st.rerun()
            if col_cancel.button("Cancel", use_container_width=True, type="secondary"):
                _cancel_edit_mode(fstate)
                st.rerun()
            st.caption("Shift+drag to add rectangles (snap-to-candidate).")
    with c4:
        auto_focus_key = f"auto_focus_{file_id}"
        # Keep widget state and per-file fstate in sync.
        if auto_focus_key not in st.session_state:
            st.session_state[auto_focus_key] = bool(fstate.get("auto_focus", True))
        fstate["auto_focus"] = st.checkbox("Auto-focus", key=auto_focus_key)


def _sync_selected_door_for_run(
    *,
    file_id: str,
    fstate: Dict[str, Any],
    all_visible: List[Dict[str, Any]],
) -> None:
    """Sync selected door state BEFORE rendering the main viewer."""
    if not all_visible:
        fstate["selected_door_id"] = None
        return

    # Normalize door ids to strings. The viewer communicates via string sinks.
    door_ids = [str(d.get("id")) for d in all_visible if d.get("id") is not None]
    if not door_ids:
        fstate["selected_door_id"] = None
        return

    jump_key = f"jump_{file_id}"
    idx_key = f"{jump_key}__idx"
    click_sink_key = f"door_click_sink_{file_id}"

    def _coerce_idx(v: Any) -> Optional[int]:
        try:
            return int(v)
        except Exception:
            return None

    # Establish current selection.
    # Priority: clicked bbox → explicit index → previous selection → first.
    current_id: Optional[Any] = None
    clicked_id = st.session_state.get(click_sink_key)
    if clicked_id not in (None, ""):
        clicked_id = str(clicked_id)
    # Consume internal sentinel events (used to force reruns from the viewer).
    if isinstance(clicked_id, str) and clicked_id.startswith("__draw_event__"):
        try:
            st.session_state[click_sink_key] = ""
        except Exception:
            pass
        clicked_id = ""
    if clicked_id in door_ids:
        # Treat the click sink as an event: consume once, then clear so it doesn't
        # keep overriding other controls on subsequent runs.
        fstate["_last_clicked_door_id"] = clicked_id
        current_id = clicked_id
        try:
            st.session_state[click_sink_key] = ""
        except Exception:
            pass
    else:
        # If the user typed an index, treat it as the requested selection.
        idx_req = _coerce_idx(st.session_state.get(idx_key))
        if isinstance(idx_req, int):
            # Wrap-around navigation.
            if idx_req <= 0:
                current_id = door_ids[-1]
            elif idx_req > len(door_ids):
                current_id = door_ids[0]
            else:
                current_id = door_ids[idx_req - 1]
        else:
            jump_raw = st.session_state.get(jump_key)
            jump_id = str(jump_raw) if jump_raw not in (None, "") else None
            if jump_id in door_ids:
                current_id = jump_id
            else:
                sel_raw = fstate.get("selected_door_id")
                sel_id = str(sel_raw) if sel_raw not in (None, "") else None
                if sel_id in door_ids:
                    current_id = sel_id
                else:
                    current_id = door_ids[0]

    # Make selection canonical for the rest of this run.
    if st.session_state.get(jump_key) != current_id:
        st.session_state[jump_key] = current_id
    try:
        st.session_state[idx_key] = door_ids.index(current_id) + 1
    except Exception:
        st.session_state[idx_key] = 1
    fstate["selected_door_id"] = str(current_id) if current_id not in (None, "") else None

    # Bump focus sequence when selection changes (so the viewer auto-focuses only on changes).
    if current_id != fstate.get("_focus_last_id"):
        fstate["_focus_last_id"] = current_id
        try:
            fstate["_focus_seq"] = int(fstate.get("_focus_seq") or 0) + 1
        except Exception:
            fstate["_focus_seq"] = 1


def right_panel_review(
    item: Dict,
    *,
    doors_data: Dict,
    fstate: Dict,
    active_doors: List,
    all_active_doors: Optional[List] = None,
) -> None:
    file_id = item["id"]
    file_dir = Path(item["path"])

    # Don't show "Doors (0)" until analysis has been run at least once.
    status = item.get("status", "not_processed")
    has_run = (status == "done") or (file_dir / "doors.json").exists()
    if not has_run:
        st.info("Analyze to see doors.")
        return

    # Use pre-calculated active_doors so the main viewer + right panel stay in sync.
    all_visible = active_doors.copy()
    all_visible.sort(key=lambda x: x["confidence"], reverse=True)

    total_doors = len(all_active_doors) if isinstance(all_active_doors, list) else len(all_visible)
    door_type_filter_key = f"door_type_filter_{file_id}"
    current_filter = str(st.session_state.get(door_type_filter_key) or "All")
    if current_filter and current_filter != "All":
        st.subheader(f"Doors ({len(all_visible)} of {total_doors})")
    else:
        st.subheader(f"Doors ({len(all_visible)})")

    if not all_visible:
        return

    # Filter panel (stored in session_state; applied before rendering the viewer).
    all_for_filter = (all_active_doors if isinstance(all_active_doors, list) else active_doors) or []
    type_values = sorted(
        {
            str(d.get("type")).strip()
            for d in all_for_filter
            if d.get("type") is not None and str(d.get("type")).strip()
        }
    )
    type_options = ["All"] + type_values
    if door_type_filter_key not in st.session_state:
        st.session_state[door_type_filter_key] = "All"
    if str(st.session_state.get(door_type_filter_key) or "All") not in type_options:
        st.session_state[door_type_filter_key] = "All"

    f_lbl, f_sel = st.columns([1, 3])
    f_lbl.markdown("<div style='line-height: 38px; opacity: 0.85;'>Type</div>", unsafe_allow_html=True)
    f_sel.selectbox("Door type", type_options, key=door_type_filter_key, label_visibility="collapsed")

    # Door navigation (index input + Prev/Next).
    door_ids = [str(d["id"]) for d in all_visible if d.get("id") is not None]
    if not door_ids:
        return

    jump_key = f"jump_{file_id}"
    idx_key = f"{jump_key}__idx"
    idx_label = f"door_jump_idx_{file_id}"

    jump_raw = st.session_state.get(jump_key)
    jump_id = str(jump_raw) if jump_raw not in (None, "") else None
    current_id = jump_id if (jump_id in door_ids) else door_ids[0]
    selected_idx = door_ids.index(current_id) if current_id in door_ids else 0
    # Keep the editable index input in sync with selection (must happen before widget instantiation).
    if st.session_state.get(idx_key) != (selected_idx + 1):
        st.session_state[idx_key] = selected_idx + 1

    col_idx = st.container()
    with col_idx:
        c_in, c_total = st.columns([1, 1])
        c_in.number_input(
            idx_label,
            min_value=0,
            max_value=len(door_ids) + 1,
            step=1,
            key=idx_key,
            label_visibility="collapsed",
        )
        c_total.markdown(
            f"<div style='text-align: left; line-height: 38px; padding-left: 6px;'>/ {len(all_visible)}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Details of selected (selection canonicalized earlier by _sync_selected_door_for_run in the app).
    current_id = str(fstate.get("selected_door_id") or "")
    if current_id not in door_ids:
        current_id = door_ids[0]
    selected_idx = door_ids.index(current_id) if current_id in door_ids else 0
    selected_door = all_visible[selected_idx]
    did = str(selected_door.get("id") or "")
    door_type = html.escape(str(selected_door.get("type", "")))
    try:
        conf = float(selected_door.get("confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    conf_pct = int(round(conf * 100))

    st.markdown(
        f"""
<div class="door_detector-door-meta">
  <div class="door_detector-door-meta-item">
    <span class="door_detector-door-meta-label">Type</span>
    <span class="door_detector-door-meta-type">{door_type}</span>
  </div>
  <div class="door_detector-door-meta-item">
    <span class="door_detector-door-meta-label">Confidence</span>
    <span class="door_detector-door-meta-confidence">{conf_pct}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Typed label control (what the reviewer says this door *is*).
    # This is separate from the model-predicted door type (displayed above).
    label_type_key = f"label_type_{file_id}"
    working = _get_working_label_state(fstate)
    # Compute current labeled type (if confirmed), else default to predicted type.
    labeled_type = None
    try:
        cbt = working.get("confirmed_by_type", {}) if isinstance(working, dict) else {}
        if isinstance(cbt, dict):
            for t in DOOR_TYPES:
                ids = cbt.get(t)
                if isinstance(ids, set) and did in ids:
                    labeled_type = t
                    break
    except Exception:
        labeled_type = None
    default_label_type = labeled_type or normalize_door_type(selected_door.get("type"), default="swing")
    if label_type_key not in st.session_state:
        st.session_state[label_type_key] = default_label_type
    if str(st.session_state.get(label_type_key) or "") not in DOOR_TYPES:
        st.session_state[label_type_key] = default_label_type
    st.selectbox("Label as", list(DOOR_TYPES), key=label_type_key)

    # Actions
    is_editing = bool(fstate.get("edit_mode"))
    c1, c2, c3 = st.columns(3)
    if c1.button("Confirm door", use_container_width=True):
        # Ensure confirmed_by_type exists.
        try:
            cbt = working.get("confirmed_by_type")
            if not isinstance(cbt, dict):
                cbt = {t: set() for t in DOOR_TYPES}
                working["confirmed_by_type"] = cbt
        except Exception:
            cbt = {t: set() for t in DOOR_TYPES}
            working["confirmed_by_type"] = cbt

        label_type = normalize_door_type(st.session_state.get(label_type_key), default=default_label_type)
        # Candidate can only be confirmed as exactly one type.
        for t in DOOR_TYPES:
            try:
                ids = cbt.get(t)
                if isinstance(ids, set):
                    ids.discard(did)
            except Exception:
                continue
        cbt.setdefault(label_type, set()).add(did)
        working["deleted_ids"].discard(did)
        # Treat as explicit confirmation (so removing a manual-add record won't unconfirm).
        if is_editing:
            try:
                fstate["_edit_manual_confirmed_ids"].discard(did)
            except Exception:
                pass
        else:
            save_current_labels(str(file_id), file_dir)
        st.rerun()
    if c2.button("Delete / Not a door", use_container_width=True):
        working["deleted_ids"].add(did)
        try:
            cbt = working.get("confirmed_by_type")
            if isinstance(cbt, dict):
                for t in DOOR_TYPES:
                    ids = cbt.get(t)
                    if isinstance(ids, set):
                        ids.discard(did)
        except Exception:
            pass
        if is_editing:
            try:
                fstate["_edit_manual_confirmed_ids"].discard(did)
            except Exception:
                pass
            # If the user marks a candidate as not-a-door, drop any manual-add records
            # that snapped to it (they are no longer meaningful).
            working["manual_additions"] = [
                r
                for r in list(working.get("manual_additions", []))
                if str(r.get("snapped_candidate_id") or "") != str(did)
            ]
        else:
            save_current_labels(str(file_id), file_dir)
        fstate["selected_door_id"] = None  # Move to next
        st.rerun()
    if c3.button("Skip", use_container_width=True):
        if selected_idx < len(all_visible) - 1:
            next_id = all_visible[selected_idx + 1]["id"]
            fstate["selected_door_id"] = next_id
            st.session_state[jump_key] = next_id
            st.rerun()

    st.divider()
    # Show stats for the currently active label state (draft while editing).
    st.write(
        f"**Stats:** "
        f"{len(flatten_confirmed_ids(working.get('confirmed_by_type', {})))} confirmed, "
        f"{len(working.get('deleted_ids', set()))} deleted, "
        f"{len(working.get('manual_additions', []))} manual-added, "
        f"{len(working.get('unmatched_manual_boxes', []))} unmatched"
    )

    if is_editing:
        st.divider()
        st.subheader("Edit Doors")

    # Train badge
    total_overrides = len(flatten_confirmed_ids(working.get("confirmed_by_type", {}))) + len(working.get("deleted_ids", set()))
    if (not is_editing) and total_overrides >= 5:
        if st.button("Train Model", use_container_width=True):
            with st.spinner("Training..."):
                fit_reweighter(Path("artifacts"), Path("models"))
                st.success("Model updated!")
                st.cache_data.clear()
                st.rerun()

