"""Right-side review panel + top controls for the Streamlit UI."""

from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from door_detector.signatures import compute_analysis_signature

from door_detector.ui.labels import (
    flatten_confirmed_ids,
    flatten_rejected_ids,
    get_working_label_state as _get_working_label_state,
    make_labels_payload_from_fstate,
    save_labels,
)
from door_detector.doors.types import DOOR_TYPES, normalize_door_type
from door_detector.ui.ui_debug import push_breadcrumb, tail_breadcrumbs, warn_once, sample_ids, ui_event_log


def _default_config_path_str() -> str:
    """Best-effort default config path that works when launched outside repo root."""
    p = Path("configs/door_rules.json")
    if p.exists():
        return str(p)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        p2 = repo_root / "configs" / "door_rules.json"
        return str(p2)
    except Exception:
        return str(p)


def _nav_intent_key(file_id: str) -> str:
    # Flag set in session_state to indicate the user explicitly navigated to a new door
    # (e.g. by typing an index). Used to control autofocus behavior.
    return f"_door_detector_nav_intent__{file_id}"


def _suppress_autofocus_key(file_id: str) -> str:
    # One-shot suppression flag for autofocus on the *next* selection sync.
    # Used for cases like Delete/Not-a-door where selection advances automatically.
    return f"_door_detector_suppress_autofocus_once__{file_id}"


def _mark_nav_intent(file_id: str) -> None:
    try:
        st.session_state[_nav_intent_key(str(file_id))] = True
    except Exception:
        return


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


def _archive_library_item_and_reset_ui(file_id: str) -> None:
    lib = _get_lib()
    lib.archive_item(file_id)
    # Clear selection + per-file UI state to avoid dangling widget keys.
    try:
        st.session_state.files.pop(file_id, None)
    except Exception:
        pass
    for k in [
        f"auto_focus_{file_id}",
        f"auto_focus_sink_{file_id}",
        f"viewer_display_mode_{file_id}",
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

    task = st.session_state.get("door_detector_pipeline_task")
    is_running_for_file = bool(task and task.get("file_id") == str(file_id))
    # Used as a stable label for the queued pipeline task.
    analysis_label = f"Analyzing {item.get('original_name', '')}".strip() or "Analyzing…"

    # Small top pad so the right panel aligns with the viewer window.
    st.markdown('<div class="door_detector-review-panel-top-pad"></div>', unsafe_allow_html=True)

    # Row 1: Analyze + Delete
    c_an, c_del = st.columns([4, 1])
    with c_an:
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

        config_path = _default_config_path_str()
        current_sig = get_current_signature(config_path)
        stored_sig = doors_data.get("analysis_signature")
        is_out_of_date = stored_sig and current_sig and stored_sig != current_sig

        label = "Re-analyze" if status == "done" else "Analyze"
        if is_out_of_date:
            label = f"{label} (!)"

        st.button(
            label,
            type="primary" if not status == "done" else "secondary",
            use_container_width=True,
            disabled=is_running_for_file,
            on_click=_queue_pipeline_run,
            args=(str(file_id), str(file_dir), str(config_path), analysis_label),
        )
    with c_del:
        confirm_key = f"confirm_delete_{file_id}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False

        if not st.session_state[confirm_key]:
            if st.button(
                "Archive",
                key=f"delete_btn_{key_suffix}",
                help="Archive this PDF (removes it from the library list but keeps labels/artifacts on disk).",
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
                _archive_library_item_and_reset_ui(file_id)
                st.rerun()
            if st.button(
                "Cancel",
                key=f"delete_cancel_btn_{key_suffix}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state[confirm_key] = False
                st.rerun()

    # Row 2: Highlight doors toggle + Auto-focus (left-justified, side-by-side)
    c_hl, c_auto, _spacer = st.columns([1, 1, 3])
    with c_hl:
        highlight_key = f"highlight_doors_{file_id}"
        if highlight_key not in st.session_state:
            st.session_state[highlight_key] = True
        fstate["highlight_doors"] = st.checkbox("Highlight doors", key=highlight_key)
    with c_auto:
        auto_focus_key = f"auto_focus_{file_id}"
        # Keep widget state and per-file fstate in sync.
        if auto_focus_key not in st.session_state:
            st.session_state[auto_focus_key] = bool(fstate.get("auto_focus", True))
        fstate["auto_focus"] = st.checkbox("Auto-focus", key=auto_focus_key)

    # Type filter bubbles live with the top controls.
    # Compute counts based on the visible candidate set (excluding deleted/rejected),
    # and include any manual candidates / confirmed extras so "All (N)" matches the UI list.
    working = _get_working_label_state(fstate)
    try:
        hidden_ids = set(working.get("deleted_ids", set())) | set(flatten_rejected_ids(working.get("rejected_by_type", {})))
    except Exception:
        hidden_ids = set()

    doors = list(doors_data.get("doors", []) or [])
    candidates = list(doors_data.get("candidates", []) or [])
    try:
        candidates.extend(list(working.get("manual_candidates", []) or []))
    except Exception:
        pass

    by_id = {str(c.get("id")): c for c in candidates if isinstance(c, dict) and c.get("id") is not None}
    door_ids = {str(d.get("id")) for d in doors if isinstance(d, dict) and d.get("id") is not None}

    try:
        extra_ids = set(flatten_confirmed_ids(working.get("confirmed_by_type", {})))
    except Exception:
        extra_ids = set()
    try:
        for mc in list(working.get("manual_candidates", []) or []):
            if isinstance(mc, dict) and mc.get("id") not in (None, ""):
                extra_ids.add(str(mc.get("id")))
    except Exception:
        pass
    try:
        prop = fstate.get("_proposal") or {}
        if isinstance(prop, dict) and prop.get("snapped_candidate_id") not in (None, ""):
            extra_ids.add(str(prop.get("snapped_candidate_id")))
    except Exception:
        pass

    visible_items: list[dict[str, Any]] = []
    for d in doors:
        if not isinstance(d, dict) or d.get("id") is None:
            continue
        sid = str(d.get("id"))
        if sid in hidden_ids:
            continue
        visible_items.append(d)
    for sid in sorted(extra_ids - door_ids):
        if not sid or sid in hidden_ids:
            continue
        c = by_id.get(str(sid))
        if isinstance(c, dict):
            visible_items.append(c)

    # Unified single-stage filter pills:
    # - One "All" option only
    # - Other options are either {Confirmed, Unconfirmed} OR a door type
    # - User chooses exactly one slice (no combining type+confirmed filters)
    try:
        confirmed_ids = set(flatten_confirmed_ids(working.get("confirmed_by_type", {})))
    except Exception:
        confirmed_ids = set()

    base_total = len(visible_items)
    try:
        base_confirmed = sum(
            1 for d in visible_items if isinstance(d, dict) and str(d.get("id") or "") in confirmed_ids
        )
    except Exception:
        base_confirmed = 0
    base_unconfirmed = max(0, base_total - base_confirmed)

    counts: dict[str, int] = {}
    for d in visible_items:
        t = str(d.get("type") or "").strip()
        if not t:
            continue
        counts[t] = counts.get(t, 0) + 1
    type_values = sorted(counts.keys())
    # IMPORTANT: keep filter state separate from the widget key.
    #
    # Using the same key for both the Streamlit widget and our application state caused
    # intermittent resets on action-triggered reruns (e.g. Confirm), which could briefly
    # render the UI as "All" before restoring.
    #
    # - `door_filter_state_key` is the canonical app state used by app.py to filter overlays.
    # - `door_filter_widget_key` is the radio widget state.
    # Use a distinct prefix so Streamlit never confuses it with historical widget keys.
    door_filter_state_key = f"_door_detector_door_filter_state_{file_id}"
    door_filter_widget_key = f"door_filter_widget_{file_id}"  # radio widget key
    filter_options = ["All", "Confirmed", "Unconfirmed"] + type_values
    if door_filter_state_key not in st.session_state:
        try:
            prev = str(fstate.get("_door_filter") or "")
        except Exception:
            prev = ""
        st.session_state[door_filter_state_key] = prev if prev else "All"
    # Initialize widget state from canonical state when needed.
    canonical_filter = str(st.session_state.get(door_filter_state_key) or "All")
    if door_filter_widget_key not in st.session_state:
        st.session_state[door_filter_widget_key] = canonical_filter
    # Preserve the user's filter choice even if its current count becomes 0.
    #
    # Streamlit radios require the current value be present in `options`; previously we
    # "fixed" missing values by resetting to "All", which felt like the UI was randomly
    # losing the filter after actions like confirm/reject/delete (those can remove the
    # last remaining door of a type, temporarily making that type absent from `type_values`).
    cur_filter = canonical_filter
    if cur_filter not in filter_options:
        # Keep the filter visible (it will show "(0)" via format_func).
        filter_options.append(cur_filter)
    # Mirror into per-file state so actions that trigger reruns (confirm/reject/delete)
    # don't accidentally fall back to "All" if Streamlit drops the widget key.
    try:
        fstate["_door_filter"] = str(cur_filter)
    except Exception:
        pass

    # Debug: detect unexpected filter resets/missing values.
    try:
        last_logged = str(fstate.get("_door_filter_last_logged") or "")
    except Exception:
        last_logged = ""
    if cur_filter != last_logged:
        _ui_log(
            "door_filter_effective",
            {
                "file_id": str(file_id),
                "cur_filter": str(cur_filter),
                "prev_filter": last_logged,
                "options": list(filter_options),
                "base_total": int(base_total),
                "base_confirmed": int(base_confirmed),
                "base_unconfirmed": int(base_unconfirmed),
            },
        )
        try:
            fstate["_door_filter_last_logged"] = str(cur_filter)
        except Exception:
            pass

    # Record explicit user interaction with the filter widget so app.py can distinguish
    # a real user change ("I clicked All") from unexpected resets ("it flipped to All").
    def _mark_door_filter_touched() -> None:
        try:
            st.session_state[f"_door_filter_user_changed_{file_id}"] = True
            v = str(st.session_state.get(door_filter_widget_key) or "All")
            st.session_state[f"_door_filter_user_value_{file_id}"] = v
            # Canonicalize into the app-state key for immediate use at rerun start.
            st.session_state[door_filter_state_key] = v
            fstate["_door_filter"] = v
        except Exception:
            return

    st.radio(
        "Filter doors",
        filter_options,
        key=door_filter_widget_key,
        horizontal=True,
        label_visibility="collapsed",
        on_change=_mark_door_filter_touched,
        format_func=lambda v: (
            f"All ({base_total})"
            if str(v) == "All"
            else f"Confirmed ({base_confirmed})"
            if str(v) == "Confirmed"
            else f"Unconfirmed ({base_unconfirmed})"
            if str(v) == "Unconfirmed"
            else f"{str(v).capitalize()} ({int(counts.get(str(v), 0))})"
        ),
    )

    # Defensive: after the widget renders, keep canonical state in sync with the widget value.
    # This preserves immediate updates on reruns (app.py reads the canonical key).
    try:
        v = str(st.session_state.get(door_filter_widget_key) or "All")
        st.session_state[door_filter_state_key] = v
        fstate["_door_filter"] = v
    except Exception:
        pass

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

    # Determine whether this rerun should be allowed to autofocus if selection changes.
    # Principle:
    # - Autofocus ONLY when the user explicitly selected/navigated (click in viewer,
    #   or explicit navigation controls in the right panel).
    # - Do NOT autofocus for automatic selection changes (initial load, delete-driven
    #   move-to-next, filter-driven reshuffles, etc.).
    allow_autofocus = False
    try:
        allow_autofocus = bool(st.session_state.pop(_nav_intent_key(file_id), False))
    except Exception:
        allow_autofocus = False

    suppress_autofocus = False
    try:
        suppress_autofocus = bool(st.session_state.pop(_suppress_autofocus_key(file_id), False))
    except Exception:
        suppress_autofocus = False

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
        allow_autofocus = True
        push_breadcrumb(
            fstate,
            {
                "kind": "selection_intent",
                "source": "viewer_click",
                "file_id": str(file_id),
                "clicked_id": str(clicked_id),
                "visible_count": int(len(door_ids)),
            },
        )
        try:
            st.session_state[click_sink_key] = ""
        except Exception:
            pass
    else:
        # If the viewer emitted an id that isn't in the current visible list, that suggests
        # stale client state or a filter/overlay mismatch.
        if clicked_id not in (None, ""):
            bad = str(clicked_id)
            k = f"viewer_click_id_not_visible::{file_id}::{bad}"
            if warn_once(fstate, k):
                # Logging intentionally suppressed (noise). Use breadcrumbs/state snapshots instead.
                pass

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
            push_breadcrumb(
                fstate,
                {
                    "kind": "selection_intent",
                    "source": "index",
                    "file_id": str(file_id),
                    "idx_req": int(idx_req),
                    "visible_count": int(len(door_ids)),
                },
            )
        else:
            jump_raw = st.session_state.get(jump_key)
            jump_id = str(jump_raw) if jump_raw not in (None, "") else None
            if jump_id in door_ids:
                current_id = jump_id
                push_breadcrumb(
                    fstate,
                    {
                        "kind": "selection_intent",
                        "source": "jump",
                        "file_id": str(file_id),
                        "jump_id": str(jump_id),
                    },
                )
            else:
                sel_raw = fstate.get("selected_door_id")
                sel_id = str(sel_raw) if sel_raw not in (None, "") else None
                if sel_id in door_ids:
                    current_id = sel_id
                else:
                    current_id = door_ids[0]
                    prev = str(sel_id or "")
                    k = f"selection_coerced_to_first::{file_id}::{prev}::{str(current_id)}"
                    if warn_once(fstate, k):
                        # Logging intentionally suppressed (noise). Use breadcrumbs/state snapshots instead.
                        pass

    # Make selection canonical for the rest of this run.
    if st.session_state.get(jump_key) != current_id:
        st.session_state[jump_key] = current_id
    try:
        st.session_state[idx_key] = door_ids.index(current_id) + 1
    except Exception:
        st.session_state[idx_key] = 1
    fstate["selected_door_id"] = str(current_id) if current_id not in (None, "") else None
    push_breadcrumb(
        fstate,
        {
            "kind": "selection_canonicalized",
            "file_id": str(file_id),
            "selected_door_id": str(fstate.get("selected_door_id") or ""),
            "visible_count": int(len(door_ids)),
        },
    )

    # Bump focus sequence when selection changes (so the viewer auto-focuses only on changes).
    if current_id != fstate.get("_focus_last_id"):
        fstate["_focus_last_id"] = current_id
        auto_focus_key = f"auto_focus_{file_id}"
        if auto_focus_key in st.session_state:
            auto_focus_enabled = bool(st.session_state.get(auto_focus_key))
        else:
            auto_focus_enabled = bool(fstate.get("auto_focus", True))

        if allow_autofocus and (not suppress_autofocus) and auto_focus_enabled:
            try:
                fstate["_focus_seq"] = int(fstate.get("_focus_seq") or 0) + 1
            except Exception:
                fstate["_focus_seq"] = 1
            # Record that we intentionally focused this door (for Focus button visibility).
            try:
                fstate["_focused_door_id"] = str(current_id) if current_id not in (None, "") else None
            except Exception:
                pass


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
    # Some discovered file ids can include characters like '(' which are not valid
    # HTML element ids; Streamlit may then omit the button id attribute. Use a
    # safe hashed suffix for widget keys that we want to style via CSS.
    key_suffix = hashlib.md5(str(file_id).encode("utf-8")).hexdigest()[:12]

    task = st.session_state.get("door_detector_pipeline_task")
    is_running_for_file = bool(task and task.get("file_id") == str(file_id))

    # While analysis is running, replace the normal menu contents with the existing
    # "Analyzing…" state (same UX as first-time Analyze on an unprocessed file).
    if is_running_for_file:
        st.info("Analyzing…")
        return

    # Don't show "Doors (0)" until analysis has been run at least once.
    status = item.get("status", "not_processed")
    has_run = (status == "done") or (file_dir / "doors.json").exists()
    if not has_run:
        st.info("Analyze to see doors.")
        return

    # Use pre-calculated active_doors so the main viewer + right panel stay in sync.
    all_visible = active_doors.copy()
    all_visible.sort(key=lambda x: x["confidence"], reverse=True)

    door_filter_state_key = f"_door_detector_door_filter_state_{file_id}"
    door_filter_widget_key = f"door_filter_widget_{file_id}"
    # Doors header removed (redundant; selection section below is explicit).

    if not all_visible:
        return

    # If there is an active selection/proposal, show the selection menu *instead of*
    # the normal "selected door" details. After confirm/deny, the proposal is cleared
    # and this function will render the selected door details again.
    try:
        srec = fstate.get("_last_draw_suggestions") or {}
        sugg_all = list(srec.get("suggestions") or [])
    except Exception:
        sugg_all = []
    prop = fstate.get("_proposal")
    if isinstance(prop, dict):
        st.divider()

        # If suggestion computation failed (or was interrupted), still render a minimal
        # proposal UI so the user can cancel and clear any proposal overlay.
        if not sugg_all:
            c1, c2 = st.columns(2)
            if c1.button("Cancel proposal", use_container_width=True, type="secondary", key=f"proposal_cancel_min_{key_suffix}"):
                try:
                    created_id = str(prop.get("created_manual_candidate_id") or "")
                except Exception:
                    created_id = ""
                if created_id:
                    try:
                        mc = list(fstate.get("manual_candidates", []) or [])
                        fstate["manual_candidates"] = [
                            c for c in mc if not (isinstance(c, dict) and str(c.get("id") or "") == created_id)
                        ]
                    except Exception:
                        pass
                # Restore previous selection (best effort).
                try:
                    prev_id = str(prop.get("prev_selected_door_id") or "")
                except Exception:
                    prev_id = ""
                if prev_id:
                    fstate["selected_door_id"] = prev_id
                    try:
                        st.session_state[f"jump_{file_id}"] = prev_id
                    except Exception:
                        pass
                fstate["_proposal"] = None
                fstate["_last_draw_suggestions"] = None
                try:
                    st.session_state[f"_draw_suggest_idx_{file_id}"] = 0
                except Exception:
                    pass
                st.rerun()
            if c2.button("Keep (no suggestions)", use_container_width=True, type="primary", key=f"proposal_keep_min_{key_suffix}"):
                st.rerun()

            st.caption("Proposal is active, but no snap list was computed. Cancel to remove the overlay and try Shift+drag again.")
            st.divider()
            st.markdown("**Tips**")
            st.caption("Shift+drag to propose selection · Drag to pan · Scroll to zoom")
            return

        # Reuse the same cycling keys so the viewer can highlight `cycle_candidate_id`
        # during the same rerun (app.py computes it from session_state).
        idx_key = f"_draw_suggest_idx_{file_id}"
        type_key = f"_draw_suggest_type_{file_id}"
        type_touched_key = f"_draw_suggest_type_touched_{file_id}"

        # Reset per-selection UI state when the selection event changes.
        try:
            ev_id = str(srec.get("event_id") or "")
        except Exception:
            ev_id = ""
        ev_key = f"_draw_suggest_event_id_{file_id}"
        if st.session_state.get(ev_key) != ev_id:
            st.session_state[ev_key] = ev_id
            st.session_state[type_touched_key] = False
            st.session_state[idx_key] = 0

        # Type dropdown behavior:
        # - Never show "All types"
        # - If the user has never interacted with it, DO NOT filter cycling; instead,
        #   auto-set the dropdown value to match the currently highlighted suggestion's type.
        # - Once the user selects a type, filter cycling to that type only.
        types = sorted({str(s.get("type") or "").strip() for s in sugg_all if str(s.get("type") or "").strip()})
        if not types:
            types = ["swing"]

        def _mark_type_touched() -> None:
            try:
                st.session_state[type_touched_key] = True
            except Exception:
                return

        touched = bool(st.session_state.get(type_touched_key, False))

        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0
        try:
            idx = int(st.session_state.get(idx_key) or 0)
        except Exception:
            idx = 0
        if idx < 0:
            idx = 0
        if idx >= len(sugg_all):
            idx = 0
        st.session_state[idx_key] = idx

        if not touched:
            cur_unfiltered = sugg_all[idx] if sugg_all else {}
            cur_unfiltered_type = str(cur_unfiltered.get("type") or "").strip()
            if cur_unfiltered_type in types:
                st.session_state[type_key] = cur_unfiltered_type
            elif str(st.session_state.get(type_key) or "") not in types:
                st.session_state[type_key] = types[0]
        else:
            if str(st.session_state.get(type_key) or "") not in types:
                st.session_state[type_key] = types[0]

        chosen_type = str(st.session_state.get(type_key) or types[0])

        use_filter = bool(st.session_state.get(type_touched_key, False))
        if use_filter:
            filtered = [s for s in sugg_all if str(s.get("type") or "").strip() == str(chosen_type)]
        else:
            filtered = sugg_all
        if not filtered:
            filtered = sugg_all
            use_filter = False

        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0
        try:
            idx = int(st.session_state.get(idx_key) or 0)
        except Exception:
            idx = 0
        if idx < 0:
            idx = 0
        if filtered and idx >= len(filtered):
            idx = 0
        st.session_state[idx_key] = idx

        # Header row aligned with non-proposal mode: Prev | Selection | Next
        c_prev, c_mid, c_next = st.columns([1, 2, 1])
        if c_prev.button("Prev", use_container_width=True, key=f"prev_draw_suggest_{file_id}"):
            st.session_state[idx_key] = (int(st.session_state.get(idx_key) or 0) - 1) % max(1, len(filtered))
            st.rerun()
        if c_mid.button(
            f"Selection Snap {idx+1} / {len(filtered)}",
            use_container_width=True,
            type="secondary",
            key=f"door_focus_btn_{key_suffix}",
        ):
            try:
                fstate["_proposal_focus_seq"] = int(fstate.get("_proposal_focus_seq") or 0) + 1
            except Exception:
                fstate["_proposal_focus_seq"] = 1
            st.rerun()
        if c_next.button("Next", use_container_width=True, key=f"next_draw_suggest_{file_id}"):
            st.session_state[idx_key] = (int(st.session_state.get(idx_key) or 0) + 1) % max(1, len(filtered))
            st.rerun()

        # Single type dropdown (controls filtering AND confirm type).
        chosen_type = st.selectbox(
            "Type",
            types,
            key=type_key,
            format_func=lambda t: str(t).capitalize(),
            label_visibility="collapsed",
            on_change=_mark_type_touched,
        )

        cur = filtered[idx] if filtered else sugg_all[0]
        cur_id = str(cur.get("id") or "")
        cur_type_raw = str(cur.get("type") or "")
        cur_iou = cur.get("iou")
        cur_conf = cur.get("confidence")

        # Selection actions apply to the currently-highlighted match.
        working = _get_working_label_state(fstate)
        label_type = normalize_door_type(str(chosen_type), default="swing")

        # Confirm / Cancel only (no typed reject / not-a-door-at-all here).
        c1, c2 = st.columns(2)
        if c1.button("Confirm", use_container_width=True, type="primary", key=f"proposal_confirm_{key_suffix}"):
            cbt = working.get("confirmed_by_type")
            if not isinstance(cbt, dict):
                cbt = {t: set() for t in DOOR_TYPES}
                working["confirmed_by_type"] = cbt
            for t in DOOR_TYPES:
                ids = cbt.get(t)
                if isinstance(ids, set):
                    ids.discard(cur_id)
            cbt.setdefault(label_type, set()).add(cur_id)
            working["deleted_ids"].discard(cur_id)
            rbt = working.get("rejected_by_type")
            if isinstance(rbt, dict):
                for t in DOOR_TYPES:
                    ids = rbt.get(t)
                    if isinstance(ids, set):
                        ids.discard(cur_id)

            # Persist immediately and clear proposal.
            save_current_labels(str(file_id), file_dir)
            fstate["_proposal"] = None
            fstate["_last_draw_suggestions"] = None
            try:
                st.session_state[idx_key] = 0
            except Exception:
                pass

            # Ensure the confirmed selection remains the selected door.
            fstate["selected_door_id"] = str(cur_id)
            try:
                st.session_state[f"jump_{file_id}"] = str(cur_id)
            except Exception:
                pass
            st.rerun()
        if c2.button("Cancel", use_container_width=True, type="secondary", key=f"proposal_deny_{key_suffix}"):
            # If the proposal created a manual candidate, drop it on deny.
            try:
                created_id = str(prop.get("created_manual_candidate_id") or "")
            except Exception:
                created_id = ""
            if created_id:
                try:
                    mc = list(fstate.get("manual_candidates", []) or [])
                    fstate["manual_candidates"] = [
                        c for c in mc if not (isinstance(c, dict) and str(c.get("id") or "") == created_id)
                    ]
                except Exception:
                    pass

            # Restore previous selection (best effort).
            try:
                prev_id = str(prop.get("prev_selected_door_id") or "")
            except Exception:
                prev_id = ""
            if prev_id:
                fstate["selected_door_id"] = prev_id
                try:
                    st.session_state[f"jump_{file_id}"] = prev_id
                except Exception:
                    pass

            fstate["_proposal"] = None
            fstate["_last_draw_suggestions"] = None
            try:
                st.session_state[idx_key] = 0
            except Exception:
                pass
            st.rerun()

        # Tips directly under selection actions.
        st.divider()
        st.markdown("**Tips**")
        st.caption("Shift+drag to propose selection · Drag to pan · Scroll to zoom")
        return

    # Door navigation.
    door_ids = [str(d["id"]) for d in all_visible if d.get("id") is not None]
    if not door_ids:
        return

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

    # Door navigation: Prev / Door X/Y / Next in one line.
    c_prev, c_mid, c_next = st.columns([1, 2, 1])
    if c_prev.button("Prev", use_container_width=True, key=f"door_prev_btn_{key_suffix}"):
        try:
            prev_id = door_ids[(int(selected_idx) - 1) % max(1, len(door_ids))]
            st.session_state[f"door_click_sink_{file_id}"] = str(prev_id)
        except Exception:
            pass
        _mark_nav_intent(str(file_id))
        st.rerun()

    if c_mid.button(
        f"Door {int(selected_idx) + 1} / {len(all_visible)}",
        use_container_width=True,
        type="secondary",
        key=f"door_focus_btn_{key_suffix}",
    ):
        try:
            fstate["_focus_request_seq"] = int(fstate.get("_focus_request_seq") or 0) + 1
        except Exception:
            fstate["_focus_request_seq"] = 1
        try:
            fstate["_focused_door_id"] = str(did)
        except Exception:
            pass
        st.rerun()

    if c_next.button("Next", use_container_width=True, key=f"door_next_btn_{key_suffix}"):
        try:
            next_id = door_ids[(int(selected_idx) + 1) % max(1, len(door_ids))]
            st.session_state[f"door_click_sink_{file_id}"] = str(next_id)
        except Exception:
            pass
        _mark_nav_intent(str(file_id))
        st.rerun()

    # (Door id display removed; it adds clutter.)

    # Use the *current working* label state so this reflects draft changes in edit mode.
    working = _get_working_label_state(fstate)
    try:
        is_confirmed = did in flatten_confirmed_ids(working.get("confirmed_by_type", {}))
    except Exception:
        is_confirmed = False
    confirmed_html = '<span class="door_detector-door-meta-confirmed">Confirmed</span>' if is_confirmed else ""

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
    {confirmed_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Typed label control (what the reviewer says this door *is*).
    # This is separate from the model-predicted door type (displayed above).
    label_type_key = f"label_type_{file_id}"
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

    # UX: default "Label as" to the *currently selected door's* detected type.
    # The widget key is per-file, so we explicitly reset it when selection changes.
    if str(fstate.get("_label_type_last_door_id") or "") != str(did):
        st.session_state[label_type_key] = default_label_type
        fstate["_label_type_last_door_id"] = str(did)

    if label_type_key not in st.session_state:
        st.session_state[label_type_key] = default_label_type
    if str(st.session_state.get(label_type_key) or "") not in DOOR_TYPES:
        st.session_state[label_type_key] = default_label_type

    st.selectbox(
        "Label as",
        list(DOOR_TYPES),
        key=label_type_key,
        format_func=lambda t: str(t).capitalize(),
    )

    # Actions
    is_editing = bool(fstate.get("edit_mode"))
    c1, c2, c3 = st.columns(3)
    # Button labels include door type to make the feedback intent unambiguous.
    ui_label_type = normalize_door_type(st.session_state.get(label_type_key), default=default_label_type)
    detected_type = normalize_door_type(selected_door.get("type"), default="swing")
    confirm_label = f"Confirm {str(ui_label_type).capitalize()} door"
    reject_label = f"Not a {str(detected_type).capitalize()} door"

    def _emit_label_action_debug(action: str, *, after_mutation: bool = False) -> None:
        """Emit a compact snapshot for confirm/deny bugs (id mismatch / stale overlays)."""
        try:
            cbt = working.get("confirmed_by_type", {}) if isinstance(working, dict) else {}
            rbt = working.get("rejected_by_type", {}) if isinstance(working, dict) else {}
            confirmed_ids = flatten_confirmed_ids(cbt) if isinstance(cbt, dict) else set()
            rejected_ids = flatten_rejected_ids(rbt) if isinstance(rbt, dict) else set()
            deleted_ids = set(working.get("deleted_ids", set()) or []) if isinstance(working, dict) else set()
        except Exception:
            confirmed_ids, rejected_ids, deleted_ids = set(), set(), set()

        # Useful to detect "action applied to different door than highlighted":
        # both `fstate["selected_door_id"]` and the event-like sink values.
        click_sink_key = f"door_click_sink_{file_id}"
        selected_sink_key = f"selected_door_sink_{file_id}"

        ui_event_log(
            "label_action",
            {
                "phase": "after_mutation" if after_mutation else "clicked",
                "action": str(action),
                "file_id": str(file_id),
                "door_id": str(did),
                "selected_door_id_fstate": str(fstate.get("selected_door_id") or ""),
                "door_click_sink": str(st.session_state.get(click_sink_key) or ""),
                "selected_door_sink": str(st.session_state.get(selected_sink_key) or ""),
                "door_filter": str(st.session_state.get(door_filter_state_key) or ""),
                "is_editing": bool(is_editing),
                "selected_idx": int(selected_idx),
                "visible_len": int(len(door_ids)),
                "ui_label_type": str(ui_label_type),
                "detected_type": str(detected_type),
                "bbox_xyxy": selected_door.get("bbox_xyxy"),
                "bbox_pdf_xyxy": selected_door.get("bbox_pdf_xyxy"),
                "confirmed_len": int(len(confirmed_ids)),
                "rejected_len": int(len(rejected_ids)),
                "deleted_len": int(len(deleted_ids)),
                "is_confirmed_now": bool(str(did) in set(map(str, confirmed_ids))),
                "is_deleted_now": bool(str(did) in set(map(str, deleted_ids))),
            },
        )

        # Persist the last action so app.py can report what it computed on the next run.
        try:
            st.session_state[f"_door_detector_last_label_action_{file_id}"] = json.dumps(
                {
                    "action": str(action),
                    "door_id": str(did),
                    "ui_label_type": str(ui_label_type),
                    "detected_type": str(detected_type),
                    "door_filter": str(st.session_state.get(door_filter_state_key) or ""),
                    "ts": time.time(),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        except Exception:
            pass

    if c1.button(confirm_label, use_container_width=True, type="primary", key=f"confirm_btn_{key_suffix}"):
        # Record expected filter across this rerun (keeps list stable across reruns).
        try:
            st.session_state[f"_door_filter_expected_{file_id}"] = str(st.session_state.get(door_filter_state_key) or "")
        except Exception:
            pass
        _emit_label_action_debug("confirm", after_mutation=False)
        # Ensure confirmed_by_type exists.
        try:
            cbt = working.get("confirmed_by_type")
            if not isinstance(cbt, dict):
                cbt = {t: set() for t in DOOR_TYPES}
                working["confirmed_by_type"] = cbt
        except Exception:
            cbt = {t: set() for t in DOOR_TYPES}
            working["confirmed_by_type"] = cbt

        label_type = ui_label_type
        # Candidate can only be confirmed as exactly one type.
        for t in DOOR_TYPES:
            try:
                ids = cbt.get(t)
                if isinstance(ids, set):
                    ids.discard(did)
            except Exception:
                continue
        cbt.setdefault(label_type, set()).add(did)
        # A confirmed candidate must not be marked as rejected/deleted.
        working["deleted_ids"].discard(did)
        try:
            rbt = working.get("rejected_by_type")
            if isinstance(rbt, dict):
                for t in DOOR_TYPES:
                    ids = rbt.get(t)
                    if isinstance(ids, set):
                        ids.discard(did)
        except Exception:
            pass
        # Treat as explicit confirmation (so removing a manual-add record won't unconfirm).
        if is_editing:
            try:
                fstate["_edit_manual_confirmed_ids"].discard(did)
            except Exception:
                pass
        else:
            save_current_labels(str(file_id), file_dir)

        _emit_label_action_debug("confirm", after_mutation=True)

        # Clear any active proposal/match list after committing a label.
        try:
            fstate["_proposal"] = None
            fstate["_last_draw_suggestions"] = None
            st.session_state[f"_draw_suggest_idx_{file_id}"] = 0
        except Exception:
            pass

        # UX: advance to the next door (wrap) after confirming.
        #
        # IMPORTANT: don't rely on mutating the number_input's idx key here; Streamlit can
        # re-apply the widget's prior value at the start of the rerun, clobbering it.
        # Instead, use the click-sink (event-like) path which is consumed once by the
        # selection sync logic and guarantees a selection change.
        try:
            next_id = door_ids[(int(selected_idx) + 1) % max(1, len(door_ids))]
            st.session_state[f"door_click_sink_{file_id}"] = str(next_id)
        except Exception:
            pass
        _mark_nav_intent(str(file_id))
        st.rerun()
    if c2.button(
        reject_label,
        use_container_width=True,
        key=f"reject_btn_{key_suffix}",
        help="Typed negative: mark this candidate as NOT its detected door type (e.g. 'not a double').",
    ):
        try:
            st.session_state[f"_door_filter_expected_{file_id}"] = str(st.session_state.get(door_filter_state_key) or "")
        except Exception:
            pass
        _emit_label_action_debug("reject", after_mutation=False)
        # Ensure rejected_by_type exists.
        try:
            rbt = working.get("rejected_by_type")
            if not isinstance(rbt, dict):
                rbt = {t: set() for t in DOOR_TYPES}
                working["rejected_by_type"] = rbt
        except Exception:
            rbt = {t: set() for t in DOOR_TYPES}
            working["rejected_by_type"] = rbt

        rbt.setdefault(detected_type, set()).add(did)
        # This is not a global "not-a-door" label.
        working["deleted_ids"].discard(did)
        # UX: if the user rejects a double, it is commonly because it's actually two swings.
        # Auto-switch the filter so the revealed swing candidates are immediately visible.
        if detected_type == "double":
            try:
                st.session_state[door_filter_state_key] = "swing"
                st.session_state[door_filter_widget_key] = "swing"
                try:
                    fstate["_door_filter"] = "swing"
                except Exception:
                    pass
            except Exception:
                pass
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

        _emit_label_action_debug("reject", after_mutation=True)

        # Clear any active proposal/match list after committing a label.
        try:
            fstate["_proposal"] = None
            fstate["_last_draw_suggestions"] = None
            st.session_state[f"_draw_suggest_idx_{file_id}"] = 0
        except Exception:
            pass

        # UX: selection will advance automatically. Treat this as explicit navigation so
        # auto-focus (if enabled) will focus the next selected door.
        _mark_nav_intent(str(file_id))
        # Advance selection using the same click-sink mechanism as Confirm. Setting
        # selected_door_id=None is not enough to express "next" (sync falls back to first).
        try:
            if len(door_ids) > 1:
                next_id = door_ids[(int(selected_idx) + 1) % max(1, len(door_ids))]
                if str(next_id) and str(next_id) != str(did):
                    st.session_state[f"door_click_sink_{file_id}"] = str(next_id)
        except Exception:
            pass
        fstate["selected_door_id"] = None  # Allow sync to pick click-sink (or first).
        st.rerun()

    # Optional global negative (rare): truly "not a door at all".
    if c3.button(
        "Not a door at all",
        use_container_width=True,
        type="secondary",
        key=f"not_door_btn_{key_suffix}",
        help="Global negative: use only when the highlighted item is truly not any door type (hide it).",
    ):
        try:
            st.session_state[f"_door_filter_expected_{file_id}"] = str(st.session_state.get(door_filter_state_key) or "")
        except Exception:
            pass
        _emit_label_action_debug("not_a_door_at_all", after_mutation=False)
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
        try:
            rbt = working.get("rejected_by_type")
            if isinstance(rbt, dict):
                for t in DOOR_TYPES:
                    ids = rbt.get(t)
                    if isinstance(ids, set):
                        ids.discard(did)
        except Exception:
            pass
        if is_editing:
            try:
                fstate["_edit_manual_confirmed_ids"].discard(did)
            except Exception:
                pass
            working["manual_additions"] = [
                r
                for r in list(working.get("manual_additions", []))
                if str(r.get("snapped_candidate_id") or "") != str(did)
            ]
        else:
            save_current_labels(str(file_id), file_dir)

        _emit_label_action_debug("not_a_door_at_all", after_mutation=True)

        # Clear any active proposal/match list after committing a label.
        try:
            fstate["_proposal"] = None
            fstate["_last_draw_suggestions"] = None
            st.session_state[f"_draw_suggest_idx_{file_id}"] = 0
        except Exception:
            pass

        # UX: selection will advance automatically. Treat this as explicit navigation so
        # auto-focus (if enabled) will focus the next selected door.
        _mark_nav_intent(str(file_id))
        # Advance selection using click-sink. (selected_door_id=None alone falls back to first.)
        try:
            if len(door_ids) > 1:
                next_id = door_ids[(int(selected_idx) + 1) % max(1, len(door_ids))]
                if str(next_id) and str(next_id) != str(did):
                    st.session_state[f"door_click_sink_{file_id}"] = str(next_id)
        except Exception:
            pass
        fstate["selected_door_id"] = None
        st.rerun()

    # Tips (below the action buttons).
    st.divider()
    st.markdown("**Tips**")
    st.caption("Shift+drag to propose selection · Drag to pan · Scroll to zoom")

    # Stats display removed (too noisy); all actions apply immediately.
