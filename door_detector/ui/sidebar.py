"""Sidebar components for the Streamlit UI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st

from door_detector.library import Library
from door_detector.reweight_fit import fit_reweighter
from door_detector.ui.labels import (
    coerce_confirmed_by_type,
    coerce_id_set,
    coerce_rejected_by_type,
    flatten_confirmed_ids,
    flatten_rejected_ids,
)


def _retrain_state_path(models_dir: Path) -> Path:
    return models_dir / "retrain_state_v1.json"


def _load_last_trained_total_samples(*, models_dir: Path) -> int:
    """Return last trained total sample count (best effort)."""
    p = _retrain_state_path(models_dir)
    try:
        if not p.exists():
            return 0
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return 0
        return max(0, int(obj.get("total_samples") or 0))
    except Exception:
        return 0


def _save_last_trained_total_samples(*, models_dir: Path, total_samples: int) -> None:
    """Persist last trained total sample count (best effort)."""
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_samples": int(max(0, int(total_samples))),
        }
        _retrain_state_path(models_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        return


def _count_overrides_in_labels_data(labels_data: Dict[str, Any]) -> int:
    """Count unique labeled candidate ids (confirmed/rejected/deleted) in one labels.json."""
    try:
        cbt = coerce_confirmed_by_type(labels_data.get("confirmed_by_type", {}))
        rbt = coerce_rejected_by_type(labels_data.get("rejected_by_type", {}))
        deleted = coerce_id_set(labels_data.get("deleted_ids", []))
        override_ids = set(deleted) | set(flatten_confirmed_ids(cbt)) | set(flatten_rejected_ids(rbt))
        return int(len(override_ids))
    except Exception:
        return 0


def _label_files_signature(artifacts_root: Path) -> Tuple[Tuple[str, int, int], ...]:
    """Stable signature for invalidating cached global sample counts."""
    out: list[Tuple[str, int, int]] = []
    try:
        root = artifacts_root / "library"
        for p in sorted(root.glob("**/labels.json")):
            try:
                stt = p.stat()
                out.append((str(p), int(stt.st_mtime_ns), int(stt.st_size)))
            except Exception:
                continue
    except Exception:
        return tuple()
    return tuple(out)


def _models_signature(models_dir: Path) -> Dict[str, int]:
    """Best-effort signature of model files to detect updates."""
    out: Dict[str, int] = {}
    try:
        for p in sorted(models_dir.glob("reweighter_*_v1.json")):
            try:
                out[str(p)] = int(p.stat().st_mtime_ns)
            except Exception:
                continue
    except Exception:
        return {}
    return out


@st.cache_data(show_spinner=False)
def _compute_global_sample_counts(sig: Tuple[Tuple[str, int, int], ...]) -> Dict[str, int]:
    total_samples = 0
    num_label_files = 0
    for path_str, _mtime_ns, _size in sig:
        try:
            p = Path(path_str)
            if not p.exists():
                continue
            obj = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                continue
            total_samples += _count_overrides_in_labels_data(obj)
            num_label_files += 1
        except Exception:
            continue
    return {"total_samples": int(total_samples), "num_label_files": int(num_label_files)}


def sidebar_library(lib: Library) -> None:
    st.sidebar.title("Library")

    # Manage actions removed (Import existing artifacts / Clear library).
    # If a prior run left the confirm flag around, clear it so it doesn't linger.
    try:
        st.session_state.pop("confirm_clear_library", None)
    except Exception:
        pass

    # Search and Add Area
    if not st.session_state.search_visible:
        col_search, col_add = st.sidebar.columns(2)
        with col_search:
            if st.button("Search", key="open_search_btn", help="Open search", use_container_width=True):
                st.session_state.search_visible = True
                st.rerun()
        with col_add:
            upload_key = f"upload_pdf_{int(st.session_state.get('upload_widget_seq') or 0)}"
            uploaded_file = st.file_uploader(
                "Upload",
                type=["pdf"],
                label_visibility="collapsed",
                key=upload_key,
            )
            if uploaded_file:
                file_id = lib.add_file(uploaded_file.name, uploaded_file.getvalue())
                # Auto-select the newly added file, and reset the uploader so we
                # don't re-add it on the next rerun.
                st.session_state.selected_file_id = file_id
                st.session_state.upload_widget_seq = int(st.session_state.get("upload_widget_seq") or 0) + 1
    else:
        col_input, col_close = st.sidebar.columns([5, 1])
        with col_input:
            search_val = st.text_input(
                "Search",
                value=st.session_state.search_query,
                label_visibility="collapsed",
                key="search_input_widget",
            )
            if search_val != st.session_state.search_query:
                st.session_state.search_query = search_val
        with col_close:
            if st.button("X", key="close_search_btn", help="Clear search"):
                st.session_state.search_query = ""
                st.session_state.search_visible = False
                st.rerun()

    st.sidebar.divider()

    items = lib.get_items()
    if st.session_state.search_query:
        items = [i for i in items if st.session_state.search_query.lower() in i["original_name"].lower()]

    # Scrollable file list region (clips above training section).
    # NOTE: We intentionally avoid trying to "wrap" Streamlit widgets with raw HTML
    # tags across multiple calls; that can break the sidebar DOM and render blank.
    list_box = st.sidebar.container(height=420, border=False)
    with list_box:
        if not items:
            st.info("No files in library.")
        else:
            for item in items:
                is_selected = st.session_state.get("selected_file_id") == item["id"]
                label = item["original_name"]

                if st.button(
                    label,
                    key=f"sel_{item['id']}",
                    help=item["original_name"],
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_file_id = item["id"]

    # --- Training section (global across the whole library) ---
    st.sidebar.divider()

    artifacts_root = Path(lib.root_dir)
    sig = _label_files_signature(artifacts_root)
    counts = _compute_global_sample_counts(sig)
    total_samples = int(counts.get("total_samples") or 0)
    num_label_files = int(counts.get("num_label_files") or 0)

    models_dir = Path("models")
    last_trained_total = _load_last_trained_total_samples(models_dir=models_dir)
    untrained = max(0, int(total_samples) - int(last_trained_total))

    st.sidebar.markdown("**Model training**")
    st.sidebar.caption(f"Untrained samples: **{untrained}**")
    st.sidebar.caption(f"Total samples: **{total_samples}** (across {num_label_files} PDF(s))")

    train_disabled = total_samples <= 0
    if st.sidebar.button(
        "Train Model",
        key="train_model_sidebar_btn",
        use_container_width=True,
        disabled=train_disabled,
        help=(
            "Fits per-type reweighters from all saved labels in the library."
            if untrained > 0
            else "Fits per-type reweighters from all saved labels in the library (no new samples since last retrain)."
        ),
    ):
        with st.sidebar:
            with st.spinner("Training..."):
                before = _models_signature(models_dir)
                fit_reweighter(artifacts_root, models_dir)
                after = _models_signature(models_dir)
                updated = False
                try:
                    keys = set(before.keys()) | set(after.keys())
                    updated = any(int(after.get(k, -1)) != int(before.get(k, -1)) for k in keys)
                except Exception:
                    updated = False

                if updated:
                    _save_last_trained_total_samples(models_dir=models_dir, total_samples=total_samples)
                    st.success("Model updated!")
                else:
                    st.warning("Training ran, but no new model was written (need more labeled samples).")
        st.cache_data.clear()
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.rerun()

