"""Sidebar components for the Streamlit UI."""

from __future__ import annotations

import json
import time
import html
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
            def _select_file(file_id: str) -> None:
                # IMPORTANT: Use an on_click callback so selection updates *before*
                # this script renders widgets on the click-triggered rerun. If we
                # instead mutate session_state inside `if st.button(...):`, the
                # sidebar list will render using the *previous* selection and only
                # show the "selected" styling after a second click.
                st.session_state.selected_file_id = str(file_id)

            for item in items:
                is_selected = st.session_state.get("selected_file_id") == item["id"]
                label = item["original_name"]

                st.button(
                    label,
                    key=f"sel_{item['id']}",
                    help=item["original_name"],
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                    on_click=_select_file,
                    args=(str(item["id"]),),
                )

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
    st.sidebar.caption(f"**{untrained} / {total_samples}** untrained")

    train_disabled = total_samples <= 0
    train_clicked = st.sidebar.button(
        "Train Model",
        key="train_model_sidebar_btn",
        use_container_width=True,
        disabled=train_disabled,
        help=(
            "Fits per-type reweighters from all saved labels in the library."
            if untrained > 0
            else "Fits per-type reweighters from all saved labels in the library (no new samples since last retrain)."
        ),
    )

    # Result output should appear directly under the button, and not persist across reruns.
    result_box = st.sidebar.container()

    if train_clicked:
        with st.sidebar:
            with st.spinner("Training..."):
                before = _models_signature(models_dir)
                progress = st.progress(0, text="Collecting labeled examples…")
                live = st.empty()
                last_lines: list[str] = []

                def _cb(ev: Dict[str, Any]) -> None:
                    try:
                        stage = str(ev.get("stage") or "")
                        t = str(ev.get("door_type") or "")
                        i = int(ev.get("i") or 0)
                        total = int(ev.get("total") or 0)
                        done = i + (1 if stage == "end_type" else 0)
                        pct = int(round(100.0 * float(done) / float(max(1, total))))
                        if stage in ("start_type", "end_type") and t:
                            status = str(ev.get("status") or "")
                            if stage == "start_type":
                                last_lines.append(f"{t.capitalize()}…")
                            else:
                                last_lines.append(f"{t.capitalize()}: {status or 'done'}")
                            # Clip long lines in narrow sidebars.
                            clipped = "\n".join(
                                f"<div style='white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{html.escape(x)}</div>"
                                for x in last_lines[-6:]
                            )
                            live.markdown(clipped, unsafe_allow_html=True)
                        progress.progress(min(100, max(0, pct)), text=f"Training progress: {pct}%")
                    except Exception:
                        return

                report = fit_reweighter(artifacts_root, models_dir, progress_cb=_cb)
                try:
                    progress.empty()
                    live.empty()
                except Exception:
                    pass
                after = _models_signature(models_dir)
                updated = False
                try:
                    keys = set(before.keys()) | set(after.keys())
                    updated = any(int(after.get(k, -1)) != int(before.get(k, -1)) for k in keys)
                except Exception:
                    updated = False

                if updated:
                    _save_last_trained_total_samples(models_dir=models_dir, total_samples=total_samples)

                # Render concise results directly under the button.
                with result_box:
                    by_type = report.get("by_type") if isinstance(report, dict) else None
                    types = report.get("types") if isinstance(report, dict) else None
                    if not (isinstance(by_type, dict) and isinstance(types, list) and types):
                        st.error("Training failed.")
                    else:
                        thresholds = report.get("thresholds") if isinstance(report, dict) else None
                        min_pos = int((thresholds or {}).get("min_pos") or 0) if isinstance(thresholds, dict) else 0
                        min_neg = int((thresholds or {}).get("min_neg") or 0) if isinstance(thresholds, dict) else 0
                        min_samples = int((thresholds or {}).get("min_samples") or 0) if isinstance(thresholds, dict) else 0

                        lines: list[str] = []
                        for t in types:
                            tr = by_type.get(t) if isinstance(by_type.get(t), dict) else {}
                            status = str(tr.get("status") or "unknown")
                            reason = str(tr.get("reason") or "")
                            label = f"{str(t).capitalize()}:"

                            if status == "trained":
                                acc = tr.get("train_accuracy")
                                if isinstance(acc, (int, float)):
                                    lines.append(f"{label} score = {float(acc):.2f}")
                                else:
                                    lines.append(f"{label} score = n/a")
                                continue

                            # Compute "need more" messaging when training didn't write a model.
                            if reason in ("not_enough_samples", "no_labeled_samples"):
                                n = int(tr.get("num_samples") or 0)
                                n_pos = int(tr.get("num_pos") or 0)
                                n_neg = int(tr.get("num_neg") or 0)
                                need_pos = max(0, int(min_pos) - int(n_pos))
                                need_neg = max(0, int(min_neg) - int(n_neg))
                                need_total = max(0, int(min_samples) - int(n))

                                if need_pos > 0 and need_pos >= need_neg:
                                    lines.append(f"{label} Need {need_pos} more positive samples")
                                elif need_neg > 0:
                                    lines.append(f"{label} Need {need_neg} more negative samples")
                                elif need_total > 0:
                                    lines.append(f"{label} Need {need_total} more samples")
                                else:
                                    lines.append(f"{label} Not enough samples")
                                continue

                            if reason == "no_labels":
                                lines.append(f"{label} No labels yet")
                                continue

                            if status == "skipped":
                                lines.append(f"{label} Skipped")
                                continue

                            lines.append(f"{label} {status}")

                        # Single-line per type, clipped if sidebar is narrow.
                        clipped = "\n".join(
                            f"<div style='white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{html.escape(x)}</div>"
                            for x in lines
                        )
                        st.markdown(clipped, unsafe_allow_html=True)
        st.cache_data.clear()
        try:
            st.cache_resource.clear()
        except Exception:
            pass

