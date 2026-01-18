"""Streamlit app entrypoint (composition + orchestration)."""

from __future__ import annotations

import html
import json
import logging
import math
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from door_detector.doors.geometry import compute_iou
from door_detector.library import Library
from door_detector.pdf.affine import apply_affine_bbox_xyxy as _apply_affine_bbox_xyxy
from door_detector.step1_pipeline import process_pdf
from door_detector.signatures import compute_step1_signature
from door_detector.step2_pipeline import run_step2

from door_detector.ui import assets
from door_detector.ui.artifacts_io import get_full_page_dims, get_or_create_page_preview, load_file_artifacts
from door_detector.doors.types import normalize_door_type
from door_detector.ui.labels import (
    coerce_confirmed_by_type,
    coerce_id_set,
    flatten_confirmed_ids,
    enter_edit_mode as _enter_edit_mode,
    get_working_label_state as _get_working_label_state,
)
from door_detector.ui.review_panel import main_viewer_controls, right_panel_review, _sync_selected_door_for_run
from door_detector.ui.sidebar import sidebar_library
from door_detector.ui.viewer import _normalize_bbox_xyxy, main_viewer_canvas
from door_detector.doors.detect import debug_explain_unmatched_box


logger = logging.getLogger("door_detector.review_app")


def _debug_log(msg: str, *args: Any) -> None:
    """Optional debug logging to the server console (disabled in the UI)."""
    try:
        if st.session_state.get("debug_perf"):
            logger.info(msg, *args)
    except Exception:
        return


def init_file_state(file_id: str, doors_data: Dict, labels_data: Dict) -> None:
    if "files" not in st.session_state:
        st.session_state.files = {}

    if file_id not in st.session_state.files:
        st.session_state.files[file_id] = {
            "confirmed_by_type": coerce_confirmed_by_type(labels_data.get("confirmed_by_type", {})),
            "deleted_ids": coerce_id_set(labels_data.get("deleted_ids", [])),
            "manual_additions": list(labels_data.get("manual_additions", [])),
            "unmatched_manual_boxes": list(labels_data.get("unmatched_manual_boxes", [])),
            "selected_door_id": None,
            "viewer_display_mode": "Highlight All",
            "auto_focus": True,
            "edit_mode": False,
            "_edit_baseline": None,
            "_edit_draft": None,
            "_edit_manual_confirmed_ids": set(),
            "_last_draw_event_id": None,
            "_last_viewer_event_id": None,
            "_last_unmatched_debug": None,
            "_focus_seq": 0,
            "_focus_last_id": None,
            "_last_clicked_door_id": None,
        }


def _clamp_bbox_xyxy(bbox_xyxy: List[float], *, w: Optional[int], h: Optional[int]) -> List[float]:
    nb = _normalize_bbox_xyxy(bbox_xyxy)
    if nb is None:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = nb
    if w is not None and w > 0:
        x0 = max(0.0, min(float(w), x0))
        x1 = max(0.0, min(float(w), x1))
    if h is not None and h > 0:
        y0 = max(0.0, min(float(h), y0))
        y1 = max(0.0, min(float(h), y1))
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _snap_to_candidate(
    drawn_bbox_xyxy: List[float],
    *,
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Return (best_candidate, iou). Candidate bboxes are assumed full-res pixels."""
    nb = _normalize_bbox_xyxy(drawn_bbox_xyxy)
    if nb is None:
        return None, 0.0
    x0, y0, x1, y1 = nb
    drawn = [x0, y0, x1, y1]

    # Only consider candidates that overlap the selection box (IoU>0).
    best_iou = -1.0
    best_by_iou: Optional[Dict[str, Any]] = None
    best_inter = -1.0
    best_by_inter: Optional[Dict[str, Any]] = None
    any_overlap = False

    for cand in candidates:
        cid = cand.get("id")
        cb = _normalize_bbox_xyxy(cand.get("bbox_xyxy"))
        if cid is None or cb is None:
            continue
        cx0, cy0, cx1, cy1 = cb
        cbox = [cx0, cy0, cx1, cy1]
        iou = float(compute_iou(drawn, cbox))
        if iou <= 0.0:
            continue
        any_overlap = True
        if iou > best_iou:
            best_iou = iou
            best_by_iou = cand
        # Track maximum intersection area as fallback.
        inter_x0 = max(drawn[0], cbox[0])
        inter_y0 = max(drawn[1], cbox[1])
        inter_x1 = min(drawn[2], cbox[2])
        inter_y1 = min(drawn[3], cbox[3])
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter = inter_w * inter_h
        if inter > best_inter:
            best_inter = inter
            best_by_inter = cand

    if not any_overlap:
        return None, 0.0

    # Primary: max IoU.
    MIN_SNAP_IOU = 0.02
    if best_by_iou is not None and best_iou >= MIN_SNAP_IOU:
        return best_by_iou, max(0.0, float(best_iou))

    # Fallback: max intersection area among overlapping candidates.
    if best_by_inter is not None and best_inter > 0.0:
        cb = _normalize_bbox_xyxy(best_by_inter.get("bbox_xyxy"))
        if cb is not None:
            return best_by_inter, max(0.0, float(compute_iou(drawn, [cb[0], cb[1], cb[2], cb[3]])))

    return None, 0.0


def _debug_unmatched_region(*, file_dir: Path, drawn_bbox_full_xyxy: List[float], config_path: str) -> Optional[str]:
    """Debug aid for unmatched Shift+drag boxes.

    Returns a JSON string; the viewer prints it to the browser console.
    """
    try:
        primitives_path = file_dir / "primitives.json"
        if not primitives_path.exists():
            return json.dumps(
                {
                    "kind": "unmatched_box_debug_v1",
                    "error": "missing_primitives.json",
                    "file_dir": str(file_dir),
                    "bbox_full_xyxy": drawn_bbox_full_xyxy,
                },
                separators=(",", ":"),
            )
        primitives = json.loads(primitives_path.read_bytes())
    except Exception as e:
        return json.dumps(
            {
                "kind": "unmatched_box_debug_v1",
                "error": "failed_to_load_primitives",
                "file_dir": str(file_dir),
                "bbox_full_xyxy": drawn_bbox_full_xyxy,
                "exception": str(e),
            },
            separators=(",", ":"),
        )

    try:
        cfg = json.loads(Path(config_path).read_bytes())
    except Exception as e:
        cfg = {"error": f"failed_to_load_config: {e}"}

    try:
        rep = debug_explain_unmatched_box(primitives=primitives, bbox_full_xyxy=drawn_bbox_full_xyxy, config=cfg)
        rep["file_dir"] = str(file_dir)
        rep["config_path"] = str(config_path)
        return json.dumps(rep, separators=(",", ":"))
    except Exception as e:
        return json.dumps(
            {
                "kind": "unmatched_box_debug_v1",
                "error": "failed_to_compute_debug_report",
                "file_dir": str(file_dir),
                "bbox_full_xyxy": drawn_bbox_full_xyxy,
                "exception": str(e),
            },
            separators=(",", ":"),
        )


def _process_draw_event_if_any(
    *,
    file_id: str,
    file_dir: Path,
    fstate: Dict[str, Any],
    doors_data: Dict[str, Any],
    preview_spec: Optional[Dict[str, Any]],
    full_dims: Optional[Tuple[int, int]],
    config_path: str,
) -> None:
    """Consume a Shift+drag draw event from the iframe (if present)."""
    if not preview_spec:
        return
    draw_key = f"draw_event_sink_{file_id}"
    raw = st.session_state.get(draw_key) or ""

    # Fallback: some Streamlit builds may not rerun on the hidden draw_event sink change.
    # In that case we also stuff the payload into the existing click sink as a sentinel.
    if not raw:
        click_key = f"door_click_sink_{file_id}"
        click_raw = st.session_state.get(click_key) or ""
        if isinstance(click_raw, str) and click_raw.startswith("__draw_event__"):
            try:
                encoded = click_raw[len("__draw_event__") :]
                # payload is URI-encoded JSON
                raw = urllib.parse.unquote(encoded)
                _debug_log("draw_event fallback from click sink file_id=%s", str(file_id))
            except Exception:
                raw = ""
    if not raw:
        return
    try:
        evt = json.loads(raw)
    except Exception:
        return
    if not isinstance(evt, dict):
        return
    if evt.get("event") != "draw_rect":
        return
    event_id = evt.get("event_id")
    bbox = evt.get("bbox_xyxy")
    bbox_pdf = evt.get("bbox_pdf_xyxy")
    snapped_candidate_id = evt.get("snapped_candidate_id")
    if not event_id:
        return
    if str(event_id) == str(fstate.get("_last_draw_event_id")):
        return
    fstate["_last_draw_event_id"] = str(event_id)

    # JS only emits in edit mode, but guard anyway.
    if not bool(fstate.get("edit_mode")):
        return

    _enter_edit_mode(fstate)
    draft = _get_working_label_state(fstate)

    drawn_full = None
    # New path: PDF.js emits bbox_pdf_xyxy in PDF coords; convert PDF → pixel using Step1 transform.
    if isinstance(bbox_pdf, list) and len(bbox_pdf) == 4:
        try:
            tpath = file_dir / "transform.json"
            tob = json.loads(tpath.read_text()) if tpath.exists() else {}
            m = tob.get("pdf_to_pix_affine") if isinstance(tob, dict) else None
            if isinstance(m, list) and len(m) == 6:
                drawn_full = _apply_affine_bbox_xyxy(m, bbox_pdf)
        except Exception:
            drawn_full = None

    # Legacy path: iframe emits preview-space pixels.
    if drawn_full is None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return
        scale_full_to_preview = float(preview_spec.get("scale", 1.0) or 1.0)
        if not (scale_full_to_preview > 0):
            return
        try:
            x0p, y0p, x1p, y1p = [float(v) for v in bbox]
        except Exception:
            return
        drawn_full = [
            x0p / scale_full_to_preview,
            y0p / scale_full_to_preview,
            x1p / scale_full_to_preview,
            y1p / scale_full_to_preview,
        ]

    full_w = full_dims[0] if full_dims else None
    full_h = full_dims[1] if full_dims else None
    drawn_full = _clamp_bbox_xyxy([float(v) for v in drawn_full], w=full_w, h=full_h)

    candidates = list(doors_data.get("candidates", doors_data.get("doors", [])) or [])
    best = None
    iou = 0.0

    if snapped_candidate_id:
        sid = str(snapped_candidate_id)
        best = next((c for c in candidates if str(c.get("id") or "") == sid), None)
        if best is not None:
            cb = _normalize_bbox_xyxy(best.get("bbox_xyxy"))
            if cb is not None:
                iou = float(compute_iou(drawn_full, [cb[0], cb[1], cb[2], cb[3]]))
                # Reject client-proposed snaps that don't actually overlap the selection.
                if iou <= 0.0:
                    best = None
    if best is None:
        best, iou = _snap_to_candidate(drawn_full, candidates=candidates)

    if best is not None and best.get("id") is not None:
        cid = str(best["id"])
        snapped_full = _normalize_bbox_xyxy(best.get("bbox_xyxy")) or _normalize_bbox_xyxy(drawn_full) or (0.0, 0.0, 0.0, 0.0)
        label_type = normalize_door_type(best.get("type"), default="swing")
        rec = {
            "drawn_bbox_xyxy": drawn_full,
            "snapped_candidate_id": cid,
            "iou": float(iou),
            "snapped_bbox_xyxy": [float(snapped_full[0]), float(snapped_full[1]), float(snapped_full[2]), float(snapped_full[3])],
            "label_type": label_type,
        }
        draft["manual_additions"].append(rec)
        # Typed confirmation: ensure the id belongs to exactly one confirmed bucket.
        try:
            cbt = draft.get("confirmed_by_type")
            if not isinstance(cbt, dict):
                cbt = {}
                draft["confirmed_by_type"] = cbt
            for t, ids in list(cbt.items()):
                if isinstance(ids, set):
                    ids.discard(cid)
            cbt.setdefault(label_type, set()).add(cid)
        except Exception:
            pass
        draft["deleted_ids"].discard(cid)
        try:
            fstate["_edit_manual_confirmed_ids"].add(cid)
        except Exception:
            pass
        # Make the snapped door the current selection.
        try:
            fstate["selected_door_id"] = cid
            st.session_state[f"jump_{file_id}"] = cid
            # Suppress focus bump when selection change originates from draw/snap.
            fstate["_focus_last_id"] = cid
        except Exception:
            pass
    else:
        # Persist the unmatched box (in the edit draft only) so it survives reruns.
        # The viewer will render only the *current edit session* unmatched boxes.
        try:
            draft.setdefault("unmatched_manual_boxes", [])
            draft["unmatched_manual_boxes"].append(
                {
                    "bbox_xyxy": drawn_full,
                    "note": "unmatched (no overlapping candidate)",
                    "event_id": str(event_id),
                }
            )
        except Exception:
            pass
        fstate["_last_unmatched_debug"] = _debug_unmatched_region(
            file_dir=file_dir,
            drawn_bbox_full_xyxy=drawn_full,
            config_path=str(config_path),
        )

    # Consume the sink value so it doesn't grow / re-trigger on reconnects.
    try:
        st.session_state[draw_key] = ""
    except Exception:
        pass

    # Also clear the click sink sentinel if it was used.
    try:
        click_key = f"door_click_sink_{file_id}"
        if isinstance(st.session_state.get(click_key), str) and str(st.session_state.get(click_key)).startswith("__draw_event__"):
            st.session_state[click_key] = ""
    except Exception:
        pass


def run_pipeline(file_id: str, file_dir: Path, config_path: str) -> None:
    lib = st.session_state.library
    lib.update_status(file_id, "processing")
    try:
        primitives_path = file_dir / "primitives.json"
        meta_path = file_dir / "meta.json"
        image_path = file_dir / "page.png"
        pdf_path = file_dir / "source.pdf"

        has_step1_artifacts = primitives_path.exists() and meta_path.exists() and image_path.exists()
        can_run_step1 = pdf_path.exists()

        step1_dpi = 400
        step1_page_index = 0

        if can_run_step1:
            should_run_step1 = True
            if has_step1_artifacts:
                try:
                    stored_meta = json.loads(meta_path.read_text())
                except Exception:
                    stored_meta = {}

                stored_sig = (stored_meta.get("step1") or {}).get("signature")
                try:
                    current_sig = compute_step1_signature(
                        pdf_path=pdf_path, dpi=step1_dpi, page_index=step1_page_index
                    ).get("signature")
                except Exception:
                    current_sig = None

                if stored_sig and current_sig and stored_sig == current_sig:
                    should_run_step1 = False
                    logger.info("Skipping Step 1 for %s (signature match)", str(file_dir))

            if should_run_step1:
                process_pdf(pdf_path, file_dir, dpi=step1_dpi, page_index=step1_page_index)
        elif not has_step1_artifacts:
            raise FileNotFoundError(
                f"Missing Step 1 artifacts in {file_dir} and no source.pdf found; cannot rerun Step 1."
            )

        run_step2(file_dir, Path(config_path))
        lib.update_status(file_id, "done")
        st.cache_data.clear()
        try:
            st.cache_resource.clear()
        except Exception:
            pass
    except Exception as e:
        lib.update_status(file_id, "error", str(e))


def main() -> None:
    # Page config must be set before any other Streamlit calls.
    st.set_page_config(page_title="Door Detector: Door Detection & Review", layout="wide", initial_sidebar_state="expanded")

    assets.inject_global_styles()

    # --- Initialize Library + session state ---
    if "library" not in st.session_state:
        st.session_state.library = Library(Path("artifacts"))

    if "search_visible" not in st.session_state:
        st.session_state.search_visible = False
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "upload_widget_seq" not in st.session_state:
        st.session_state.upload_widget_seq = 0

    if "door_detector_pipeline_task" not in st.session_state:
        # { file_id, file_dir, config_path, label, _started }
        st.session_state.door_detector_pipeline_task = None

    # Debug logs toggle removed from the UI; ensure it stays disabled.
    try:
        st.session_state.pop("debug_perf", None)
    except Exception:
        pass

    lib = st.session_state.library

    sidebar_library(lib)

    col_app = st.container()
    with col_app:
        if "selected_file_id" in st.session_state and st.session_state.selected_file_id:
            items = lib.get_items()
            selected_item = next((i for i in items if i["id"] == st.session_state.selected_file_id), None)

            if selected_item:
                file_id = selected_item["id"]
                file_dir = Path(selected_item["path"])

                # Render title + a stable two-column layout. Use placeholders so the
                # viewer is *replaced* by the loader (instead of leaving the old viewer
                # output below it on reruns).
                title = html.escape(str(selected_item.get("original_name", "")))
                st.markdown(f"<div class='door_detector-pdf-title'><h3>{title}</h3></div>", unsafe_allow_html=True)

                col_main, col_review = st.columns([2, 1])
                with col_main:
                    # Always keep the sidebar auto-open logic mounted.
                    components.html(assets.sidebar_autopen_component_html(), height=0, scrolling=False)
                    viewer_slot = st.empty()
                with col_review:
                    review_slot = st.empty()

                # If a pipeline run is queued for this file, replace the viewer with a loader,
                # then run the pipeline synchronously (so the title stays constant).
                task = st.session_state.get("door_detector_pipeline_task")
                if task and task.get("file_id") == str(file_id) and not bool(task.get("_started")):
                    try:
                        task["_started"] = True
                        st.session_state.door_detector_pipeline_task = task
                    except Exception:
                        pass
                    try:
                        # IMPORTANT: explicitly clear prior viewer output so it doesn't
                        # remain stacked underneath the loader (Streamlit can otherwise
                        # leave a previous iframe render in place).
                        try:
                            viewer_slot.empty()
                            review_slot.empty()
                        except Exception:
                            pass
                        # Hide any stale Streamlit component iframes while analyzing.
                        # Some Streamlit builds can leave a previous components.html iframe
                        # attached below the loader; this CSS prevents that UI artifact.
                        viewer_slot.markdown(
                            "<div id='door_detector-analyzing-flag' style='display:none;'></div>"
                            "<style>"
                            "body:has(#door_detector-analyzing-flag) section.main iframe,"
                            "body:has(#door_detector-analyzing-flag) section.stMain iframe{"
                            "display:none !important;"
                            "}"
                            "</style>"
                            "<div class='door_detector-viewer-loading' style='height: 650px;'><div class='door_detector-viewer-loading-inner'><div class='door_detector-spinner'></div><div class='door_detector-viewer-loading-title'>Analyzing…</div><div class='door_detector-viewer-loading-sub'>Updating detections and refreshing results. This can take a moment.</div></div></div>",
                            unsafe_allow_html=True,
                        )
                        review_slot.info("Analysis is running…")

                        run_pipeline(
                            str(file_id),
                            Path(str(task.get("file_dir") or str(file_dir))),
                            str(task.get("config_path") or "configs/door_rules.json"),
                        )
                    finally:
                        st.session_state.door_detector_pipeline_task = None
                    st.rerun()

                try:
                    doors_data, labels_data, meta_data = load_file_artifacts(str(file_dir))
                except Exception as e:
                    st.error(str(e))
                    st.stop()
                init_file_state(file_id, doors_data, labels_data)
                fstate = st.session_state.files[file_id]

                full_dims = get_full_page_dims(meta_data)
                page_png_path = file_dir / "page.png"
                try:
                    page_png_mtime_ns = page_png_path.stat().st_mtime_ns
                except Exception:
                    page_png_mtime_ns = 0
                preview_spec = get_or_create_page_preview(
                    str(file_dir),
                    full_width=full_dims[0] if full_dims else None,
                    full_height=full_dims[1] if full_dims else None,
                    page_png_mtime_ns=page_png_mtime_ns,
                )

                # --- Consume PDF.js component events early (before snapping + selection sync) ---
                # The PDF.js viewer is a Streamlit custom component keyed by `pdfjs_viewer_{file_id}`.
                # Its value is available in session_state at the start of a rerun after user actions.
                try:
                    viewer_key = f"pdfjs_viewer_{file_id}"
                    evt = st.session_state.get(viewer_key)
                    if isinstance(evt, dict):
                        evt_id = str(evt.get("event_id") or "")
                        if evt_id and evt_id != str(fstate.get("_last_viewer_event_id") or ""):
                            fstate["_last_viewer_event_id"] = evt_id
                            et = str(evt.get("type") or "")
                            click_sink_label = f"door_click_sink_{file_id}"
                            if et == "door_click":
                                did = evt.get("door_id")
                                if did not in (None, ""):
                                    st.session_state[click_sink_label] = str(did)
                            elif et == "draw_rect":
                                st.session_state[f"draw_event_sink_{file_id}"] = json.dumps(
                                    {
                                        "event": "draw_rect",
                                        "event_id": evt.get("event_id"),
                                        "bbox_pdf_xyxy": evt.get("bbox_pdf_xyxy"),
                                        "snapped_candidate_id": evt.get("snapped_candidate_id"),
                                        "iou": evt.get("iou"),
                                        "snapped_bbox_pdf_xyxy": evt.get("snapped_bbox_pdf_xyxy"),
                                        "ts": evt.get("ts"),
                                    },
                                    separators=(",", ":"),
                                )
                except Exception:
                    pass

                # Consume any Shift+drag events before computing the visible list/overlay.
                _process_draw_event_if_any(
                    file_id=str(file_id),
                    file_dir=file_dir,
                    fstate=fstate,
                    doors_data=doors_data,
                    preview_spec=preview_spec,
                    full_dims=full_dims,
                    config_path=str(doors_data.get("config_path") or "configs/door_rules.json"),
                )

                # Compute active doors once so the main viewer + right panel stay in perfect sync.
                detections = doors_data.get("doors", [])
                deleted_ids = coerce_id_set(_get_working_label_state(fstate).get("deleted_ids", set()))
                overlay_doors: List[Dict[str, Any]] = [d for d in detections if d.get("id") is not None]

                # Ensure any confirmed snapped candidates are also rendered (even if they were
                # not in the strict output doors list).
                try:
                    working = _get_working_label_state(fstate)
                    extra_ids = flatten_confirmed_ids(working.get("confirmed_by_type", {}))
                    for rec in list(working.get("manual_additions", [])):
                        cid = rec.get("snapped_candidate_id")
                        if cid:
                            extra_ids.add(str(cid))
                    existing_ids = {str(d.get("id")) for d in overlay_doors if d.get("id") is not None}
                    pool = list(doors_data.get("candidates", []) or [])
                    pool_map = {str(c.get("id")): c for c in pool if c.get("id") is not None}
                    for cid in sorted(extra_ids - existing_ids):
                        cand = pool_map.get(str(cid))
                        if not cand:
                            continue
                        bbox = cand.get("bbox_xyxy")
                        if not bbox:
                            continue
                        overlay_doors.append(
                            {
                                "id": str(cid),
                                "type": str(cand.get("type") or "candidate"),
                                "bbox_xyxy": bbox,
                                "confidence": float(cand.get("confidence", 0.0) or 0.0),
                                "features": cand.get("features", {}),
                            }
                        )
                except Exception:
                    pass

                active_doors_all: List[Dict[str, Any]] = [d for d in overlay_doors if str(d.get("id")) not in deleted_ids]

                # Door type filter (affects navigation list + right panel; overlay still includes all).
                door_type_filter_key = f"door_type_filter_{file_id}"
                if door_type_filter_key not in st.session_state:
                    st.session_state[door_type_filter_key] = "All"
                selected_type = str(st.session_state.get(door_type_filter_key) or "All")
                id_to_type = {
                    str(d.get("id")): str(d.get("type") or "").strip()
                    for d in active_doors_all
                    if d.get("id") is not None
                }

                # If the user clicked a door of a different type, switch filter to that type.
                click_sink_label = f"door_click_sink_{file_id}"
                clicked_id = st.session_state.get(click_sink_label)
                if clicked_id is not None:
                    clicked_type = id_to_type.get(str(clicked_id))
                    if clicked_type and selected_type != "All" and clicked_type != selected_type:
                        st.session_state[door_type_filter_key] = clicked_type
                        selected_type = clicked_type

                if selected_type and selected_type != "All":
                    active_doors: List[Dict[str, Any]] = [
                        d for d in active_doors_all if str(d.get("type") or "").strip() == selected_type
                    ]
                else:
                    active_doors = active_doors_all

                # Sync selection state before rendering the viewer.
                all_visible = active_doors.copy()
                all_visible.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
                _sync_selected_door_for_run(file_id=str(file_id), fstate=fstate, all_visible=all_visible)

                with viewer_slot.container():
                    # Scan-mode UX: be explicit that vector-first detection cannot run.
                    try:
                        page_mode = str(meta_data.get("mode") or "").strip().lower()
                    except Exception:
                        page_mode = ""
                    if page_mode == "scan":
                        st.warning(
                            "This PDF page is classified as **scan** (not a vector drawing), so the current "
                            "vector-first door detector can’t process it. Try a vector/hybrid floor plan PDF.",
                            icon="⚠️",
                        )
                    main_viewer_canvas(
                        selected_item,
                        preview_spec=preview_spec,
                        full_dims=full_dims,
                        doors_data=doors_data,
                        fstate=fstate,
                        active_doors=overlay_doors,
                        click_sink_label=click_sink_label,
                    )

                with review_slot.container():
                    main_viewer_controls(
                        selected_item,
                        full_dims=full_dims,
                        doors_data=doors_data,
                        fstate=fstate,
                    )
                    st.divider()
                    right_panel_review(
                        selected_item,
                        preview_spec=preview_spec,
                        doors_data=doors_data,
                        fstate=fstate,
                        active_doors=active_doors,
                        all_active_doors=active_doors_all,
                    )
            else:
                components.html(assets.sidebar_autopen_component_html(), height=0, scrolling=False)
                st.info("Select a file from the library to begin.")
        else:
            components.html(assets.sidebar_autopen_component_html(), height=0, scrolling=False)
            st.info("Select a file from the library to begin.")


