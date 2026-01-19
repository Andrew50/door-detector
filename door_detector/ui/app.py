"""Streamlit app entrypoint (composition + orchestration)."""

from __future__ import annotations

import html
import json
import logging
import math
import time
import hashlib
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
from door_detector.ui.artifacts_io import get_full_page_dims, load_file_artifacts
from door_detector.doors.types import normalize_door_type
from door_detector.ui.labels import (
    coerce_confirmed_by_type,
    coerce_rejected_by_type,
    coerce_id_set,
    flatten_confirmed_ids,
    flatten_rejected_ids,
    enter_edit_mode as _enter_edit_mode,
    get_working_label_state as _get_working_label_state,
)
from door_detector.ui.review_panel import main_viewer_controls, right_panel_review, _sync_selected_door_for_run
from door_detector.ui.sidebar import sidebar_library
from door_detector.ui.viewer import _normalize_bbox_xyxy, main_viewer_canvas
from door_detector.doors.detect import debug_explain_unmatched_box
from door_detector.ui.ui_debug import push_breadcrumb, tail_breadcrumbs, warn_once, sample_ids, ui_event_log
from door_detector.perf import enabled as perf_enabled, span as perf_span, log as perf_log


logger = logging.getLogger("door_detector.review_app")


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



def init_file_state(file_id: str, doors_data: Dict, labels_data: Dict) -> None:
    if "files" not in st.session_state:
        st.session_state.files = {}

    if file_id not in st.session_state.files:
        st.session_state.files[file_id] = {
            "confirmed_by_type": coerce_confirmed_by_type(labels_data.get("confirmed_by_type", {})),
            "rejected_by_type": coerce_rejected_by_type(labels_data.get("rejected_by_type", {})),
            "deleted_ids": coerce_id_set(labels_data.get("deleted_ids", [])),
            "manual_candidates": list(labels_data.get("manual_candidates", [])),
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
            "_last_draw_suggestions": None,
            "_proposal": None,
            "_last_viewer_event_id": None,
            "_last_unmatched_debug": None,
            "_focus_seq": 0,
            "_focus_last_id": None,
            # Manual focus requests (via a button in the right panel). This should
            # trigger a one-shot focus in the viewer regardless of auto-focus toggle.
            "_focus_request_seq": 0,
            # One-shot focus requests for proposal overlays (Shift+drag cyan/green boxes).
            "_proposal_focus_seq": 0,
            # Best-effort record of which door we've most recently focused. This is
            # UI-only (used to decide whether to show the Focus button).
            "_focused_door_id": None,
            "_last_clicked_door_id": None,
            # Debug: rolling breadcrumb trail of UI actions affecting selection/highlight.
            "_ui_breadcrumbs": [],
            "_ui_warned_keys": set(),
        }


def _remap_fstate_ids_using_legacy_ids(*, fstate: Dict[str, Any], doors_data: Dict[str, Any]) -> None:
    """Remap in-memory ids (fstate) using `doors.json` candidate `legacy_ids`.

    Why:
    - `labels.json` is remapped on load via `legacy_ids`, but `fstate` persists across reruns.
    - After reanalysis (or after Step 1 changes) candidate ids can change; without remapping,
      the UI can show confirmations that no longer match any current overlay ids (no green).
    """
    try:
        candidates = list(doors_data.get("candidates", doors_data.get("doors", [])) or [])
    except Exception:
        candidates = []

    legacy_to_current: Dict[str, str] = {}
    try:
        for c in candidates:
            if not isinstance(c, dict):
                continue
            cid = c.get("id")
            if cid is None:
                continue
            cid_s = str(cid)
            if not cid_s:
                continue
            legacy_to_current[cid_s] = cid_s
            for lid in list(c.get("legacy_ids") or []):
                if lid is None:
                    continue
                s = str(lid)
                if s:
                    legacy_to_current[s] = cid_s
    except Exception:
        legacy_to_current = {}

    if not legacy_to_current:
        return

    def _map_id(x: Any) -> Any:
        if x is None:
            return x
        s = str(x)
        if not s:
            return x
        return legacy_to_current.get(s, s)

    def _remap_state_dict(state: Dict[str, Any]) -> None:
        # Confirmed/rejected/deleted sets
        try:
            cbt = coerce_confirmed_by_type(state.get("confirmed_by_type", {}))
            state["confirmed_by_type"] = {t: {str(_map_id(i)) for i in ids} for t, ids in cbt.items()}
        except Exception:
            pass
        try:
            rbt = coerce_rejected_by_type(state.get("rejected_by_type", {}))
            state["rejected_by_type"] = {t: {str(_map_id(i)) for i in ids} for t, ids in rbt.items()}
        except Exception:
            pass
        try:
            dids = coerce_id_set(state.get("deleted_ids", set()))
            state["deleted_ids"] = {str(_map_id(i)) for i in dids}
        except Exception:
            pass

        # Manual additions (snapped ids)
        try:
            for rec in list(state.get("manual_additions", []) or []):
                if not isinstance(rec, dict):
                    continue
                sid = rec.get("snapped_candidate_id")
                if sid in (None, ""):
                    continue
                rec["snapped_candidate_id"] = str(_map_id(sid))
        except Exception:
            pass

        # Selection + proposal/cycle ids (best-effort).
        try:
            sel = state.get("selected_door_id")
            if sel not in (None, ""):
                state["selected_door_id"] = str(_map_id(sel))
        except Exception:
            pass
        try:
            cycle = state.get("_cycle_candidate_id")
            if cycle not in (None, ""):
                state["_cycle_candidate_id"] = str(_map_id(cycle))
        except Exception:
            pass
        try:
            prop = state.get("_proposal")
            if isinstance(prop, dict):
                sid = prop.get("snapped_candidate_id")
                if sid not in (None, ""):
                    prop["snapped_candidate_id"] = str(_map_id(sid))
        except Exception:
            pass

    # Remap the committed state and any active edit buffers so the viewer stays consistent.
    _remap_state_dict(fstate)
    try:
        if isinstance(fstate.get("_edit_draft"), dict):
            _remap_state_dict(fstate["_edit_draft"])
        if isinstance(fstate.get("_edit_baseline"), dict):
            _remap_state_dict(fstate["_edit_baseline"])
    except Exception:
        pass


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


def _manual_candidate_id_from_bbox(*, file_id: str, bbox_xyxy: List[float], quant_step_px: float = 1.0) -> str:
    """Stable id for a UI-created manual candidate derived from bbox geometry."""
    nb = _normalize_bbox_xyxy(bbox_xyxy) or [0.0, 0.0, 0.0, 0.0]
    try:
        q = [int(round(float(v) / float(quant_step_px))) for v in nb]
    except Exception:
        q = [int(round(float(v))) for v in nb]
    payload = {"kind": "manual_box_v1", "file_id": str(file_id), "bbox": q}
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "m_" + hashlib.sha1(stable).hexdigest()[:12]


def _preferred_label_type_for_draw(*, file_id: str, default: str = "swing") -> str:
    """Best-effort label type for a draw event based on the UI type filter (if touched)."""
    try:
        touched = bool(st.session_state.get(f"_draw_suggest_type_touched_{file_id}", False))
        chosen = str(st.session_state.get(f"_draw_suggest_type_{file_id}", "") or "")
    except Exception:
        touched, chosen = False, ""
    if touched and chosen and chosen != "All types":
        return normalize_door_type(chosen, default=default)
    return normalize_door_type(default, default="swing")


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
    drawn_area = max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
    # Only consider candidates that overlap the selection box (IoU>0).
    best_iou = -1.0
    best_by_iou: Optional[Dict[str, Any]] = None
    best_iou_inter = -1.0
    best_inter = -1.0
    best_by_inter: Optional[Dict[str, Any]] = None
    best_inter_iou = -1.0
    best_inter_coverage = -1.0
    best_coverage = -1.0
    best_by_coverage: Optional[Dict[str, Any]] = None
    best_coverage_inter = -1.0
    any_overlap = False

    # Conservative anti-symbol heuristics (circles often appear near doors).
    SQUARE_AR_MAX = 1.28
    MIN_SNAP_IOU_FOR_SQUARE = 0.10

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

        # Track maximum intersection area (and capture the intersection for the IoU-best pick).
        inter_x0 = max(drawn[0], cbox[0])
        inter_y0 = max(drawn[1], cbox[1])
        inter_x1 = min(drawn[2], cbox[2])
        inter_y1 = min(drawn[3], cbox[3])
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter = inter_w * inter_h
        c_area = max(0.0, float(cx1 - cx0)) * max(0.0, float(cy1 - cy0))
        coverage = (float(inter) / float(c_area)) if c_area > 0.0 else 0.0

        # Penalize near-square candidates (often circles/symbols) unless match is strong.
        # Exception: `swing_arc` candidates (door arcs) are naturally square-ish; allow them.
        cw = max(0.0, cx1 - cx0)
        ch = max(0.0, cy1 - cy0)
        if cw > 0.0 and ch > 0.0:
            ar = max(cw / ch, ch / cw)
            if ar <= SQUARE_AR_MAX:
                ctype = str(cand.get("type") or "")
                feats = cand.get("features") if isinstance(cand, dict) else None
                arc_only = False
                angle_span = None
                if isinstance(feats, dict):
                    try:
                        arc_only = float(feats.get("arc_only", 0.0) or 0.0) >= 0.5
                    except Exception:
                        arc_only = False
                    try:
                        angle_span = float(feats.get("angle_span")) if feats.get("angle_span") is not None else None
                    except Exception:
                        angle_span = None
                is_swing_arc_like = ctype == "swing_arc" and arc_only and (angle_span is not None) and (20.0 <= angle_span <= 125.0)
                if (not is_swing_arc_like) and iou < MIN_SNAP_IOU_FOR_SQUARE:
                    continue

        if iou > best_iou:
            best_iou = iou
            best_by_iou = cand
            best_iou_inter = inter
        if inter > best_inter:
            best_inter = inter
            best_by_inter = cand
            best_inter_iou = iou
            best_inter_coverage = coverage
        if coverage > best_coverage:
            best_coverage = coverage
            best_by_coverage = cand
            best_coverage_inter = inter

    if not any_overlap:
        return None, 0.0

    # Primary: max IoU.
    # NOTE: IoU can be tiny when the user draws a big box around a small candidate.
    # Use candidate coverage (intersection / candidate area) as an alternate signal,
    # and avoid snapping on tiny corner overlaps.
    MIN_SNAP_IOU = 0.06
    MIN_CAND_COVERAGE = 0.25
    MIN_INTER_FRAC_OF_DRAWN = 0.06
    # Reviewers often draw around a swing door including the room tag bubble
    # (commonly placed inside the arc). That makes the drawn ROI much larger
    # than the true door candidate bbox, producing tiny IoU and causing the
    # anti-false-positive guardrail to reject snapping entirely.
    #
    # Keep the stricter threshold for generic symbol-like candidates, but allow
    # a weaker "candidate is inside ROI" snap for door-like types with decent confidence.
    MIN_INTER_FRAC_OF_DRAWN_DOORLIKE = 0.01
    MIN_CONF_DOORLIKE_RELAX = 0.55
    DOORLIKE_TYPES = {"swing", "swing_arc", "double", "pocket", "bifold", "swing_leaf"}
    if best_by_iou is not None and best_iou >= MIN_SNAP_IOU:
        # Heuristic: if the IoU-best candidate overlaps *far* less than another overlapping
        # candidate, prefer maximum intersection. This avoids snapping to small nearby
        # shapes (e.g. circles) when the user drew a larger box around a door.
        MIN_IOU_INTER_FRAC_OF_MAX_INTER = 0.72
        if (
            best_by_inter is not None
            and best_inter > 0.0
            and best_iou_inter > 0.0
            and best_iou_inter < (best_inter * MIN_IOU_INTER_FRAC_OF_MAX_INTER)
        ):
            # Guardrail:
            # Large false-positive candidates (especially long-span `bifold` tracks) can overlap
            # the entire drawn ROI, giving huge intersection area, but tiny IoU/coverage. In
            # those cases, overriding the IoU-best pick is counterproductive (it selects the
            # wrong thing and then we fall back to a manual-box candidate).
            #
            # If the IoU-best pick is reasonably strong, only override when the max-intersection
            # candidate is also plausible (non-tiny IoU OR meaningfully covered by the ROI).
            if float(best_iou) >= 0.10 and (float(best_inter_iou) < 0.04 or float(best_inter_coverage) < 0.06):
                # Keep the IoU-best pick.
                pass
            else:
                cb = _normalize_bbox_xyxy(best_by_inter.get("bbox_xyxy"))
                if cb is not None:
                    return best_by_inter, max(0.0, float(compute_iou(drawn, [cb[0], cb[1], cb[2], cb[3]])))
        return best_by_iou, max(0.0, float(best_iou))

    # Fallback: candidate is mostly covered by the selection.
    if best_by_coverage is not None and best_coverage >= MIN_CAND_COVERAGE:
        # Guardrail: avoid snapping to a tiny candidate that happens to be fully contained
        # by a much larger drawn box (common false-positive near doors).
        inter_frac = (float(best_coverage_inter) / float(drawn_area)) if drawn_area > 0.0 else 0.0
        try:
            ctype = str(best_by_coverage.get("type") or "")
        except Exception:
            ctype = ""
        try:
            cconf = float(best_by_coverage.get("confidence", 0.0) or 0.0)
        except Exception:
            cconf = 0.0
        min_inter_frac = MIN_INTER_FRAC_OF_DRAWN
        if ctype in DOORLIKE_TYPES and cconf >= MIN_CONF_DOORLIKE_RELAX:
            min_inter_frac = min(min_inter_frac, MIN_INTER_FRAC_OF_DRAWN_DOORLIKE)
        if inter_frac < min_inter_frac:
            return None, 0.0
        cb = _normalize_bbox_xyxy(best_by_coverage.get("bbox_xyxy"))
        if cb is not None:
            return best_by_coverage, max(0.0, float(compute_iou(drawn, [cb[0], cb[1], cb[2], cb[3]])))

    # Fallback: max intersection area, but only if it is meaningful relative to the drawn box.
    if best_by_inter is not None and best_inter > 0.0:
        inter_frac = (float(best_inter) / float(drawn_area)) if drawn_area > 0.0 else 0.0
        try:
            itype = str(best_by_inter.get("type") or "")
        except Exception:
            itype = ""
        try:
            iconf = float(best_by_inter.get("confidence", 0.0) or 0.0)
        except Exception:
            iconf = 0.0
        min_inter_frac = MIN_INTER_FRAC_OF_DRAWN
        if itype in DOORLIKE_TYPES and iconf >= MIN_CONF_DOORLIKE_RELAX:
            min_inter_frac = min(min_inter_frac, MIN_INTER_FRAC_OF_DRAWN_DOORLIKE)
        if inter_frac >= min_inter_frac:
            cb = _normalize_bbox_xyxy(best_by_inter.get("bbox_xyxy"))
            if cb is not None:
                return best_by_inter, max(0.0, float(compute_iou(drawn, [cb[0], cb[1], cb[2], cb[3]])))

    return None, 0.0


def _debug_unmatched_region(
    *,
    file_dir: Path,
    drawn_bbox_full_xyxy: List[float],
    config_path: str,
    extra: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Optional[str]:
    """Debug aid for unmatched Shift+drag boxes.

    Returns a JSON string; the viewer prints it to the browser console.
    """

    def _shrink_report_for_transport(rep: Any) -> Any:
        """Reduce report size for normal (non-verbose) transport.

        Streamlit component props are sent over a websocket; large debug blobs can
        cause disconnects (and noisy Tornado WebSocketClosedError traces). The
        server already prints a compact summary, so for the browser-side log we
        keep only high-signal fields unless verbose mode is enabled.
        """
        if not isinstance(rep, dict):
            return rep
        return {
            "kind": rep.get("kind"),
            "verbose": bool(rep.get("verbose", False)),
            "bbox_full_xyxy": rep.get("bbox_full_xyxy"),
            "roi_full_xyxy": rep.get("roi_full_xyxy"),
            "counts": rep.get("counts"),
            "summary": rep.get("summary"),
            "extra": rep.get("extra"),
            "file_dir": rep.get("file_dir"),
            "config_path": rep.get("config_path"),
            "truncation": rep.get("truncation"),
        }

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
        prim_bytes = None
        if perf_enabled():
            try:
                prim_bytes = int(primitives_path.stat().st_size)
            except Exception:
                prim_bytes = None
        with perf_span(
            "ui.unmatched_debug.load_primitives",
            file_dir=str(file_dir),
            bytes=prim_bytes,
            verbose=bool(verbose),
        ):
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
        with perf_span("ui.unmatched_debug.load_config", config_path=str(config_path)):
            cfg = json.loads(Path(config_path).read_bytes())
    except Exception as e:
        cfg = {"error": f"failed_to_load_config: {e}"}

    try:
        with perf_span(
            "ui.unmatched_debug.explain",
            verbose=bool(verbose),
            drawn_bbox_xyxy=[float(v) for v in (drawn_bbox_full_xyxy or [])[:4]],
        ):
            rep = debug_explain_unmatched_box(
                primitives=primitives, bbox_full_xyxy=drawn_bbox_full_xyxy, config=cfg, verbose=bool(verbose)
            )
        rep["file_dir"] = str(file_dir)
        rep["config_path"] = str(config_path)
        if isinstance(extra, dict) and extra:
            rep["extra"] = extra
        if not bool(verbose):
            rep = _shrink_report_for_transport(rep)
        # Also emit a short server-side summary so we can debug without copying a huge JSON blob.
        try:
            summ = rep.get("summary") if isinstance(rep, dict) else None
            if isinstance(summ, dict):
                payload = {
                    "file_dir": str(file_dir),
                    "bbox_full_xyxy": [float(v) for v in drawn_bbox_full_xyxy],
                    "extra": extra if isinstance(extra, dict) else None,
                    "summary": summ,
                }
                print("[door_detector] unmatched_debug_summary", json.dumps(payload, separators=(",", ":")))
        except Exception:
            pass
        with perf_span("ui.unmatched_debug.dumps", verbose=bool(verbose)):
            out = json.dumps(rep, separators=(",", ":"))
        if perf_enabled():
            perf_log("ui.unmatched_debug.payload", bytes=len(out), verbose=bool(verbose))
        return out
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
    full_dims: Optional[Tuple[int, int]],
    config_path: str,
) -> None:
    """Consume a Shift+drag draw event from the PDF viewer (if present)."""
    draw_key = f"draw_event_sink_{file_id}"
    raw = st.session_state.get(draw_key) or ""
    if not raw:
        return
    if perf_enabled():
        perf_log("ui.draw_event.sink_nonempty", file_id=str(file_id), bytes=len(str(raw)))
    # Always clear the sink, even if parsing or downstream processing fails.
    # Otherwise the UI can get stuck showing a client-only temp overlay with no way to cancel it.
    try:
        with perf_span("ui.draw_event.parse_json", file_id=str(file_id), bytes=len(str(raw))):
            evt = json.loads(raw)
    except Exception:
        evt = None
    finally:
        try:
            st.session_state[draw_key] = ""
        except Exception:
            pass

    if not isinstance(evt, dict):
        return
    if evt.get("event") != "draw_rect":
        return
    event_id = evt.get("event_id")
    bbox_pdf = evt.get("bbox_pdf_xyxy")
    snapped_candidate_id = evt.get("snapped_candidate_id")
    if not event_id:
        return
    if str(event_id) == str(fstate.get("_last_draw_event_id")):
        return
    fstate["_last_draw_event_id"] = str(event_id)
    if perf_enabled():
        perf_log(
            "ui.draw_event.accepted",
            file_id=str(file_id),
            event_id=str(event_id),
            has_snapped=bool(snapped_candidate_id not in (None, "")),
        )
    prev_selected = str(fstate.get("selected_door_id") or "")
    push_breadcrumb(
        fstate,
        {
            "kind": "draw_event_received",
            "file_id": str(file_id),
            "event_id": str(event_id),
            "prev_selected_door_id": prev_selected,
            "snapped_candidate_id": str(snapped_candidate_id) if snapped_candidate_id not in (None, "") else "",
        },
    )

    # PDF.js emits bbox_pdf_xyxy in PDF coords; convert PDF → pixel using Step1 transform.
    if not (isinstance(bbox_pdf, list) and len(bbox_pdf) == 4):
        return
    try:
        with perf_span("ui.draw_event.pdf_to_pix", file_id=str(file_id), event_id=str(event_id)):
            tpath = file_dir / "transform.json"
            tob = json.loads(tpath.read_text()) if tpath.exists() else {}
            m = tob.get("pdf_to_pix_affine") if isinstance(tob, dict) else None
            cb = tob.get("cropbox") if isinstance(tob, dict) else None
            if not (isinstance(m, list) and len(m) == 6 and isinstance(cb, dict)):
                return
            # `pdf_to_pix_affine` expects fitz coordinates (Y-down). The PDF.js viewer emits
            # PDF-spec coords (Y-up). Use our shared conversion logic which also handles
            # "centered" PDF page boxes (negative coords) correctly.
            from door_detector.pdf.affine import pdfjs_bbox_to_fitz_bbox_xyxy

            bbox_fitz = pdfjs_bbox_to_fitz_bbox_xyxy([float(v) for v in bbox_pdf], cropbox=cb)
            drawn_full = _apply_affine_bbox_xyxy(m, bbox_fitz)
    except Exception:
        return

    full_w = full_dims[0] if full_dims else None
    full_h = full_dims[1] if full_dims else None
    drawn_full = _clamp_bbox_xyxy([float(v) for v in drawn_full], w=full_w, h=full_h)

    # Candidate pool: detected candidates + any UI-generated manual candidates.
    candidates = list(doors_data.get("candidates", doors_data.get("doors", [])) or [])
    try:
        candidates.extend(list(fstate.get("manual_candidates", []) or []))
    except Exception:
        pass
    if perf_enabled():
        perf_log(
            "ui.draw_event.candidate_pool",
            file_id=str(file_id),
            event_id=str(event_id),
            candidates=int(len(candidates)),
        )

    # --- Build a ranked suggestion list for cycling (UI) ---
    # Users often need to cycle through overlapping candidates (e.g. double vs two swings,
    # arc-only vs full swing). We compute a ranked list here so the right panel can
    # offer Prev/Next selection without rerunning geometry detection.
    try:
        with perf_span(
            "ui.draw_event.build_suggestions",
            file_id=str(file_id),
            event_id=str(event_id),
            candidates=int(len(candidates)),
        ):
            bx0, by0, bx1, by1 = _normalize_bbox_xyxy(drawn_full) or (0.0, 0.0, 0.0, 0.0)
            bcx = 0.5 * (bx0 + bx1)
            bcy = 0.5 * (by0 + by1)

            scored: List[Tuple[Tuple[float, float, float], Dict[str, Any]]] = []
            nearest: List[Tuple[Tuple[float, float], Dict[str, Any]]] = []
            for cand in candidates:
                cid = cand.get("id")
                cb = _normalize_bbox_xyxy(cand.get("bbox_xyxy"))
                if cid is None or cb is None:
                    continue
                cx0, cy0, cx1, cy1 = cb
                ix0 = max(bx0, cx0)
                iy0 = max(by0, cy0)
                ix1 = min(bx1, cx1)
                iy1 = min(by1, cy1)
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                if inter <= 0.0:
                    # Track "nearby" candidates so the UI can still offer cycling when
                    # the selection box barely misses the right candidate bbox.
                    try:
                        ccx = 0.5 * (cx0 + cx1)
                        ccy = 0.5 * (cy0 + cy1)
                        dist = float(math.hypot(ccx - bcx, ccy - bcy))
                        nearest.append(
                            ((dist, float(cand.get("confidence", cand.get("heuristic_confidence", 0.0)) or 0.0)), cand)
                        )
                    except Exception:
                        pass
                    continue
                iou_c = float(compute_iou([bx0, by0, bx1, by1], [cx0, cy0, cx1, cy1]))
                conf = float(cand.get("confidence", cand.get("heuristic_confidence", 0.0)) or 0.0)
                scored.append(((iou_c, inter, conf), cand))

            scored.sort(key=lambda x: x[0], reverse=True)
            nearest.sort(key=lambda x: x[0])

            pool_map = {str(c.get("id")): c for c in candidates if c.get("id") is not None}

            suggestions: List[Dict[str, Any]] = []
            seen: set[str] = set()

            for (iou_c, inter, conf), cand in scored[:40]:
                cid = str(cand.get("id") or "")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                suggestions.append(
                    {
                        "id": cid,
                        "type": str(cand.get("type") or ""),
                        "iou": float(iou_c),
                        "inter": float(inter),
                        "confidence": float(conf),
                        "source": "overlap",
                    }
                )

                # Expand doubles into component swings so the reviewer can cycle to them.
                if str(cand.get("type") or "") == "double":
                    try:
                        comps = (cand.get("components") or {}) if isinstance(cand, dict) else {}
                        swing_ids = comps.get("swing_ids") or []
                        if isinstance(swing_ids, list):
                            for sid in swing_ids:
                                ssid = str(sid) if sid not in (None, "") else ""
                                if not ssid or ssid in seen:
                                    continue
                                sc = pool_map.get(ssid)
                                if not sc:
                                    continue
                                scb = _normalize_bbox_xyxy(sc.get("bbox_xyxy"))
                                if scb is None:
                                    continue
                                siou = float(compute_iou([bx0, by0, bx1, by1], [scb[0], scb[1], scb[2], scb[3]]))
                                seen.add(ssid)
                                suggestions.append(
                                    {
                                        "id": ssid,
                                        "type": str(sc.get("type") or "swing"),
                                        "iou": float(siou),
                                        "inter": 0.0,
                                        "confidence": float(sc.get("confidence", sc.get("heuristic_confidence", 0.0)) or 0.0),
                                        "source": "double_component",
                                        "parent_double_id": cid,
                                    }
                                )
                    except Exception:
                        pass

            # If nothing overlaps, include a few nearby candidates (best-effort) so the
            # reviewer can still cycle when bboxes are slightly mislocalized.
            if not suggestions and nearest:
                MAX_NEARBY = 12
                MAX_NEARBY_DIST_PX = 320.0
                for (dist, conf), cand in nearest[:MAX_NEARBY]:
                    if float(dist) > MAX_NEARBY_DIST_PX:
                        continue
                    cid = str(cand.get("id") or "")
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)
                    suggestions.append(
                        {
                            "id": cid,
                            "type": str(cand.get("type") or ""),
                            "iou": 0.0,
                            "inter": 0.0,
                            "confidence": float(conf),
                            "source": "nearby",
                            "dist_px": float(dist),
                        }
                    )

            fstate["_last_draw_suggestions"] = {
                "event_id": str(event_id),
                "drawn_bbox_xyxy": [float(v) for v in drawn_full],
                "suggestions": suggestions[:50],
            }
    except Exception:
        fstate["_last_draw_suggestions"] = {"event_id": str(event_id), "suggestions": []}

    # Server-side snap is the source of truth (uses full candidate list).
    with perf_span("ui.draw_event.snap_server", file_id=str(file_id), event_id=str(event_id), candidates=int(len(candidates))):
        best_server, iou_server = _snap_to_candidate(drawn_full, candidates=candidates)

    # Client-proposed snap (from the viewer's candidatePool) is only a hint.
    # The pool is intentionally downsampled for performance, so accepting it blindly
    # can produce incorrect snaps (e.g. snapping to a nearby false-positive circle).
    best_client = None
    iou_client = 0.0
    if snapped_candidate_id:
        sid = str(snapped_candidate_id)
        best_client = next((c for c in candidates if str(c.get("id") or "") == sid), None)
        if best_client is not None:
            cb = _normalize_bbox_xyxy(best_client.get("bbox_xyxy"))
            if cb is not None:
                iou_client = float(compute_iou(drawn_full, [cb[0], cb[1], cb[2], cb[3]]))
                # Treat non-overlapping client snaps as invalid.
                if iou_client <= 0.0:
                    best_client = None
                    iou_client = 0.0

    best = best_server
    iou = float(iou_server or 0.0)
    if best is None and best_client is not None:
        best = best_client
        iou = float(iou_client or 0.0)
    elif best is not None and best_client is not None:
        # If the client suggestion is strictly better than the server pick, allow it.
        # (In practice this should be rare; mostly useful when both overlap and the
        # server fallback picked a different candidate with a lower IoU.)
        if float(iou_client) > float(iou_server) + 1e-9:
            best = best_client
            iou = float(iou_client)

    # Diagnostics: mismatch + "weak snap" debugging.
    #
    # The core issue in many Edit Doors failures is *candidate existence*, not snap logic:
    # the user draws around a door symbol that never became a candidate, so the snap
    # falls back to some other nearby overlapping symbol (often with very low IoU).
    #
    # Historically we only emitted `unmatched_debug_report` when *no* candidate overlapped.
    # That misses the important case "overlapCandidates > 0 but the intended door is absent".
    #
    # So we also emit a bounded ROI explanation when the chosen snap is suspiciously weak.
    try:
        sid = str(snapped_candidate_id) if snapped_candidate_id not in (None, "") else ""
        server_id = str(best_server.get("id")) if isinstance(best_server, dict) and best_server.get("id") is not None else ""
        client_id = str(best_client.get("id")) if isinstance(best_client, dict) and best_client.get("id") is not None else ""

        debug_reason = ""
        if sid and server_id and client_id and server_id != client_id:
            debug_reason = "snap_mismatch"

        # Track whether there were *any* overlapping candidates (even if snap thresholds reject them).
        overlap_suggestions = 0
        try:
            srec = fstate.get("_last_draw_suggestions") or {}
            sugg = list(srec.get("suggestions") or [])
            overlap_suggestions = sum(1 for r in sugg if isinstance(r, dict) and str(r.get("source") or "") == "overlap")
        except Exception:
            overlap_suggestions = 0

        # "Weak snap" heuristic: if the best match barely overlaps the drawn region,
        # it is often not the user's intended door. Emit a detailed ROI report so we can
        # see whether the intended door's primitives fail arc/leaf criteria.
        WEAK_SNAP_IOU = 0.12
        if not debug_reason and (
            (best is not None and float(iou or 0.0) < WEAK_SNAP_IOU)
            or (best is None and overlap_suggestions > 0)
        ):
            debug_reason = "weak_snap_low_iou"

        # If nothing overlaps at all, we still want a geometry explanation.
        # Otherwise we cannot answer: "was the intended door missing as a candidate because
        # the arc/leaf primitives were missing, or because arc extraction/pairing failed?"
        #
        # This case is especially common on plans where swing arcs are rasterized or drawn
        # as dashed polylines (and therefore can be missed by conservative arc extraction).
        if not debug_reason and best is None and overlap_suggestions <= 0:
            debug_reason = "no_overlap_candidates"

        if debug_reason:
            # Always emit a lightweight, copy-friendly ROI summary when the snap is likely wrong.
            # Use `debug_draw_roi=true` (session state) only to switch to the full verbose report.
            verbose_roi = bool(st.session_state.get("debug_draw_roi", False))
            with perf_span(
                "ui.draw_event.unmatched_debug",
                file_id=str(file_id),
                event_id=str(event_id),
                reason=str(debug_reason),
                verbose=bool(verbose_roi),
            ):
                fstate["_last_unmatched_debug"] = _debug_unmatched_region(
                    file_dir=file_dir,
                    drawn_bbox_full_xyxy=drawn_full,
                    config_path=str(config_path),
                    verbose=verbose_roi,
                    extra={
                        "debug_reason": str(debug_reason),
                        "event_id": str(event_id),
                        "overlap_suggestions": int(overlap_suggestions),
                        "client_snapped_candidate_id": client_id,
                        "client_iou": float(iou_client),
                        "server_snapped_candidate_id": server_id,
                        "server_iou": float(iou_server),
                        "chosen_candidate_id": str(best.get("id") or "") if isinstance(best, dict) else "",
                        "chosen_iou": float(iou or 0.0),
                        "drawn_bbox_pdf_xyxy": [float(v) for v in bbox_pdf]
                        if isinstance(bbox_pdf, list) and len(bbox_pdf) == 4
                        else [],
                        "verbose": bool(verbose_roi),
                    },
                )
        else:
            # Clear any prior mismatch/unmatched debug so future changes re-trigger logs.
            fstate["_last_unmatched_debug"] = None
    except Exception:
        pass

    # If nothing overlaps, create a UI-only manual candidate from the drawn box so
    # the reviewer can still proceed (propose + confirm) even when detection missed.
    #
    # Also treat *extremely weak* snaps as effectively unmatched. In these cases the
    # selection box usually only barely clips an unrelated nearby symbol/candidate,
    # and the user’s intended door is absent from the candidate list.
    created_manual_candidate_id = ""
    try:
        if best is not None and isinstance(fstate.get("_last_unmatched_debug"), str):
            # Keep this heuristic strict: only override when the snap is so weak that it
            # would be more harmful than helpful as the default selection.
            debug_reason = ""
            try:
                raw_dbg = str(fstate.get("_last_unmatched_debug") or "")
                if raw_dbg:
                    obj_dbg = json.loads(raw_dbg)
                    extra_dbg = obj_dbg.get("extra") if isinstance(obj_dbg, dict) else None
                    if isinstance(extra_dbg, dict):
                        debug_reason = str(extra_dbg.get("debug_reason") or "")
            except Exception:
                debug_reason = ""
            MIN_IOU_TREAT_AS_UNMATCHED = 0.04
            if debug_reason == "weak_snap_low_iou" and float(iou or 0.0) < MIN_IOU_TREAT_AS_UNMATCHED:
                best = None
                iou = 0.0
    except Exception:
        pass

    if best is None:
        try:
            cid = _manual_candidate_id_from_bbox(file_id=str(file_id), bbox_xyxy=drawn_full, quant_step_px=1.0)
            mtype = _preferred_label_type_for_draw(file_id=str(file_id), default="swing")
            manual = {
                "id": str(cid),
                # Use the UI's preferred type so downstream controls (label defaults,
                # typed reject button text, filters) behave consistently for manual boxes.
                "type": str(mtype),
                "bbox_xyxy": list(drawn_full),
                "confidence": 0.0,
                "heuristic_confidence": 0.0,
                "pool": True,
                "features": {},
            }
            fstate.setdefault("manual_candidates", [])
            # Deduplicate by id (keep first).
            if not any(
                str(c.get("id") or "") == str(cid)
                for c in (fstate.get("manual_candidates") or [])
                if isinstance(c, dict)
            ):
                fstate["manual_candidates"].append(manual)
                created_manual_candidate_id = str(cid)
            best = manual
            iou = 1.0
            # Ensure it shows up in Selection matches immediately.
            try:
                srec = fstate.get("_last_draw_suggestions") or {}
                sugg = list(srec.get("suggestions") or [])
                sugg = [r for r in sugg if str(r.get("id") or "") != str(cid)]
                sugg.insert(
                    0,
                    {
                        "id": str(cid),
                        "type": str(mtype),
                        "iou": 1.0,
                        "inter": 0.0,
                        "confidence": 0.0,
                        "source": "manual_box",
                    },
                )
                fstate["_last_draw_suggestions"] = {
                    "event_id": str(event_id),
                    "drawn_bbox_xyxy": [float(v) for v in drawn_full],
                    "suggestions": sugg[:50],
                }
                st.session_state[f"_draw_suggest_idx_{file_id}"] = 0
            except Exception:
                pass
        except Exception:
            best = None

    if best is not None and best.get("id") is not None:
        cid = str(best["id"])
        # Guarantee the proposal menu can render even if suggestion ranking failed earlier.
        # (If suggestions are empty, the right panel previously hid proposal controls,
        # leaving the user with a non-interactive green "snapped" overlay.)
        try:
            srec = fstate.get("_last_draw_suggestions") or {}
            sugg = list(srec.get("suggestions") or [])
        except Exception:
            sugg = []
        if not sugg:
            try:
                sugg_type = str(best.get("type") or "")
            except Exception:
                sugg_type = ""
            fstate["_last_draw_suggestions"] = {
                "event_id": str(event_id),
                "drawn_bbox_xyxy": [float(v) for v in drawn_full],
                "suggestions": [
                    {
                        "id": str(cid),
                        "type": sugg_type,
                        "iou": float(iou or 0.0),
                        "inter": 0.0,
                        "confidence": float(best.get("confidence", best.get("heuristic_confidence", 0.0)) or 0.0)
                        if isinstance(best, dict)
                        else 0.0,
                        "source": "fallback_best",
                    }
                ],
            }
        # Align the cycling index to the chosen id (best effort).
        try:
            srec = fstate.get("_last_draw_suggestions") or {}
            sugg = list(srec.get("suggestions") or [])
            idx = next((i for i, r in enumerate(sugg) if str(r.get("id") or "") == cid), None)
            if isinstance(idx, int) and idx >= 0:
                st.session_state[f"_draw_suggest_idx_{file_id}"] = int(idx)
        except Exception:
            pass
        snapped_full = (
            _normalize_bbox_xyxy(best.get("bbox_xyxy"))
            or _normalize_bbox_xyxy(drawn_full)
            or (0.0, 0.0, 0.0, 0.0)
        )
        # Record proposal state (used for overlays + discard behavior).
        try:
            fstate["_proposal"] = {
                "event_id": str(event_id),
                "drawn_bbox_xyxy": [float(v) for v in drawn_full],
                "drawn_bbox_pdf_xyxy": [float(v) for v in bbox_pdf],
                "snapped_candidate_id": str(cid),
                "prev_selected_door_id": str(prev_selected),
                "snapped_bbox_xyxy": [
                    float(snapped_full[0]),
                    float(snapped_full[1]),
                    float(snapped_full[2]),
                    float(snapped_full[3]),
                ],
                "iou": float(iou),
                "created_manual_candidate_id": str(created_manual_candidate_id) if created_manual_candidate_id else "",
            }
        except Exception:
            pass

        # Make the snapped candidate the current selection (but do not label it yet).
        try:
            fstate["selected_door_id"] = cid
            st.session_state[f"jump_{file_id}"] = cid
            # Suppress focus bump when selection change originates from draw/snap.
            fstate["_focus_last_id"] = cid
        except Exception:
            pass
        push_breadcrumb(
            fstate,
            {
                "kind": "draw_event_proposed_selection",
                "file_id": str(file_id),
                "event_id": str(event_id),
                "prev_selected_door_id": prev_selected,
                "selected_door_id": str(fstate.get("selected_door_id") or ""),
                "proposal_snapped_candidate_id": cid,
                "iou": float(iou or 0.0),
                "created_manual_candidate_id": str(created_manual_candidate_id or ""),
            },
        )
    else:
        # No match (unexpected). Record a proposal so the UI can show the drawn region.
        try:
            fstate["_proposal"] = {
                "event_id": str(event_id),
                "drawn_bbox_xyxy": [float(v) for v in drawn_full],
                "drawn_bbox_pdf_xyxy": [float(v) for v in bbox_pdf],
                "snapped_candidate_id": "",
                "prev_selected_door_id": str(prev_selected),
                "snapped_bbox_xyxy": [],
                "iou": 0.0,
                "created_manual_candidate_id": "",
            }
        except Exception:
            pass
        fstate["_last_unmatched_debug"] = _debug_unmatched_region(
            file_dir=file_dir,
            drawn_bbox_full_xyxy=drawn_full,
            config_path=str(config_path),
            extra={
                "debug_reason": "no_match_after_manual_candidate_fallback_failed",
                "event_id": str(event_id),
                "drawn_bbox_pdf_xyxy": [float(v) for v in bbox_pdf] if isinstance(bbox_pdf, list) and len(bbox_pdf) == 4 else [],
            },
        )
        push_breadcrumb(
            fstate,
            {
                "kind": "draw_event_unmatched",
                "file_id": str(file_id),
                "event_id": str(event_id),
                "prev_selected_door_id": prev_selected,
            },
        )


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
                    # Logging intentionally suppressed (noise).
                    pass

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
                    # NOTE: Streamlit treats height=0 as "default" in some builds; use 1px.
                    components.html(assets.sidebar_autopen_component_html(), height=1, scrolling=False)
                    # IMPORTANT: avoid `st.empty()` here.
                    #
                    # Using a placeholder tends to *replace* its children on each rerun,
                    # which can destroy and recreate the PDF.js component iframe.
                    # That forces a PDF reload + canvas rerender even for selection-only
                    # state changes, and causes pan/zoom "reset flashes".
                    viewer_slot = st.container()
                with col_review:
                    review_slot = st.container()

                # If a pipeline run is queued for this file, keep rendering the *existing*
                # viewer state while analysis runs (don't swap in a loader).
                # We'll run the pipeline *after* the viewer/right panel have rendered so the
                # previous state remains visible until results refresh.
                task = st.session_state.get("door_detector_pipeline_task")
                should_run_pipeline_now = bool(
                    task and task.get("file_id") == str(file_id) and not bool(task.get("_started"))
                )
                if should_run_pipeline_now:
                    try:
                        task["_started"] = True
                        st.session_state.door_detector_pipeline_task = task
                    except Exception:
                        pass

                try:
                    with perf_span("ui.load_file_artifacts", file_id=str(file_id)):
                        doors_data, labels_data, meta_data = load_file_artifacts(str(file_dir))
                except Exception as e:
                    st.error(str(e))
                    st.stop()
                init_file_state(file_id, doors_data, labels_data)
                fstate = st.session_state.files[file_id]
                _remap_fstate_ids_using_legacy_ids(fstate=fstate, doors_data=doors_data)

                full_dims = get_full_page_dims(meta_data)

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
                            if perf_enabled():
                                perf_log(
                                    "ui.viewer_event",
                                    file_id=str(file_id),
                                    type=str(et),
                                    event_id=str(evt_id),
                                )
                            click_sink_label = f"door_click_sink_{file_id}"
                            if et == "door_click":
                                did = evt.get("door_id")
                                if did not in (None, ""):
                                    st.session_state[click_sink_label] = str(did)
                                push_breadcrumb(
                                    fstate,
                                    {
                                        "kind": "viewer_event",
                                        "file_id": str(file_id),
                                        "type": "door_click",
                                        "event_id": evt_id,
                                        "door_id": str(did) if did not in (None, "") else "",
                                        "selected_door_id_before": str(fstate.get("selected_door_id") or ""),
                                    },
                                )
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
                                push_breadcrumb(
                                    fstate,
                                    {
                                        "kind": "viewer_event",
                                        "file_id": str(file_id),
                                        "type": "draw_rect",
                                        "event_id": evt_id,
                                        "snapped_candidate_id": str(evt.get("snapped_candidate_id") or ""),
                                        "ts_client": evt.get("ts"),
                                        "selected_door_id_before": str(fstate.get("selected_door_id") or ""),
                                    },
                                )
                            elif et == "focus_state":
                                did = evt.get("door_id")
                                in_focus = bool(evt.get("in_focus"))
                                did_s = str(did) if did not in (None, "") else ""
                                cur_s = str(fstate.get("selected_door_id") or "")
                                if did_s and did_s == cur_s:
                                    if in_focus:
                                        fstate["_focused_door_id"] = did_s
                                    else:
                                        if str(fstate.get("_focused_door_id") or "") == did_s:
                                            fstate["_focused_door_id"] = None
                                push_breadcrumb(
                                    fstate,
                                    {
                                        "kind": "viewer_event",
                                        "file_id": str(file_id),
                                        "type": "focus_state",
                                        "event_id": evt_id,
                                        "door_id": did_s,
                                        "in_focus": bool(in_focus),
                                        "selected_door_id_before": cur_s,
                                        "focused_door_id_before": str(fstate.get("_focused_door_id") or ""),
                                    },
                                )
                except Exception:
                    pass

                # Consume any Shift+drag events before computing the visible list/overlay.
                _process_draw_event_if_any(
                    file_id=str(file_id),
                    file_dir=file_dir,
                    fstate=fstate,
                    doors_data=doors_data,
                    full_dims=full_dims,
                    config_path=str(doors_data.get("config_path") or _default_config_path_str()),
                )

                # If a prior run stored an unmatched debug blob (from a Shift+drag),
                # emit a compact summary to the server console once per event.
                try:
                    raw_dbg = fstate.get("_last_unmatched_debug")
                    if isinstance(raw_dbg, str) and raw_dbg:
                        obj_dbg = json.loads(raw_dbg)
                        extra_dbg = obj_dbg.get("extra") if isinstance(obj_dbg, dict) else None
                        summ_dbg = obj_dbg.get("summary") if isinstance(obj_dbg, dict) else None
                        ev_dbg = ""
                        if isinstance(extra_dbg, dict):
                            ev_dbg = str(extra_dbg.get("event_id") or "")
                        if not ev_dbg:
                            ev_dbg = str(fstate.get("_last_draw_event_id") or "")
                        if ev_dbg and str(fstate.get("_last_unmatched_debug_printed_event_id") or "") != ev_dbg:
                            fstate["_last_unmatched_debug_printed_event_id"] = ev_dbg
                            if isinstance(summ_dbg, dict):
                                print(
                                    "[door_detector] unmatched_debug_summary",
                                    json.dumps(
                                        {
                                            "file_id": str(file_id),
                                            "file_dir": str(file_dir),
                                            "event_id": ev_dbg,
                                            "extra": extra_dbg if isinstance(extra_dbg, dict) else None,
                                            "summary": summ_dbg,
                                        },
                                        separators=(",", ":"),
                                    ),
                                )
                except Exception:
                    pass

                # Compute active doors once so the main viewer + right panel stay in perfect sync.
                detections = doors_data.get("doors", [])
                working = _get_working_label_state(fstate)
                deleted_ids = coerce_id_set(working.get("deleted_ids", set()))
                rejected_ids = flatten_rejected_ids(working.get("rejected_by_type", {}))
                hidden_ids = set(deleted_ids) | set(rejected_ids)
                overlay_doors: List[Dict[str, Any]] = [d for d in detections if d.get("id") is not None]

                # Ensure any confirmed snapped candidates are also rendered (even if they were
                # not in the strict output doors list).
                try:
                    extra_ids = set(flatten_confirmed_ids(working.get("confirmed_by_type", {})))
                    for rec in list(working.get("manual_additions", [])):
                        cid = rec.get("snapped_candidate_id")
                        if cid:
                            extra_ids.add(str(cid))

                    # Always include any persisted manual candidates (so they can be selected/labeled).
                    for mc in list(working.get("manual_candidates", []) or []):
                        if not isinstance(mc, dict):
                            continue
                        mid = mc.get("id")
                        if mid not in (None, ""):
                            extra_ids.add(str(mid))

                    # Also include the currently proposed candidate (if any), so it is labelable
                    # even when it is not part of the strict `doors` output list.
                    try:
                        prop = fstate.get("_proposal") or {}
                        if isinstance(prop, dict):
                            pid = prop.get("snapped_candidate_id")
                            if pid not in (None, ""):
                                extra_ids.add(str(pid))
                    except Exception:
                        pass

                    # If a double candidate was rejected (e.g. it was actually two swings),
                    # reveal its component swing candidates so the reviewer can label them.
                    pool = list(doors_data.get("candidates", []) or [])
                    try:
                        pool.extend(list(working.get("manual_candidates", []) or []))
                    except Exception:
                        pass
                    pool_map = {str(c.get("id")): c for c in pool if c.get("id") is not None}
                    try:
                        rbt = working.get("rejected_by_type", {})
                        double_rej = set()
                        if isinstance(rbt, dict):
                            ids = rbt.get("double")
                            if isinstance(ids, set):
                                double_rej = set(ids)
                        for rid in list(double_rej):
                            dc = pool_map.get(str(rid))
                            comps = (dc.get("components") or {}) if isinstance(dc, dict) else {}
                            if not isinstance(comps, dict):
                                continue
                            swing_ids = comps.get("swing_ids") or []
                            if isinstance(swing_ids, list):
                                for sid in swing_ids:
                                    if sid not in (None, ""):
                                        extra_ids.add(str(sid))
                    except Exception:
                        pass

                    existing_ids = {str(d.get("id")) for d in overlay_doors if d.get("id") is not None}
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

                active_doors_all: List[Dict[str, Any]] = [d for d in overlay_doors if str(d.get("id")) not in hidden_ids]

                # Unified single-stage filter: choose exactly one of {All, Confirmed, Unconfirmed, <type>}.
                try:
                    confirmed_ids = set(flatten_confirmed_ids(working.get("confirmed_by_type", {})))
                except Exception:
                    confirmed_ids = set()
                # Unified single-stage filter: choose exactly one of {All, Confirmed, Unconfirmed, <type>}.
                #
                # IMPORTANT: keep the canonical filter state separate from the widget key.
                # The radio widget uses `door_filter_widget_{file_id}`; our app state uses
                # `door_filter_{file_id}` which is stable across action-triggered reruns.
                # Canonical filter state (NOT a widget key). Use a distinct prefix so Streamlit
                # never confuses it with historical widget keys (prevents odd resets).
                door_filter_key = f"_door_detector_door_filter_state_{file_id}"
                door_filter_widget_key = f"door_filter_widget_{file_id}"

                # Apply action-expected filter first (Confirm/Reject/Delete actions set this).
                # This prevents transient widget resets from affecting the filtered door list.
                try:
                    expected = st.session_state.pop(f"_door_filter_expected_{file_id}", None)
                except Exception:
                    expected = None
                if expected not in (None, ""):
                    try:
                        st.session_state[door_filter_key] = str(expected)
                        st.session_state[door_filter_widget_key] = str(expected)
                        fstate["_door_filter"] = str(expected)
                    except Exception:
                        pass

                # Apply user-driven filter changes (radio on_change sets these flags).
                user_changed_key = f"_door_filter_user_changed_{file_id}"
                try:
                    user_changed = bool(st.session_state.pop(user_changed_key, False))
                except Exception:
                    user_changed = False
                if user_changed:
                    try:
                        user_value = st.session_state.pop(f"_door_filter_user_value_{file_id}", None)
                    except Exception:
                        user_value = None
                    if user_value not in (None, ""):
                        try:
                            st.session_state[door_filter_key] = str(user_value)
                            fstate["_door_filter"] = str(user_value)
                        except Exception:
                            pass

                if door_filter_key not in st.session_state:
                    try:
                        st.session_state[door_filter_key] = str(fstate.get("_door_filter") or "All")
                    except Exception:
                        st.session_state[door_filter_key] = "All"
                selected_filter = str(st.session_state.get(door_filter_key) or "All")
                try:
                    fstate["_door_filter"] = str(selected_filter)
                except Exception:
                    pass

                # Map id -> type from the full (hidden-filtered) list for click-driven filter switching.
                id_to_type = {
                    str(d.get("id")): str(d.get("type") or "").strip()
                    for d in active_doors_all
                    if d.get("id") is not None
                }

                # If the user clicked a door of a different type while already type-filtered,
                # switch the filter to that type (but do not override Confirmed/Unconfirmed modes).
                click_sink_label = f"door_click_sink_{file_id}"
                clicked_id = st.session_state.get(click_sink_label)
                if clicked_id is not None and selected_filter not in ("All", "Confirmed", "Unconfirmed"):
                    clicked_type = id_to_type.get(str(clicked_id))
                    if clicked_type and clicked_type != selected_filter:
                        st.session_state[door_filter_key] = clicked_type
                        try:
                            st.session_state[door_filter_widget_key] = clicked_type
                        except Exception:
                            pass
                        selected_filter = clicked_type
                        push_breadcrumb(
                            fstate,
                            {
                                "kind": "auto_filter_switch_on_click",
                                "file_id": str(file_id),
                                "clicked_id": str(clicked_id),
                                "clicked_type": str(clicked_type),
                                "new_filter": str(selected_filter),
                            },
                        )

                if selected_filter == "Confirmed":
                    active_doors = [d for d in active_doors_all if str(d.get("id") or "") in confirmed_ids]
                elif selected_filter == "Unconfirmed":
                    active_doors = [d for d in active_doors_all if str(d.get("id") or "") not in confirmed_ids]
                elif selected_filter and selected_filter != "All":
                    active_doors = [
                        d for d in active_doors_all if str(d.get("type") or "").strip() == selected_filter
                    ]
                else:
                    active_doors = active_doors_all

                # If the user just performed a label action (Confirm/Reject/Delete), emit a
                # post-rerun snapshot so we can see what the *server* thinks is visible/styled.
                try:
                    raw = st.session_state.pop(f"_door_detector_last_label_action_{file_id}", None)
                except Exception:
                    raw = None
                if raw:
                    try:
                        act = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
                    except Exception:
                        act = {}
                    try:
                        act_id = str((act or {}).get("door_id") or "")
                    except Exception:
                        act_id = ""
                    try:
                        confirmed_ids_now = set(flatten_confirmed_ids(working.get("confirmed_by_type", {})))
                    except Exception:
                        confirmed_ids_now = set()
                    try:
                        hidden_ids_now = set(hidden_ids)
                    except Exception:
                        hidden_ids_now = set()
                    try:
                        active_ids_now = {str(d.get("id") or "") for d in active_doors if isinstance(d, dict)}
                    except Exception:
                        active_ids_now = set()
                    try:
                        all_ids_now = {str(d.get("id") or "") for d in active_doors_all if isinstance(d, dict)}
                    except Exception:
                        all_ids_now = set()

                    act_obj = None
                    try:
                        if act_id:
                            act_obj = next((d for d in active_doors_all if isinstance(d, dict) and str(d.get("id") or "") == act_id), None)
                    except Exception:
                        act_obj = None

                    ui_event_log(
                        "post_label_action_state",
                        {
                            "file_id": str(file_id),
                            "action": str((act or {}).get("action") or ""),
                            "door_id": act_id,
                            "door_filter": str(selected_filter),
                            "selected_door_id_fstate": str(fstate.get("selected_door_id") or ""),
                            "confirmed_len": int(len(confirmed_ids_now)),
                            "hidden_len": int(len(hidden_ids_now)),
                            "active_len": int(len(active_ids_now)),
                            "all_active_len": int(len(all_ids_now)),
                            "door_in_confirmed": bool(act_id and act_id in confirmed_ids_now),
                            "door_in_hidden": bool(act_id and act_id in hidden_ids_now),
                            "door_in_active": bool(act_id and act_id in active_ids_now),
                            "door_in_all_active": bool(act_id and act_id in all_ids_now),
                            "door_obj": (
                                {
                                    "id": str(act_obj.get("id") or ""),
                                    "type": str(act_obj.get("type") or ""),
                                    "bbox_xyxy": act_obj.get("bbox_xyxy"),
                                    "bbox_pdf_xyxy": act_obj.get("bbox_pdf_xyxy"),
                                    "confidence": act_obj.get("confidence", None),
                                }
                                if isinstance(act_obj, dict)
                                else None
                            ),
                        },
                    )

                # Sync selection state before rendering the viewer.
                all_visible = active_doors.copy()
                all_visible.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
                prev_sel_before_sync = str(fstate.get("selected_door_id") or "")
                _sync_selected_door_for_run(file_id=str(file_id), fstate=fstate, all_visible=all_visible)
                sel_after_sync = str(fstate.get("selected_door_id") or "")
                if sel_after_sync != prev_sel_before_sync:
                    push_breadcrumb(
                        fstate,
                        {
                            "kind": "selection_changed_by_sync",
                            "file_id": str(file_id),
                            "from": prev_sel_before_sync,
                            "to": sel_after_sync,
                            "visible_count": int(len(all_visible)),
                            "door_filter": str(selected_filter),
                        },
                    )

                # Warn if selection appears to fall out of the visible list (can happen
                # transiently with filter changes or stale ids).
                if sel_after_sync and sel_after_sync not in {str(d.get("id")) for d in all_visible if isinstance(d, dict)}:
                    k = f"desync_selected_not_in_visible::{file_id}::{sel_after_sync}::{selected_filter}"
                    if warn_once(fstate, k):
                        # Logging intentionally suppressed (noise). Breadcrumbs capture this state.
                        pass

                # --- Sync viewer-affecting widget state BEFORE rendering the viewer ---
                # Widgets live in the right panel, but their state is available at the start
                # of a rerun. Pull from `st.session_state` so the viewer reflects changes
                # immediately (same rerun), rather than one rerun later.
                auto_focus_key = f"auto_focus_{file_id}"
                if auto_focus_key not in st.session_state:
                    st.session_state[auto_focus_key] = bool(fstate.get("auto_focus", True))
                fstate["auto_focus"] = bool(st.session_state.get(auto_focus_key))

                # Highlight toggle (UI): when disabled, viewer should show no door overlays.
                highlight_key = f"highlight_doors_{file_id}"
                if highlight_key not in st.session_state:
                    st.session_state[highlight_key] = True
                fstate["highlight_doors"] = bool(st.session_state.get(highlight_key))
                fstate["viewer_display_mode"] = "Highlight All" if fstate["highlight_doors"] else "Off"

                # --- Sync draw-suggestion cycling state BEFORE rendering the viewer ---
                # Right panel writes these widget states; we compute the currently-cycled
                # candidate id here so the viewer can draw the snap highlight immediately.
                try:
                    srec = fstate.get("_last_draw_suggestions") or {}
                    suggestions = list(srec.get("suggestions") or [])
                except Exception:
                    suggestions = []
                idx_key = f"_draw_suggest_idx_{file_id}"
                type_key = f"_draw_suggest_type_{file_id}"
                type_touched_key = f"_draw_suggest_type_touched_{file_id}"
                try:
                    idx = int(st.session_state.get(idx_key) or 0)
                except Exception:
                    idx = 0
                chosen_type = str(st.session_state.get(type_key) or "All types")
                use_filter = bool(st.session_state.get(type_touched_key, False)) and (chosen_type != "All types")
                filtered = [s for s in suggestions if (not use_filter) or (str(s.get("type") or "") == chosen_type)]
                if not filtered and suggestions:
                    filtered = suggestions
                if idx < 0:
                    idx = 0
                if filtered and idx >= len(filtered):
                    idx = 0
                cycle_id = ""
                if filtered:
                    try:
                        cycle_id = str(filtered[idx].get("id") or "")
                    except Exception:
                        cycle_id = ""
                fstate["_cycle_candidate_id"] = cycle_id

                with viewer_slot:
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
                        full_dims=full_dims,
                        doors_data=doors_data,
                        fstate=fstate,
                        # Only show overlays for the filtered type when highlighting is enabled.
                        active_doors=(active_doors if fstate.get("highlight_doors") else []),
                        click_sink_label=click_sink_label,
                    )

                with review_slot:
                    main_viewer_controls(
                        selected_item,
                        full_dims=full_dims,
                        doors_data=doors_data,
                        fstate=fstate,
                    )
                    right_panel_review(
                        selected_item,
                        doors_data=doors_data,
                        fstate=fstate,
                        active_doors=active_doors,
                        all_active_doors=active_doors_all,
                    )

                # Run analysis after the UI has rendered so the previous viewer state persists
                # during the long-running pipeline work.
                if should_run_pipeline_now:
                    try:
                        run_pipeline(
                            str(file_id),
                            Path(str(task.get("file_dir") or str(file_dir))),
                            str(task.get("config_path") or _default_config_path_str()),
                        )
                    finally:
                        st.session_state.door_detector_pipeline_task = None
                    st.rerun()
            else:
                components.html(assets.sidebar_autopen_component_html(), height=1, scrolling=False)
                st.info("Select a file from the library to begin.")
        else:
            components.html(assets.sidebar_autopen_component_html(), height=1, scrolling=False)
            st.info("Select a file from the library to begin.")


