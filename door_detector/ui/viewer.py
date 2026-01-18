"""Main viewer (pan/zoom component) and related helpers."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from door_detector.ui.assets import sidebar_autopen_component_html
from door_detector.ui.labels import flatten_confirmed_ids, get_working_label_state as _get_working_label_state


logger = logging.getLogger("door_detector.review_app")

# Increase PIL pixel limit
Image.MAX_IMAGE_PIXELS = None

VIEWER_TARGET_WIDTH_PX = 1200
VIEWER_ASPECT_RATIO_HW = 0.75  # height/width


def _debug_log(msg: str, *args: Any) -> None:
    """Optional debug logging to the server console (disabled in the UI)."""
    try:
        if st.session_state.get("debug_perf"):
            logger.info(msg, *args)
    except Exception:
        return


def _coerce_bool(v: Any, *, default: bool = False) -> bool:
    """Best-effort boolean coercion for values coming from Streamlit/session_state."""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            return bool(int(v) != 0)
        except Exception:
            return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off", ""):
            return False
        return default
    return default


def _viewer_display_mode_to_sink_value(mode: str) -> str:
    """Map UI selection to a compact string the iframe JS can poll."""
    if mode == "Highlight Selected":
        return "selected"
    if mode == "Off":
        return "off"
    return "all"


def _normalize_bbox_xyxy(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return (x0, y0, x1, y1) with x0<=x1 and y0<=y1, or None if invalid."""
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _scale_bbox_xyxy(bbox_xyxy: List[float], scale: float) -> Optional[List[float]]:
    nb = _normalize_bbox_xyxy(bbox_xyxy)
    if nb is None:
        return None
    x0, y0, x1, y1 = nb
    return [x0 * scale, y0 * scale, x1 * scale, y1 * scale]


def _manual_overlay_payload_for_sink(
    *,
    fstate: Dict[str, Any],
    preview_scale: float,
) -> Dict[str, Any]:
    """Return preview-space overlays for the iframe.

    Important UX detail: overlays should reflect *this edit round only*.
    Previously-confirmed doors (including those confirmed via manual additions in earlier
    sessions) should not reappear as magenta/cyan "selection" overlays when the user
    re-enters Edit Doors without drawing anything new.
    """
    state = _get_working_label_state(fstate)
    out_manual: List[Dict[str, Any]] = []
    out_unmatched: List[Dict[str, Any]] = []

    if not (preview_scale > 0):
        preview_scale = 1.0

    # Only render manual additions created during the *current* edit session.
    # We do this by subtracting baseline manual additions (captured when entering edit mode)
    # from the current draft.
    baseline_counts: Counter = Counter()
    baseline_unmatched_counts: Counter = Counter()
    baseline = fstate.get("_edit_baseline") if bool(fstate.get("edit_mode")) else None
    if isinstance(baseline, dict):
        for b in list(baseline.get("manual_additions", []) or []):
            try:
                tok = json.dumps(b, sort_keys=True, separators=(",", ":"))
            except Exception:
                tok = repr(b)
            baseline_counts[tok] += 1
        for b in list(baseline.get("unmatched_manual_boxes", []) or []):
            try:
                tok = json.dumps(b, sort_keys=True, separators=(",", ":"))
            except Exception:
                tok = repr(b)
            baseline_unmatched_counts[tok] += 1

    for rec in list(state.get("manual_additions", [])):
        if baseline_counts:
            try:
                tok = json.dumps(rec, sort_keys=True, separators=(",", ":"))
            except Exception:
                tok = repr(rec)
            if baseline_counts.get(tok, 0) > 0:
                baseline_counts[tok] -= 1
                continue

        drawn_full = rec.get("drawn_bbox_xyxy")
        if not isinstance(drawn_full, list) or len(drawn_full) != 4:
            continue
        drawn_prev = _scale_bbox_xyxy([float(v) for v in drawn_full], preview_scale)
        if drawn_prev is None:
            continue
        snapped_prev = None
        snapped_full = rec.get("snapped_bbox_xyxy")
        if isinstance(snapped_full, list) and len(snapped_full) == 4:
            snapped_prev = _scale_bbox_xyxy([float(v) for v in snapped_full], preview_scale)
        out_manual.append(
            {
                "drawn_bbox_xyxy": drawn_prev,
                "snapped_bbox_xyxy": snapped_prev,
                "snapped_candidate_id": rec.get("snapped_candidate_id"),
                "iou": rec.get("iou"),
            }
        )

    # Render unmatched boxes created during the current edit session (if any).
    for rec in list(state.get("unmatched_manual_boxes", [])):
        if baseline_unmatched_counts:
            try:
                tok = json.dumps(rec, sort_keys=True, separators=(",", ":"))
            except Exception:
                tok = repr(rec)
            if baseline_unmatched_counts.get(tok, 0) > 0:
                baseline_unmatched_counts[tok] -= 1
                continue
        bb_full = rec.get("bbox_xyxy")
        if not isinstance(bb_full, list) or len(bb_full) != 4:
            continue
        bb_prev = _scale_bbox_xyxy([float(v) for v in bb_full], preview_scale)
        if bb_prev is None:
            continue
        out_unmatched.append(
            {
                "bbox_xyxy": bb_prev,
                "note": rec.get("note") or "unmatched",
            }
        )

    # Keep the key for backwards compatibility with the client code.
    return {"manual_additions": out_manual, "unmatched_manual_boxes": out_unmatched}


def _image_path_to_streamlit_url(image_path: str) -> str:
    """Return a URL (or small data URL fallback) for an on-disk image."""
    p = Path(image_path)
    try:
        img = Image.open(p)
        try:
            from streamlit.elements.lib.image_utils import image_to_url
            from streamlit.elements.lib.layout_utils import LayoutConfig

            st_image_id = f"img|{p}|{p.stat().st_mtime_ns}"
            url = image_to_url(
                image=img,
                layout_config=LayoutConfig(width=None),
                clamp=False,
                channels="RGB",
                output_format="JPEG",
                image_id=st_image_id,
            )
            if url:
                return url
        finally:
            try:
                img.close()
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: embed the preview (should be small).
    try:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def _rects_to_svg(
    *,
    active_doors: List[Dict[str, Any]],
    fstate: Dict[str, Any],
    scale: float,
    img_width: int,
    img_height: int,
) -> str:
    parts: List[str] = []

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    for d in active_doors:
        did = d.get("id")

        nb = _normalize_bbox_xyxy(d.get("bbox_xyxy"))
        if nb is None:
            continue
        x0, y0, x1, y1 = nb

        # Map full-res pixels → preview pixels.
        x0p = x0 * scale
        y0p = y0 * scale
        x1p = x1 * scale
        y1p = y1 * scale

        x0p = _clamp(x0p, 0.0, float(img_width))
        y0p = _clamp(y0p, 0.0, float(img_height))
        x1p = _clamp(x1p, 0.0, float(img_width))
        y1p = _clamp(y1p, 0.0, float(img_height))

        w = max(0.0, x1p - x0p)
        h = max(0.0, y1p - y0p)
        if w <= 0.0 or h <= 0.0:
            continue

        # Styling is applied client-side (confirmed/selected/deleted/view modes) so the
        # iframe does not need to remount for label-only state changes.
        stroke = "#ffa500"  # default (orange)
        stroke_width = 2
        did_attr = html.escape(str(did), quote=True)
        parts.append(
            f'<rect x="{x0p:.2f}" y="{y0p:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'vector-effect="non-scaling-stroke" '
            f'data-door-id="{did_attr}" '
            f'data-x="{x0p:.2f}" data-y="{y0p:.2f}" data-w="{w:.2f}" data-h="{h:.2f}" '
            f'style="pointer-events: all; cursor: pointer;" />'
        )

    return "\n".join(parts)


def _panzoom_image_viewer(
    *,
    img_src: str,
    img_width: int,
    img_height: int,
    rects_svg: str,
    height: int,
    key: str,
    click_sink_aria_label: str,
    selected_sink_aria_label: str,
    focus_seq_sink_aria_label: str,
    edit_mode_sink_aria_label: str,
    draw_event_sink_aria_label: str,
    manual_overlay_sink_aria_label: str,
    door_state_sink_aria_label: str,
    viewer_display_sink_aria_label: str,
    auto_focus_sink_aria_label: str,
    unmatched_debug_sink_aria_label: str,
    candidate_pool_sink_aria_label: str,
) -> None:
    # This viewer provides:
    # - scrollwheel zoom (centered at cursor)
    # - click+drag pan
    # - initial fit-to-container with letterboxing
    click_sink_aria_label_esc = html.escape(click_sink_aria_label, quote=True)
    selected_sink_aria_label_esc = html.escape(selected_sink_aria_label, quote=True)
    focus_seq_sink_aria_label_esc = html.escape(focus_seq_sink_aria_label, quote=True)
    edit_mode_sink_aria_label_esc = html.escape(edit_mode_sink_aria_label, quote=True)
    draw_event_sink_aria_label_esc = html.escape(draw_event_sink_aria_label, quote=True)
    manual_overlay_sink_aria_label_esc = html.escape(manual_overlay_sink_aria_label, quote=True)
    door_state_sink_aria_label_esc = html.escape(door_state_sink_aria_label, quote=True)
    viewer_display_sink_aria_label_esc = html.escape(viewer_display_sink_aria_label, quote=True)
    auto_focus_sink_aria_label_esc = html.escape(auto_focus_sink_aria_label, quote=True)
    unmatched_debug_sink_aria_label_esc = html.escape(unmatched_debug_sink_aria_label, quote=True)
    candidate_pool_sink_aria_label_esc = html.escape(candidate_pool_sink_aria_label, quote=True)
    viewer_html = f"""
{sidebar_autopen_component_html()}
<div id="pz_root_{key}" style="width: 100%; height: {height}px; overflow: hidden; background: #0e1117; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.12);">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
    }}
    /* Scoped to this component instance */
    #pz_root_{key} .pz-reset {{
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 5;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      background: rgba(17, 25, 40, 0.65);
      color: rgba(255, 255, 255, 0.92);
      font-size: 12px;
      line-height: 1;
      cursor: pointer;
      backdrop-filter: blur(6px);
      -webkit-backdrop-filter: blur(6px);
      box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35);
      opacity: 0;
      transform: translateY(-4px);
      transition: opacity 120ms ease, transform 120ms ease, border-color 120ms ease, background 120ms ease;
      user-select: none;
      pointer-events: none; /* enabled only when visible */
    }}
    #pz_root_{key} .pz-reset.pz-reset--visible {{
      opacity: 1;
      transform: translateY(0px);
      pointer-events: auto;
    }}
    #pz_root_{key} .pz-reset:hover {{
      background: rgba(17, 25, 40, 0.82);
      border-color: rgba(255, 255, 255, 0.26);
    }}
    #pz_root_{key} .pz-reset:active {{
      transform: translateY(0px) scale(0.98);
    }}
  </style>
  <div id="pz_stage_{key}" style="width: 100%; height: 100%; position: relative; user-select: none; cursor: grab; touch-action: none;">
    <button id="pz_reset_{key}" class="pz-reset" type="button" aria-label="Reset zoom" aria-hidden="true" tabindex="-1">Reset</button>
    <div id="pz_content_{key}" style="position: absolute; left: 0; top: 0; width: {int(img_width)}px; height: {int(img_height)}px; transform-origin: 0 0; will-change: transform;">
      <img
        id="pz_img_{key}"
        src="{img_src}"
        width="{int(img_width)}"
        height="{int(img_height)}"
        style="position: absolute; left: 0; top: 0; pointer-events: none;"
      />
      <svg
        id="pz_svg_{key}"
        width="{int(img_width)}"
        height="{int(img_height)}"
        viewBox="0 0 {int(img_width)} {int(img_height)}"
        style="position: absolute; left: 0; top: 0; pointer-events: auto;"
        xmlns="http://www.w3.org/2000/svg"
      >
        {rects_svg}
      </svg>
    </div>
  </div>
</div>

<script>
(function() {{
  const root = document.getElementById("pz_root_{key}");
  const stage = document.getElementById("pz_stage_{key}");
  const content = document.getElementById("pz_content_{key}");
  const img = document.getElementById("pz_img_{key}");
  const svg = document.getElementById("pz_svg_{key}");
  const resetBtn = document.getElementById("pz_reset_{key}");
  if (!root || !stage || !content || !img) return;

  const persistKey = "door_detector_pz_state_{html.escape(str(key), quote=True)}";
  const clickSinkLabel = "{click_sink_aria_label_esc}";
  const selectedSinkLabel = "{selected_sink_aria_label_esc}";
  const focusSeqSinkLabel = "{focus_seq_sink_aria_label_esc}";
  const editModeSinkLabel = "{edit_mode_sink_aria_label_esc}";
  const drawEventSinkLabel = "{draw_event_sink_aria_label_esc}";
  const manualOverlaySinkLabel = "{manual_overlay_sink_aria_label_esc}";
  const doorStateSinkLabel = "{door_state_sink_aria_label_esc}";
  const viewerDisplaySinkLabel = "{viewer_display_sink_aria_label_esc}";

  const autoFocusSinkLabel = "{auto_focus_sink_aria_label_esc}";
  const unmatchedDebugSinkLabel = "{unmatched_debug_sink_aria_label_esc}";
  const candidatePoolSinkLabel = "{candidate_pool_sink_aria_label_esc}";
  function parseBool(v, defaultValue) {{
    if (v === true) return true;
    if (v === false) return false;
    const s = String(v ?? "").trim().toLowerCase();
    if (s === "1" || s === "true" || s === "t" || s === "yes" || s === "y" || s === "on") return true;
    if (s === "0" || s === "false" || s === "f" || s === "no" || s === "n" || s === "off" || s === "") return false;
    return !!defaultValue;
  }}
  function getAutoFocus() {{
    return parseBool(readParentInputValue(autoFocusSinkLabel), true);
  }}
  let autoFocus = getAutoFocus();
  let lastUnmatchedDebugRaw = null;

  try {{
    console.log("[door_detector] pz init", {{
      key: {json.dumps(str(key))},
      autoFocus,
      persistKey,
      clickSinkLabel,
      selectedSinkLabel,
      focusSeqSinkLabel,
      editModeSinkLabel,
      drawEventSinkLabel,
      manualOverlaySinkLabel,
      doorStateSinkLabel,
      viewerDisplaySinkLabel,
      autoFocusSinkLabel,
      unmatchedDebugSinkLabel,
      candidatePoolSinkLabel,
      ts: Date.now(),
    }});
  }} catch (_) {{}}

  let scale = 1;
  let tx = 0;
  let ty = 0;
  // Track last-seen focus sequence from Streamlit so pan/zoom persistence can
  // also persist which selection-focus we've already applied (prevents refocus
  // on reruns).
  let focusSeq = 0;

  // "Original state" = initial fit-to-container transform.
  let baseScale = 1;
  let baseTx = 0;
  let baseTy = 0;

  let dragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartTx = 0;
  let dragStartTy = 0;

  function clamp(v, lo, hi) {{
    return Math.max(lo, Math.min(hi, v));
  }}

  function applyTransform() {{
    content.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
    updateResetVisibility();
    scheduleSaveState();
  }}

  function isAtBase() {{
    const eps = 0.5; // pixels for tx/ty (and ~0.5 for scale*1000 below)
    const scaleClose = Math.abs(scale - baseScale) < 0.0005;
    const txClose = Math.abs(tx - baseTx) < eps;
    const tyClose = Math.abs(ty - baseTy) < eps;
    return scaleClose && txClose && tyClose;
  }}

  function updateResetVisibility() {{
    if (!resetBtn) return;
    const visible = !isAtBase();
    if (visible) {{
      resetBtn.classList.add("pz-reset--visible");
      resetBtn.setAttribute("aria-hidden", "false");
      resetBtn.tabIndex = 0;
    }} else {{
      resetBtn.classList.remove("pz-reset--visible");
      resetBtn.setAttribute("aria-hidden", "true");
      resetBtn.tabIndex = -1;
    }}
  }}

  function fitToContainer() {{
    const cw = root.clientWidth;
    const ch = root.clientHeight;
    // Use the known image pixel dimensions passed from Python.
    // Relying on `img.width` can be wrong if CSS/layout clamps the image.
    const iw = {int(img_width)};
    const ih = {int(img_height)};
    if (!cw || !ch || !iw || !ih) return;

    // Initial zoom should be the MAX zoom that keeps the entire image visible:
    // fit-to-container in BOTH directions (no initial cropping on extreme aspect ratios).
    const pad = 6; // subtle breathing room against the rounded border
    const cwPad = cw - pad * 2;
    const chPad = ch - pad * 2;
    if (cwPad <= 0 || chPad <= 0) return;
    scale = clamp(Math.min(cwPad / iw, chPad / ih), 0.05, 20);
    tx = (cw - iw * scale) / 2;
    ty = (ch - ih * scale) / 2;

    baseScale = scale;
    baseTx = tx;
    baseTy = ty;
    applyTransform();
  }}

  function loadState() {{
    try {{
      const raw = sessionStorage.getItem(persistKey);
      if (!raw) return null;
      const obj = JSON.parse(raw);
      if (!obj) return null;
      if (!Number.isFinite(obj.tx) || !Number.isFinite(obj.ty) || !Number.isFinite(obj.scale)) return null;
      return obj;
    }} catch (_) {{
      return null;
    }}
  }}

  function readParentInputValue(label) {{
    try {{
      const input = window.parent?.document?.querySelector(`input[aria-label="${{label}}"]`);
      return input ? String(input.value || "") : "";
    }} catch (_) {{
      return "";
    }}
  }}

  function setParentInputValue(label, value) {{
    try {{
      const doc = window.parent?.document;
      if (!doc || !label) return false;
      let input = null;
      try {{
        input = doc.querySelector(`input[aria-label="${{label}}"]`);
      }} catch (_) {{
        input = null;
      }}
      if (!input) {{
        // Fallback: Streamlit sometimes mutates aria-label; allow prefix match.
        try {{
          input = doc.querySelector(`input[aria-label^="${{label}}"]`);
        }} catch (_) {{
          input = null;
        }}
      }}
      if (!input) return false;

      // Use the native setter so React/Streamlit observe the change.
      const v = String(value);
      const prev = String(input.value || "");
      // Some Streamlit builds may only commit widget state on focused inputs.
      try {{ input.focus({{ preventScroll: true }}); }} catch (_) {{}}
      // Streamlit's frontend is React; resetting the value tracker makes "input"
      // events reliably propagate widget state updates.
      let hadTracker = false;
      try {{
        const tracker = input._valueTracker;
        if (tracker && tracker.setValue) {{
          hadTracker = true;
          tracker.setValue(prev);
        }}
      }} catch (_) {{}}
      try {{
        // IMPORTANT: dispatch events created from the *parent realm* so React's
        // event system sees them (cross-iframe Event instances can be ignored).
        const evWin = input.ownerDocument && input.ownerDocument.defaultView ? input.ownerDocument.defaultView : (window.parent || window);
        const setter = Object.getOwnPropertyDescriptor(evWin.HTMLInputElement.prototype, "value")?.set;
        if (setter) setter.call(input, v);
        else input.value = v;
      }} catch (_) {{
        input.value = v;
      }}

      // Dispatch events from the parent realm (see note above).
      let inputEvtOk = false;
      let changeEvtOk = false;
      try {{
        const evWin = input.ownerDocument && input.ownerDocument.defaultView ? input.ownerDocument.defaultView : (window.parent || window);
        const InputEv = evWin.InputEvent || evWin.Event;
        const inputEvt = evWin.InputEvent
          ? new evWin.InputEvent("input", {{ bubbles: true, cancelable: true, data: v, inputType: "insertReplacementText" }})
          : new InputEv("input", {{ bubbles: true, cancelable: true }});
        inputEvtOk = !!input.dispatchEvent(inputEvt);
        const changeEvt = new evWin.Event("change", {{ bubbles: true, cancelable: true }});
        changeEvtOk = !!input.dispatchEvent(changeEvt);
      }} catch (_) {{
        // Fallback (same-realm) if the above fails for any reason.
        try {{ inputEvtOk = !!input.dispatchEvent(new Event("input", {{ bubbles: true }})); }} catch (_) {{}}
        try {{ changeEvtOk = !!input.dispatchEvent(new Event("change", {{ bubbles: true }})); }} catch (_) {{}}
      }}
      try {{
        window.__door_detectorLastSinkWrite = {{
          ok: true,
          label,
          ariaLabel: input.getAttribute ? input.getAttribute("aria-label") : null,
          prev,
          next: v,
          after: String(input.value || ""),
          hadTracker,
          inputEvtOk,
          changeEvtOk,
          ts: Date.now(),
        }};
      }} catch (_) {{}}
      try {{ input.blur(); }} catch (_) {{}}
      return true;
    }} catch (_) {{
      try {{
        window.__door_detectorLastSinkWrite = {{ ok: false, label, ts: Date.now(), err: "exception" }};
      }} catch (_) {{}}
      return false;
    }}
  }}

  function readParentJson(label) {{
    try {{
      const raw = readParentInputValue(label);
      if (!raw) return null;
      return JSON.parse(raw);
    }} catch (_) {{
      return null;
    }}
  }}

  function getSelectedId() {{
    const v = readParentInputValue(selectedSinkLabel);
    return v ? v : null;
  }}

  function getFocusSeq() {{
    const raw = readParentInputValue(focusSeqSinkLabel);
    const n = parseInt(raw || "0", 10);
    return Number.isFinite(n) ? n : 0;
  }}

  function findRectByDoorId(doorId) {{
    if (!svg || !doorId) return null;
    const rects = svg.querySelectorAll("rect[data-door-id]");
    for (const r of rects) {{
      if (r && r.getAttribute && r.getAttribute("data-door-id") === doorId) return r;
    }}
    return null;
  }}

  function getEditEnabled() {{
    return readParentInputValue(editModeSinkLabel) === "1";
  }}

  // In Edit Doors mode, we intentionally suppress auto-focus so the view doesn't
  // keep yanking/zooming while the user is trying to pan/zoom and manipulate boxes.
  // (Selection highlight + right-panel preview still update as normal.)
  function getEffectiveAutoFocus() {{
    return getAutoFocus() && !getEditEnabled();
  }}

  function getViewerDisplayMode() {{
    const v = readParentInputValue(viewerDisplaySinkLabel);
    return v ? v : "all";
  }}

  let lastDoorStateRaw = null;
  let confirmedSet = new Set();
  let deletedSet = new Set();
  let lastViewerDisplay = null;
  let lastEditEnabled = null;
  let localSelectedId = null;

  function updateDoorStateFromSinks() {{
    const raw = readParentInputValue(doorStateSinkLabel);
    if (raw === lastDoorStateRaw) return false;
    lastDoorStateRaw = raw;
    const obj = readParentJson(doorStateSinkLabel) || {{}};
    const c = Array.isArray(obj.confirmed_ids) ? obj.confirmed_ids : [];
    const d = Array.isArray(obj.deleted_ids) ? obj.deleted_ids : [];
    confirmedSet = new Set(c.map(String));
    deletedSet = new Set(d.map(String));
    return true;
  }}

  function applyDoorStyles() {{
    if (!svg) return;
    // Prefer local selection for immediate feedback; Streamlit selection takes over after rerun.
    const selectedId = localSelectedId || getSelectedId();
    const viewMode = getViewerDisplayMode();

    const rects = svg.querySelectorAll("rect[data-door-id]");
    try {{
      if (!applyDoorStyles._loggedOnce) {{
        applyDoorStyles._loggedOnce = true;
        const first = rects && rects.length ? rects[0] : null;
        let bb = null;
        try {{ bb = first ? first.getBBox() : null; }} catch (_) {{ bb = null; }}

        // Compute bbox range across all candidates (helps spot coord mismatches).
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        if (rects) {{
          for (const rr of rects) {{
            let b = null;
            try {{ b = rr.getBBox(); }} catch (_) {{ b = null; }}
            if (!b) continue;
            const x0 = Number(b.x), y0 = Number(b.y);
            const x1 = x0 + Number(b.width), y1 = y0 + Number(b.height);
            if (!Number.isFinite(x0) || !Number.isFinite(y0) || !Number.isFinite(x1) || !Number.isFinite(y1)) continue;
            minX = Math.min(minX, x0);
            minY = Math.min(minY, y0);
            maxX = Math.max(maxX, x1);
            maxY = Math.max(maxY, y1);
          }}
        }}

        const payload = {{
          count: rects ? rects.length : 0,
          firstId: first ? first.getAttribute("data-door-id") : null,
          firstBBox: bb ? {{ x: Number(bb.x), y: Number(bb.y), w: Number(bb.width), h: Number(bb.height) }} : null,
          candRange: (minX !== Infinity) ? [minX, minY, maxX, maxY] : null,
          svgViewBox: svg ? svg.getAttribute("viewBox") : null,
        }};
        console.log("[door_detector] applyDoorStyles rects", payload);
        console.log("[door_detector] applyDoorStyles rects json", JSON.stringify(payload));
      }}
    }} catch (_) {{}}
    for (const r of rects) {{
      const did = r.getAttribute("data-door-id");
      if (!did) continue;

      const isSelected = selectedId && did === selectedId;
      const isDeleted = deletedSet.has(did);

      let visible = true;
      if (viewMode === "off") visible = false;
      else if (viewMode === "selected") visible = !!isSelected;
      if (isDeleted) visible = false;

      if (!visible) {{
        r.style.display = "none";
        r.style.pointerEvents = "none";
        continue;
      }}

      r.style.display = "";
      r.style.pointerEvents = "all";

      if (isSelected) {{
        r.setAttribute("stroke", "#ff4b4b");
        r.setAttribute("stroke-width", "3");
        try {{ svg.appendChild(r); }} catch (_) {{}}
      }} else if (confirmedSet.has(did)) {{
        r.setAttribute("stroke", "#00ff00");
        r.setAttribute("stroke-width", "2");
      }} else {{
        r.setAttribute("stroke", "#ffa500");
        r.setAttribute("stroke-width", "2");
      }}
    }}
  }}

  let saveTimer = null;
  function scheduleSaveState() {{
    try {{
      if (saveTimer) clearTimeout(saveTimer);
      saveTimer = setTimeout(() => {{
        saveTimer = null;
        try {{
          sessionStorage.setItem(
            persistKey,
            JSON.stringify({{ tx: tx, ty: ty, scale: scale, focusSeq: focusSeq }})
          );
        }} catch (_) {{}}
      }}, 100);
    }} catch (_) {{}}
  }}

  function easeInOut(t) {{
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }}

  function animateTo(targetTx, targetTy, targetScale, durationMs) {{
    const start = performance.now();
    const sTx = tx, sTy = ty, sScale = scale;
    const dTx = targetTx - sTx;
    const dTy = targetTy - sTy;
    const dScale = targetScale - sScale;

    function step(now) {{
      const t = clamp((now - start) / durationMs, 0, 1);
      const e = easeInOut(t);
      tx = sTx + dTx * e;
      ty = sTy + dTy * e;
      scale = sScale + dScale * e;
      applyTransform();
      if (t < 1) requestAnimationFrame(step);
    }}

    requestAnimationFrame(step);
  }}

  function focusToBBox(bbox) {{
    if (!bbox || !Array.isArray(bbox) || bbox.length !== 4) return;
    const cw = root.clientWidth;
    const ch = root.clientHeight;
    if (!cw || !ch) return;

    const x0 = bbox[0], y0 = bbox[1], x1 = bbox[2], y1 = bbox[3];
    const bx0 = Math.min(x0, x1);
    const by0 = Math.min(y0, y1);
    const bx1 = Math.max(x0, x1);
    const by1 = Math.max(y0, y1);

    const bw = Math.max(1, bx1 - bx0);
    const bh = Math.max(1, by1 - by0);
    const cx = (bx0 + bx1) / 2;
    const cy = (by0 + by1) / 2;

    // Show some context around the selected door (larger = less zoomed-in).
    const padFactor = 3.0;
    const targetScale = clamp(Math.min(cw / (bw * padFactor), ch / (bh * padFactor)), baseScale, baseScale * 6);
    const targetTx = (cw / 2) - targetScale * cx;
    const targetTy = (ch / 2) - targetScale * cy;

    try {{
      console.log("[door_detector] pz focusToBBox", {{
        key: {json.dumps(str(key))},
        focusSeq,
        bbox,
        target: {{ tx: targetTx, ty: targetTy, scale: targetScale }},
        ts: Date.now(),
      }});
    }} catch (_) {{}}
    animateTo(targetTx, targetTy, targetScale, 260);
  }}

  function focusToDoorId(doorId) {{
    const r = findRectByDoorId(doorId);
    if (!r) return;
    const x = parseFloat(r.getAttribute("data-x") || r.getAttribute("x") || "0");
    const y = parseFloat(r.getAttribute("data-y") || r.getAttribute("y") || "0");
    const w = parseFloat(r.getAttribute("data-w") || r.getAttribute("width") || "0");
    const h = parseFloat(r.getAttribute("data-h") || r.getAttribute("height") || "0");
    if (!(w > 0) || !(h > 0)) return;
    focusToBBox([x, y, x + w, y + h]);
  }}

  function zoomAt(clientX, clientY, zoomFactor) {{
    const rect = root.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;

    const oldScale = scale;
    scale = clamp(scale * zoomFactor, 0.05, 20);

    // Keep the point under the cursor stable:
    // p = t + s * q  => q = (p - t)/s
    const qx = (px - tx) / oldScale;
    const qy = (py - ty) / oldScale;
    tx = px - scale * qx;
    ty = py - scale * qy;
    applyTransform();
  }}

  // Fit once image is ready.
  function applyInitialView() {{
    // Always establish base fit first.
    fitToContainer();

    // Restore last view first so focus can animate from the current pan/zoom.
    const saved = loadState();
    if (saved) {{
      try {{
        console.log("[door_detector] pz restoreState", {{
          key: {json.dumps(str(key))},
          saved,
          ts: Date.now(),
        }});
      }} catch (_) {{}}
      scale = clamp(saved.scale, 0.05, 20);
      tx = saved.tx;
      ty = saved.ty;
      applyTransform();
    }}

    // Sync highlight and optional focus based on parent state.
    const doorId = getSelectedId();
    const seq = getFocusSeq();
    focusSeq = seq;
    updateDoorStateFromSinks();
    applyDoorStyles();
    renderManualOverlays();
    const effectiveAutoFocus = getEffectiveAutoFocus();
    if (doorId && effectiveAutoFocus && (!saved || saved.focusSeq !== seq)) {{
      try {{
        console.log("[door_detector] pz autoFocus", {{
          key: {json.dumps(str(key))},
          fromSaved: !!saved,
          savedFocusSeq: saved ? saved.focusSeq : null,
          focusSeq: seq,
          ts: Date.now(),
        }});
      }} catch (_) {{}}
      focusToDoorId(doorId);
    }}
  }}

  if (img.complete) applyInitialView();
  else img.addEventListener("load", applyInitialView, {{ once: true }});

  // Keep the "original state" in sync on resize, but only if user hasn't deviated.
  try {{
    const ro = new ResizeObserver(() => {{
      if (isAtBase()) fitToContainer();
    }});
    ro.observe(root);
  }} catch (_) {{}}

  if (resetBtn) {{
    // Prevent pan-drag initiation when clicking the reset button.
    resetBtn.addEventListener("pointerdown", (e) => {{
      e.preventDefault();
      e.stopPropagation();
    }});
    resetBtn.addEventListener("click", (e) => {{
      e.preventDefault();
      e.stopPropagation();
      // Reset back to original fit-to-container for current container size.
      fitToContainer();
    }});
  }}

  root.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomAt(e.clientX, e.clientY, zoomFactor);
  }}, {{ passive: false }});

  // Manual overlays (server-supplied) and draw overlays (client-only until rerun).
  //
  // NOTE: These are referenced from applyInitialView(), which can run before this
  // section executes. Use `var` (function-scoped, hoisted) to avoid TDZ errors.
  var manualLayer = null;
  var tempLayer = null;
  var lastManualOverlayRaw = null;
  var suppressSvgClickUntil = 0;

  function ensureLayer(id) {{
    if (!svg) return null;
    let g = null;
    try {{ g = svg.querySelector(`#${{id}}`); }} catch (_) {{ g = null; }}
    if (g) return g;
    try {{
      g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("id", id);
      // Make edit overlays crisp (avoid fuzzy antialiased edges).
      g.setAttribute("shape-rendering", "crispEdges");
      svg.appendChild(g);
      return g;
    }} catch (_) {{
      return null;
    }}
  }}

  function clearLayer(g) {{
    if (!g) return;
    try {{
      while (g.firstChild) g.removeChild(g.firstChild);
    }} catch (_) {{}}
  }}

  function drawBox(layer, bbox, stroke, strokeWidth, dashArray, opacity, titleText) {{
    if (!layer || !bbox || !Array.isArray(bbox) || bbox.length !== 4) return;
    // Quantize to half-pixel boundaries for crisper strokes.
    const q = (v) => Math.round(Number(v) * 2) / 2;
    const x0 = q(Math.min(bbox[0], bbox[2]));
    const y0 = q(Math.min(bbox[1], bbox[3]));
    const x1 = q(Math.max(bbox[0], bbox[2]));
    const y1 = q(Math.max(bbox[1], bbox[3]));
    const w = q(Math.max(0, x1 - x0));
    const h = q(Math.max(0, y1 - y0));
    if (!(w > 0) || !(h > 0)) return;

    let r = null;
    try {{
      r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", String(x0));
      r.setAttribute("y", String(y0));
      r.setAttribute("width", String(w));
      r.setAttribute("height", String(h));
      r.setAttribute("fill", "none");
      r.setAttribute("stroke", stroke || "#00ffff");
      r.setAttribute("stroke-width", String(strokeWidth || 2));
      r.setAttribute("stroke-linecap", "square");
      r.setAttribute("shape-rendering", "crispEdges");
      r.setAttribute("vector-effect", "non-scaling-stroke");
      if (dashArray) r.setAttribute("stroke-dasharray", String(dashArray));
      // Use stroke-opacity instead of overall opacity to avoid fuzzy edges.
      if (Number.isFinite(opacity)) r.setAttribute("stroke-opacity", String(opacity));
      r.style.pointerEvents = "none";
      if (titleText) {{
        const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
        t.textContent = String(titleText);
        r.appendChild(t);
      }}
      layer.appendChild(r);
    }} catch (_) {{}}
  }}

  function computeIoU(a, b) {{
    if (!a || !b || a.length !== 4 || b.length !== 4) return 0;
    const ax0 = Math.min(a[0], a[2]), ay0 = Math.min(a[1], a[3]);
    const ax1 = Math.max(a[0], a[2]), ay1 = Math.max(a[1], a[3]);
    const bx0 = Math.min(b[0], b[2]), by0 = Math.min(b[1], b[3]);
    const bx1 = Math.max(b[0], b[2]), by1 = Math.max(b[1], b[3]);
    const ix0 = Math.max(ax0, bx0);
    const iy0 = Math.max(ay0, by0);
    const ix1 = Math.min(ax1, bx1);
    const iy1 = Math.min(ay1, by1);
    const iw = Math.max(0, ix1 - ix0);
    const ih = Math.max(0, iy1 - iy0);
    const inter = iw * ih;
    const aa = Math.max(0, ax1 - ax0) * Math.max(0, ay1 - ay0);
    const ba = Math.max(0, bx1 - bx0) * Math.max(0, by1 - by0);
    const denom = aa + ba - inter;
    if (!(denom > 0)) return 0;
    return inter / denom;
  }}

  function computeIntersectionArea(a, b) {{
    if (!a || !b || a.length !== 4 || b.length !== 4) return 0;
    const ax0 = Math.min(a[0], a[2]), ay0 = Math.min(a[1], a[3]);
    const ax1 = Math.max(a[0], a[2]), ay1 = Math.max(a[1], a[3]);
    const bx0 = Math.min(b[0], b[2]), by0 = Math.min(b[1], b[3]);
    const bx1 = Math.max(b[0], b[2]), by1 = Math.max(b[1], b[3]);
    const ix0 = Math.max(ax0, bx0);
    const iy0 = Math.max(ay0, by0);
    const ix1 = Math.min(ax1, bx1);
    const iy1 = Math.min(ay1, by1);
    const iw = Math.max(0, ix1 - ix0);
    const ih = Math.max(0, iy1 - iy0);
    return iw * ih;
  }}

  function bboxCenter(b) {{
    return {{ x: (b[0] + b[2]) / 2, y: (b[1] + b[3]) / 2 }};
  }}

  function snapCandidateForDraw(drawn) {{
    // Prefer a broader candidate pool (server-supplied), otherwise fall back to
    // the rendered overlay rects.
    const poolRaw = readParentInputValue(candidatePoolSinkLabel);
    const poolObj = readParentJson(candidatePoolSinkLabel);
    const pool = poolObj && Array.isArray(poolObj.candidates) ? poolObj.candidates : null;
    const hasPool = !!(pool && pool.length);

    if (!svg) return null;
    const rects = svg.querySelectorAll("rect[data-door-id]");
    if ((!rects || rects.length === 0) && !hasPool) return null;

    const x0 = Math.min(drawn[0], drawn[2]);
    const y0 = Math.min(drawn[1], drawn[3]);
    const x1 = Math.max(drawn[0], drawn[2]);
    const y1 = Math.max(drawn[1], drawn[3]);
    const norm = [x0, y0, x1, y1];
    const dw = Math.max(1, x1 - x0);
    const dh = Math.max(1, y1 - y0);
    const dc = bboxCenter(norm);

    let best = null;
    let bestIou = -1;
    const centersInside = [];
    const centersAll = [];
    const overlap = []; // candidates with inter > 0 (i.e. actually inside selection)
    let bestCoverage = -1;
    let bestByCoverage = null;
    let bestInter = -1;
    let bestByInter = null;

    const iter = hasPool ? pool : rects;
    for (const r of iter) {{
      let did = null;
      let cand = null;
      if (hasPool) {{
        did = r && r.id ? String(r.id) : null;
        cand = r && Array.isArray(r.bbox_xyxy) ? r.bbox_xyxy : null;
      }} else {{
        did = r.getAttribute("data-door-id");
        if (!did) continue;
        // Use SVG geometry instead of relying on attributes (which can be missing/NaN).
        let bb = null;
        try {{ bb = r.getBBox(); }} catch (_) {{ bb = null; }}
        if (!bb) continue;
        const rw = Number(bb.width);
        const rh = Number(bb.height);
        const rx = Number(bb.x);
        const ry = Number(bb.y);
        if (!(rw > 0) || !(rh > 0) || !Number.isFinite(rx) || !Number.isFinite(ry)) continue;
        cand = [rx, ry, rx + rw, ry + rh];
      }}
      if (!did || !cand || cand.length !== 4) continue;
      const rx0 = Number(cand[0]), ry0 = Number(cand[1]), rx1 = Number(cand[2]), ry1 = Number(cand[3]);
      if (!Number.isFinite(rx0) || !Number.isFinite(ry0) || !Number.isFinite(rx1) || !Number.isFinite(ry1)) continue;
      cand = [rx0, ry0, rx1, ry1];
      const iou = computeIoU(norm, cand);
      const inter = computeIntersectionArea(norm, cand);
      const candArea = Math.max(0, (cand[2] - cand[0])) * Math.max(0, (cand[3] - cand[1]));
      const coverage = candArea > 0 ? (inter / candArea) : 0; // how much of candidate is inside drawn
      // Only consider candidates that overlap the selection for snapping.
      // This prevents snapping to a nearby candidate outside the drawn box.
      if (inter > 0) {{
        overlap.push({{ id: did, bbox: cand, iou: iou, inter: inter, coverage: coverage }});
        if (iou > bestIou) {{
          bestIou = iou;
          best = {{ id: did, bbox: cand, iou: iou }};
        }}
        if (coverage > bestCoverage) {{
          bestCoverage = coverage;
          bestByCoverage = {{ id: did, bbox: cand, iou: iou, inter: inter, coverage: coverage }};
        }}
        if (inter > bestInter) {{
          bestInter = inter;
          bestByInter = {{ id: did, bbox: cand, iou: iou, inter: inter, coverage: coverage }};
        }}
      }}
      const cc = bboxCenter(cand);
      const dist = Math.hypot(cc.x - dc.x, cc.y - dc.y);
      const rec = {{ id: did, bbox: cand, iou: iou, dist: dist }};
      centersAll.push(rec);
      if (cc.x >= x0 && cc.x <= x1 && cc.y >= y0 && cc.y <= y1) centersInside.push(rec);
    }}

    // Snapping only considers candidates that overlap the selection.
    const MIN_SNAP_IOU = 0.02; // selector rectangles can be larger than candidates
    const MIN_CAND_COVERAGE = 0.20;
    try {{
      const closest = centersAll.length ? centersAll.slice().sort((a, b) => (a.dist - b.dist) || (b.iou - a.iou))[0] : null;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      let sample = [];
      for (let i = 0; i < Math.min(3, overlap.length); i++) {{
        const o = overlap[i];
        sample.push({{ id: o.id, bbox: o.bbox, iou: o.iou, inter: o.inter, coverage: o.coverage }});
      }}
      // Compute candidate coordinate range (using the same source we’re snapping against).
      if (hasPool) {{
        for (const rr of pool) {{
          if (!rr || !Array.isArray(rr.bbox_xyxy) || rr.bbox_xyxy.length !== 4) continue;
          const rx0 = Number(rr.bbox_xyxy[0]), ry0 = Number(rr.bbox_xyxy[1]);
          const rx1 = Number(rr.bbox_xyxy[2]), ry1 = Number(rr.bbox_xyxy[3]);
          if (!Number.isFinite(rx0) || !Number.isFinite(ry0) || !Number.isFinite(rx1) || !Number.isFinite(ry1)) continue;
          minX = Math.min(minX, rx0);
          minY = Math.min(minY, ry0);
          maxX = Math.max(maxX, rx1);
          maxY = Math.max(maxY, ry1);
        }}
      }} else {{
        for (const rr of rects) {{
          let bb = null;
          try {{ bb = rr.getBBox(); }} catch (_) {{ bb = null; }}
          if (!bb) continue;
          const rx0 = Number(bb.x), ry0 = Number(bb.y);
          const rx1 = rx0 + Number(bb.width), ry1 = ry0 + Number(bb.height);
          if (!Number.isFinite(rx0) || !Number.isFinite(ry0) || !Number.isFinite(rx1) || !Number.isFinite(ry1)) continue;
          minX = Math.min(minX, rx0);
          minY = Math.min(minY, ry0);
          maxX = Math.max(maxX, rx1);
          maxY = Math.max(maxY, ry1);
        }}
      }}

      // Also compute the nearest candidate to the drawn box (even if no overlap).
      let nearest = null;
      let bestDist = Infinity;
      for (const rr of rects) {{
        const did = rr.getAttribute("data-door-id");
        if (!did) continue;
        let bb = null;
        try {{ bb = rr.getBBox(); }} catch (_) {{ bb = null; }}
        if (!bb) continue;
        const rx0 = Number(bb.x), ry0 = Number(bb.y);
        const rx1 = rx0 + Number(bb.width), ry1 = ry0 + Number(bb.height);
        if (!Number.isFinite(rx0) || !Number.isFinite(ry0) || !Number.isFinite(rx1) || !Number.isFinite(ry1)) continue;
        const cand = [rx0, ry0, rx1, ry1];
        const cc = bboxCenter(cand);
        const dist = Math.hypot(cc.x - dc.x, cc.y - dc.y);
        if (dist < bestDist) {{
          bestDist = dist;
          nearest = {{ id: did, bbox: cand, dist }};
        }}
      }}

      // Compute max intersection among all candidates (diagnostic).
      let maxInterAny = 0;
      let maxInterAnyId = null;
      let maxInterAnyBBox = null;
      for (const rr of rects) {{
        const did = rr.getAttribute("data-door-id");
        if (!did) continue;
        let bb = null;
        try {{ bb = rr.getBBox(); }} catch (_) {{ bb = null; }}
        if (!bb) continue;
        const rx0 = Number(bb.x), ry0 = Number(bb.y);
        const rx1 = rx0 + Number(bb.width), ry1 = ry0 + Number(bb.height);
        if (!Number.isFinite(rx0) || !Number.isFinite(ry0) || !Number.isFinite(rx1) || !Number.isFinite(ry1)) continue;
        const cand = [rx0, ry0, rx1, ry1];
        const inter = computeIntersectionArea(norm, cand);
        if (inter > maxInterAny) {{
          maxInterAny = inter;
          maxInterAnyId = did;
          maxInterAnyBBox = cand;
        }}
      }}

      console.log("[door_detector] snapCandidateForDraw", {{
        drawn: norm,
        drawnWH: [dw, dh],
        candRange: (minX !== Infinity) ? [minX, minY, maxX, maxY] : null,
        source: hasPool ? "pool" : "svg",
        poolRawBytes: poolRaw ? poolRaw.length : 0,
        poolCandidates: hasPool ? pool.length : 0,
        candidates: hasPool ? pool.length : rects.length,
        overlapCandidates: overlap.length,
        bestByIoU: best ? {{ reason: "iou", id: best.id, iou: best.iou }} : null,
        bestByCoverage: bestByCoverage ? {{ reason: "coverage", id: bestByCoverage.id, coverage: bestByCoverage.coverage, iou: bestByCoverage.iou, inter: bestByCoverage.inter }} : null,
        bestByInter: bestByInter ? {{ reason: "max_intersection", id: bestByInter.id, inter: bestByInter.inter, coverage: bestByInter.coverage, iou: bestByInter.iou }} : null,
        centersInside: centersInside.length,
        closest: closest ? {{ id: closest.id, dist: closest.dist, iou: closest.iou }} : null,
        thresholds: {{ MIN_SNAP_IOU, MIN_CAND_COVERAGE }},
        overlapSample: sample,
        nearestCandidate: nearest,
        maxInterAny: {{ id: maxInterAnyId, inter: maxInterAny, bbox: maxInterAnyBBox }},
      }});
      console.log(
        "[door_detector] snapCandidateForDraw json",
        JSON.stringify(
          {{
            drawn: norm,
            drawnWH: [dw, dh],
            candRange: (minX !== Infinity) ? [minX, minY, maxX, maxY] : null,
            source: hasPool ? "pool" : "svg",
            poolRawBytes: poolRaw ? poolRaw.length : 0,
            poolCandidates: hasPool ? pool.length : 0,
            candidates: hasPool ? pool.length : rects.length,
            overlapCandidates: overlap.length,
            bestByIoU: best ? {{ id: best.id, iou: best.iou }} : null,
            bestByCoverage: bestByCoverage ? {{ id: bestByCoverage.id, coverage: bestByCoverage.coverage, iou: bestByCoverage.iou, inter: bestByCoverage.inter }} : null,
            bestByInter: bestByInter ? {{ id: bestByInter.id, inter: bestByInter.inter, coverage: bestByInter.coverage, iou: bestByInter.iou }} : null,
            closest: closest ? {{ id: closest.id, dist: closest.dist, iou: closest.iou }} : null,
            thresholds: {{ MIN_SNAP_IOU, MIN_CAND_COVERAGE }},
            nearestCandidate: nearest,
            maxInterAny: {{ id: maxInterAnyId, inter: maxInterAny, bbox: maxInterAnyBBox }},
          }}
        )
      );
    }} catch (_) {{}}

    if (!overlap.length) {{
      try {{ console.log("[door_detector] snapCandidateForDraw no match (no overlap)"); }} catch (_) {{}}
      return null;
    }}

    if (best && best.iou >= MIN_SNAP_IOU) {{
      try {{ console.log("[door_detector] snapCandidateForDraw chosen", {{ reason: "iou", id: best.id, iou: best.iou }}); }} catch (_) {{}}
      return best;
    }}

    // If IoU is too low, fall back to maximum overlap (intersection area).
    if (bestByInter) {{
      try {{ console.log("[door_detector] snapCandidateForDraw chosen", {{ reason: "max_intersection", id: bestByInter.id, inter: bestByInter.inter, iou: bestByInter.iou }}); }} catch (_) {{}}
      return bestByInter;
    }}

    // Fallback: if the drawn rectangle covers a substantial fraction of a candidate,
    // treat it as a match even when IoU is low.
    if (bestByCoverage && bestByCoverage.coverage >= MIN_CAND_COVERAGE) {{
      try {{ console.log("[door_detector] snapCandidateForDraw chosen", {{ reason: "coverage", id: bestByCoverage.id, coverage: bestByCoverage.coverage, iou: bestByCoverage.iou }}); }} catch (_) {{}}
      return bestByCoverage;
    }}

    try {{ console.log("[door_detector] snapCandidateForDraw no match (overlap too weak)", {{ bestByIoU: best ? {{ id: best.id, iou: best.iou }} : null }}); }} catch (_) {{}}
    return null;
  }}

  function renderManualOverlays() {{
    const raw = readParentInputValue(manualOverlaySinkLabel);
    if (raw === lastManualOverlayRaw) return false;
    lastManualOverlayRaw = raw;
    try {{ console.log("[door_detector] renderManualOverlays update", {{ bytes: raw ? raw.length : 0 }}); }} catch (_) {{}}

    const obj = readParentJson(manualOverlaySinkLabel) || {{}};
    const manual = Array.isArray(obj.manual_additions) ? obj.manual_additions : [];
    const unmatched = Array.isArray(obj.unmatched_manual_boxes) ? obj.unmatched_manual_boxes : [];

    manualLayer = ensureLayer("pz_manual");
    tempLayer = ensureLayer("pz_temp");
    clearLayer(manualLayer);
    // Once server overlays update, drop any client-only temp boxes to avoid duplicates.
    clearLayer(tempLayer);
    try {{ console.log("[door_detector] renderManualOverlays cleared temp layer"); }} catch (_) {{}}

    for (const m of manual) {{
      const drawn = m.drawn_bbox_xyxy;
      const snapped = m.snapped_bbox_xyxy;
      const iou = m.iou;
      const cid = m.snapped_candidate_id;
      drawBox(
        manualLayer,
        drawn,
        "rgb(0,255,255)",
        2,
        "6,4",
        0.47,
        cid ? `drawn (→ ${{cid}}, iou=${{iou}})` : "drawn"
      );
      if (snapped && Array.isArray(snapped) && snapped.length === 4) {{
        drawBox(
          manualLayer,
          snapped,
          "rgb(0,255,0)",
          3,
          "4,3",
          0.77,
          cid ? `snapped (${{cid}}, iou=${{iou}})` : "snapped"
        );
      }}
    }}

    for (const u of unmatched) {{
      const bbox = u.bbox_xyxy;
      const note = u.note || "unmatched";
      drawBox(manualLayer, bbox, "rgb(255,0,255)", 2, "6,4", 0.63, String(note));
    }}
    return true;
  }}

  function pollUnmatchedDebug() {{
    const raw = readParentInputValue(unmatchedDebugSinkLabel);
    if (!raw) return false;
    if (raw === lastUnmatchedDebugRaw) return false;
    lastUnmatchedDebugRaw = raw;
    try {{
      let obj = null;
      try {{ obj = JSON.parse(raw); }} catch (_) {{ obj = null; }}
      // Print as normal lines so it’s easy to copy/paste without expanding groups.
      console.log("[door_detector] unmatched_debug_report raw", raw);
      if (obj) console.log("[door_detector] unmatched_debug_report parsed", obj);
      else console.warn("[door_detector] unmatched_debug_report parse_failed");
    }} catch (_) {{}}
    return true;
  }}

  function clientToImage(clientX, clientY) {{
    // Robust mapping based on the transformed content element’s DOM rect.
    // This avoids subtle drift/mismatch if tx/ty/scale bookkeeping ever diverges.
    try {{
      const cr = content.getBoundingClientRect();
      if (!(cr.width > 0) || !(cr.height > 0)) throw new Error("bad content rect");
      const iw = {int(img_width)};
      const ih = {int(img_height)};
      const qx = (clientX - cr.left) * (iw / cr.width);
      const qy = (clientY - cr.top) * (ih / cr.height);
      clientToImage._lastMethod = "domRect";
      return {{ x: qx, y: qy }};
    }} catch (_) {{
      // Fallback to internal transform math.
      const rect = root.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      const qx = (px - tx) / scale;
      const qy = (py - ty) / scale;
      clientToImage._lastMethod = "txScale";
      return {{ x: qx, y: qy }};
    }}
  }}

  let drawing = false;
  let drawStart = null;
  let drawRect = null;

  root.addEventListener("pointerdown", (e) => {{
    if (resetBtn && resetBtn.contains(e.target)) return;
    if (e.button !== 0) return;
    // Shift+drag draws a selector rectangle in Edit Doors mode.
    if (getEditEnabled() && e.shiftKey) {{
      drawing = true;
      stage.style.cursor = "crosshair";
      drawStart = clientToImage(e.clientX, e.clientY);
      tempLayer = ensureLayer("pz_temp");
      if (tempLayer) {{
        try {{
          drawRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          drawRect.setAttribute("fill", "rgb(0,255,255)");
          drawRect.setAttribute("fill-opacity", "0.12");
          drawRect.setAttribute("stroke", "rgb(0,255,255)");
          drawRect.setAttribute("stroke-opacity", "0.90");
          drawRect.setAttribute("stroke-width", "2");
          drawRect.setAttribute("stroke-dasharray", "6,4");
          drawRect.setAttribute("stroke-linecap", "square");
          drawRect.setAttribute("shape-rendering", "crispEdges");
          drawRect.setAttribute("vector-effect", "non-scaling-stroke");
          drawRect.style.pointerEvents = "none";
          tempLayer.appendChild(drawRect);
        }} catch (_) {{}}
      }}
      suppressSvgClickUntil = performance.now() + 250;
      e.preventDefault();
      e.stopPropagation();
      try {{ root.setPointerCapture(e.pointerId); }} catch (_) {{}}
      return;
    }}

    // Default: pan drag.
    dragging = true;
    stage.style.cursor = "grabbing";
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragStartTx = tx;
    dragStartTy = ty;
    try {{ root.setPointerCapture(e.pointerId); }} catch (_) {{}}
  }});

  root.addEventListener("pointermove", (e) => {{
    if (drawing && drawStart) {{
      const p = clientToImage(e.clientX, e.clientY);
      const q = (v) => Math.round(Number(v) * 2) / 2;
      const x0 = q(Math.min(drawStart.x, p.x));
      const y0 = q(Math.min(drawStart.y, p.y));
      const x1 = q(Math.max(drawStart.x, p.x));
      const y1 = q(Math.max(drawStart.y, p.y));
      const w = q(Math.max(0, x1 - x0));
      const h = q(Math.max(0, y1 - y0));
      if (drawRect) {{
        try {{
          drawRect.setAttribute("x", String(x0));
          drawRect.setAttribute("y", String(y0));
          drawRect.setAttribute("width", String(w));
          drawRect.setAttribute("height", String(h));
        }} catch (_) {{}}
      }}
      e.preventDefault();
      return;
    }}
    if (!dragging) return;
    tx = dragStartTx + (e.clientX - dragStartX);
    ty = dragStartTy + (e.clientY - dragStartY);
    applyTransform();
  }});

  function endDrag(e) {{
    if (drawing) {{
      drawing = false;
      stage.style.cursor = "grab";
      const end = clientToImage(e.clientX, e.clientY);
      const x0 = Math.min(drawStart?.x ?? end.x, end.x);
      const y0 = Math.min(drawStart?.y ?? end.y, end.y);
      const x1 = Math.max(drawStart?.x ?? end.x, end.x);
      const y1 = Math.max(drawStart?.y ?? end.y, end.y);
      drawStart = null;
      // Keep the drawn rect faintly visible until the rerun overlays arrive.
      if (drawRect) {{
        try {{
          drawRect.setAttribute("fill", "rgb(0,255,255)");
          drawRect.setAttribute("fill-opacity", "0.08");
          drawRect.setAttribute("stroke", "rgb(0,255,255)");
          drawRect.setAttribute("stroke-opacity", "0.65");
        }} catch (_) {{}}
      }}
      drawRect = null;

      // Emit draw event to Streamlit.
      const w = Math.abs(x1 - x0);
      const h = Math.abs(y1 - y0);
      if (w >= 2 && h >= 2) {{
        const drawn = [x0, y0, x1, y1];
        const snap = snapCandidateForDraw(drawn);
        try {{
          const n = svg ? svg.querySelectorAll("rect[data-door-id]").length : 0;
          const rr = root ? root.getBoundingClientRect() : null;
          const cr = content ? content.getBoundingClientRect() : null;
          const used = clientToImage._lastMethod || "unknown";
          // Compare the two mapping methods at the end point (diagnostic).
          const rect = rr || {{ left: 0, top: 0 }};
          const px = e.clientX - rect.left;
          const py = e.clientY - rect.top;
          const qx_math = (px - tx) / scale;
          const qy_math = (py - ty) / scale;
          const qx_dom = cr && cr.width ? ((e.clientX - cr.left) * ({int(img_width)} / cr.width)) : null;
          const qy_dom = cr && cr.height ? ((e.clientY - cr.top) * ({int(img_height)} / cr.height)) : null;
          console.log("[door_detector] draw_rect endDrag", {{
            drawn,
            drawnRounded: drawn.map((v) => Math.round(v)),
            drawnJson: JSON.stringify(drawn),
            candidates: n,
            snap: snap ? {{ id: snap.id, iou: snap.iou, bbox: snap.bbox }} : null,
            editEnabled: getEditEnabled(),
            viewMode: getViewerDisplayMode(),
            tx, ty, scale,
            rootRect: rr ? {{ left: rr.left, top: rr.top, w: rr.width, h: rr.height }} : null,
            contentRect: cr ? {{ left: cr.left, top: cr.top, w: cr.width, h: cr.height }} : null,
            mapCompare: {{ used, math: [qx_math, qy_math], dom: [qx_dom, qy_dom] }},
          }});
        }} catch (_) {{}}
        if (snap && tempLayer) {{
          try {{
            drawBox(
              tempLayer,
              snap.bbox,
              "rgb(0,255,0)",
              3,
              "4,3",
              0.77,
              `snapped (${{snap.id}}, iou=${{snap.iou}})`
            );
            try {{ console.log("[door_detector] draw_rect drew snap overlay", {{ id: snap.id }}); }} catch (_) {{}}
            // Immediate UX: select the snapped door locally (server selection arrives on rerun).
            // Auto-focus is suppressed in Edit Doors mode by `getEffectiveAutoFocus()`.
            localSelectedId = String(snap.id || "");
            try {{ applyDoorStyles(); }} catch (_) {{}}
          }} catch (_) {{}}
        }} else {{
          try {{ console.log("[door_detector] draw_rect no snap overlay drawn", {{ hasSnap: !!snap, hasTempLayer: !!tempLayer }}); }} catch (_) {{}}
          // Show an explicit unmatched marker immediately so the user isn't left
          // wondering if anything happened.
          if (tempLayer) {{
            try {{
              drawBox(
                tempLayer,
                drawn,
                "rgb(255,0,255)",
                2,
                "6,4",
                0.68,
                "unmatched (no overlapping candidate)"
              );
              console.log("[door_detector] draw_rect drew unmatched overlay");
            }} catch (_) {{}}
          }}
        }}
        const payload = {{
          event: "draw_rect",
          event_id: `${{Date.now()}}_${{Math.random().toString(16).slice(2)}}`,
          bbox_xyxy: drawn,
          snapped_candidate_id: snap ? snap.id : null,
          iou: snap ? snap.iou : null,
          snapped_bbox_xyxy: snap ? snap.bbox : null,
          ts: Date.now(),
        }};
        try {{
          const ok = setParentInputValue(drawEventSinkLabel, JSON.stringify(payload));
          // Some Streamlit builds don't rerun reliably on a hidden text_input change.
          // As a robust fallback, also poke the existing click sink with a sentinel
          // value that will not match any door id, solely to trigger the rerun.
          if (ok) {{
            const encoded = encodeURIComponent(JSON.stringify(payload));
            const ok2 = setParentInputValue(clickSinkLabel, `__draw_event__${{encoded}}`);
            try {{
              console.log("[door_detector] draw_rect sink_write", {{
                drawEventOk: ok,
                clickSentinelOk: ok2,
                drawEventLabel: drawEventSinkLabel,
                clickSinkLabel,
                lastSinkWrite: window.__door_detectorLastSinkWrite || null,
              }});
            }} catch (_) {{}}
          }}
        }} catch (_) {{}}
      }}
      return;
    }}

    if (!dragging) return;
    dragging = false;
    stage.style.cursor = "grab";
  }}

  root.addEventListener("pointerup", endDrag);
  root.addEventListener("pointercancel", endDrag);
  root.addEventListener("pointerleave", endDrag);

  function setSelectedDoorId(doorId) {{
    if (!doorId) return;
    const ok = setParentInputValue(clickSinkLabel, String(doorId));
    try {{
      console.log("[door_detector] click->sink", {{
        doorId,
        ok,
        clickSinkLabel,
        parentValueAfter: readParentInputValue(clickSinkLabel),
        lastSinkWrite: window.__door_detectorLastSinkWrite || null,
        ts: Date.now(),
      }});
    }} catch (_) {{}}
    // Diagnostic: if Streamlit ingests the sink update, it should rerun and either:
    // - clear the click sink (we consume it server-side), and/or
    // - update selectedSinkLabel / focusSeqSinkLabel.
    try {{
      const t0 = Date.now();
      setTimeout(() => {{
        try {{
          console.log("[door_detector] click->sink followup", {{
            doorId,
            dtMs: Date.now() - t0,
            clickSinkNow: readParentInputValue(clickSinkLabel),
            selectedSinkNow: readParentInputValue(selectedSinkLabel),
            focusSeqNow: readParentInputValue(focusSeqSinkLabel),
            ts: Date.now(),
          }});
        }} catch (_) {{}}
      }}, 450);
    }} catch (_) {{}}
  }}

  // Watch selection changes coming from Streamlit (right panel).
  let lastSelectedId = null;
  let lastFocusSeq = null;
  function pollSelection() {{
    const did = getSelectedId();
    const seq = getFocusSeq();
    focusSeq = seq;
    const viewMode = getViewerDisplayMode();
    const editEnabled = getEditEnabled();
    const doorStateChanged = updateDoorStateFromSinks();
    renderManualOverlays();
    pollUnmatchedDebug();

    const newAutoFocus = getAutoFocus();
    if (newAutoFocus !== autoFocus) {{
      autoFocus = newAutoFocus;
      // If the user just enabled auto-focus, focus immediately to the current selection.
      if (autoFocus && did) {{
        try {{ focusToDoorId(did); }} catch (_) {{}}
        lastFocusSeq = seq;
      }}
    }}

    let needsStyle = false;
    if (did !== lastSelectedId) {{
      lastSelectedId = did;
      localSelectedId = null;
      needsStyle = true;
    }}
    if (viewMode !== lastViewerDisplay) {{
      lastViewerDisplay = viewMode;
      needsStyle = true;
    }}
    if (editEnabled !== lastEditEnabled) {{
      const wasEditing = !!lastEditEnabled;
      lastEditEnabled = editEnabled;
      needsStyle = true;
      // On exit from Edit Doors, clear any edit-only overlays immediately
      // (blue drawn box, magenta unmatched boxes, etc).
      if (wasEditing && !editEnabled) {{
        try {{
          manualLayer = ensureLayer("pz_manual");
          tempLayer = ensureLayer("pz_temp");
          clearLayer(manualLayer);
          clearLayer(tempLayer);
          lastManualOverlayRaw = null;
          lastUnmatchedDebugRaw = null;
          console.log("[door_detector] edit_mode exited: cleared overlays");
        }} catch (_) {{}}
      }}
    }}
    if (doorStateChanged) needsStyle = true;
    if (needsStyle) applyDoorStyles();

    const effectiveAutoFocus = getAutoFocus() && !editEnabled;
    if (effectiveAutoFocus && did) {{
      if (lastFocusSeq === null) {{
        lastFocusSeq = seq;
      }} else if (seq !== lastFocusSeq) {{
        lastFocusSeq = seq;
        focusToDoorId(did);
      }}
    }}
  }}
  try {{ setInterval(pollSelection, 120); }} catch (_) {{}}

  if (svg) {{
    // If a door bbox is clicked, select it (and don't start a pan drag).
    svg.addEventListener("pointerdown", (e) => {{
      // In Edit Doors mode, Shift+drag should start drawing even if you start on a door box.
      if (getEditEnabled() && e.shiftKey) return;
      const t = e.target;
      if (t && t.getAttribute && t.getAttribute("data-door-id")) {{
        e.preventDefault();
        e.stopPropagation();
      }}
    }}, true);

    svg.addEventListener("click", (e) => {{
      if (performance.now && performance.now() < suppressSvgClickUntil) return;
      const t = e.target;
      if (!t || !t.getAttribute) return;
      const did = t.getAttribute("data-door-id");
      if (!did) return;
      e.preventDefault();
      e.stopPropagation();
      // Immediate UX: highlight/focus locally without waiting for the rerun.
      localSelectedId = did;
      applyDoorStyles();
      if (getEffectiveAutoFocus()) focusToDoorId(did);
      setSelectedDoorId(did);
    }});
  }}
}})();
</script>
"""
    try:
        viewer_hash = hashlib.sha256(viewer_html.encode("utf-8")).hexdigest()[:12]
        _debug_log("viewer key=%s html_hash=%s", str(key), viewer_hash)
    except Exception:
        pass
    components.html(viewer_html, height=height, scrolling=False)


def main_viewer_canvas(
    item: Dict,
    *,
    preview_spec: Optional[Dict[str, Any]],
    full_dims: Optional[Tuple[int, int]],
    doors_data: Dict,
    fstate: Dict,
    active_doors: List[Dict[str, Any]],
    click_sink_label: str,
):
    file_id = item["id"]
    file_dir = Path(item["path"])

    if preview_spec:
        # Door click sink: JS writes the clicked door id into this hidden widget,
        # which triggers a rerun; on rerun we sync it into the "Jump to door" control.
        click_sink_key = click_sink_label
        st.text_input(click_sink_label, key=click_sink_key, label_visibility="collapsed")

        # Auto-focus sink: viewer polls this so toggling the checkbox doesn't remount the iframe.
        auto_focus_key = f"auto_focus_{file_id}"
        auto_focus_sink_label = f"auto_focus_sink_{file_id}"
        try:
            # Canonicalize early (main_viewer_controls may render after the viewer).
            auto_focus_val = _coerce_bool(
                st.session_state.get(auto_focus_key),
                default=_coerce_bool(fstate.get("auto_focus", True), default=True),
            )
            fstate["auto_focus"] = auto_focus_val
            st.session_state[auto_focus_sink_label] = "1" if auto_focus_val else "0"
        except Exception:
            pass
        st.text_input(auto_focus_sink_label, key=auto_focus_sink_label, label_visibility="collapsed")

        # Selection/focus sinks: the viewer polls these so selection changes don't require
        # changing the viewer HTML (reduces iframe reload flicker).
        selected_sink_label = f"selected_door_sink_{file_id}"
        focus_seq_sink_label = f"focus_seq_sink_{file_id}"
        try:
            st.session_state[selected_sink_label] = str(fstate.get("selected_door_id") or "")
            st.session_state[focus_seq_sink_label] = str(int(fstate.get("_focus_seq") or 0))
        except Exception:
            pass
        st.text_input(selected_sink_label, key=selected_sink_label, label_visibility="collapsed")
        st.text_input(focus_seq_sink_label, key=focus_seq_sink_label, label_visibility="collapsed")

        # Edit-mode + overlay sinks: the viewer polls these so we can enable Shift+drag
        # drawing and update styles/overlays without remounting the iframe.
        edit_mode_sink_label = f"edit_mode_sink_{file_id}"
        draw_event_sink_label = f"draw_event_sink_{file_id}"
        manual_overlay_sink_label = f"manual_overlay_sink_{file_id}"
        door_state_sink_label = f"door_state_sink_{file_id}"
        viewer_display_sink_label = f"viewer_display_sink_{file_id}"
        unmatched_debug_sink_label = f"unmatched_debug_sink_{file_id}"
        candidate_pool_sink_label = f"candidate_pool_sink_{file_id}"

        # Server → iframe values (updated every run).
        try:
            working = _get_working_label_state(fstate)
            st.session_state[edit_mode_sink_label] = "1" if bool(fstate.get("edit_mode")) else "0"
            st.session_state[viewer_display_sink_label] = _viewer_display_mode_to_sink_value(
                str(fstate.get("viewer_display_mode") or "Highlight All")
            )
            st.session_state[door_state_sink_label] = json.dumps(
                {
                    "confirmed_ids": sorted(list(flatten_confirmed_ids(working.get("confirmed_by_type", {})))),
                    "deleted_ids": sorted(list(working.get("deleted_ids", set()))),
                },
                separators=(",", ":"),
            )
            # Manual overlay data is supplied via a separate sink (preview-space bboxes).
            # Only show these while editing so exiting Edit Doors clears the overlays.
            if bool(fstate.get("edit_mode")):
                manual_payload = _manual_overlay_payload_for_sink(
                    fstate=fstate,
                    preview_scale=float(preview_spec.get("scale", 1.0) or 1.0),
                )
            else:
                manual_payload = {"manual_additions": [], "unmatched_manual_boxes": []}
            st.session_state[manual_overlay_sink_label] = json.dumps(manual_payload, separators=(",", ":"))
            st.session_state[unmatched_debug_sink_label] = str(fstate.get("_last_unmatched_debug") or "")
            # Candidate pool for snapping: preview-space bboxes for a broad set of candidates.
            # This allows snap-to-candidate even when the candidate is not in the final `doors` list.
            try:
                scale = float(preview_spec.get("scale", 1.0) or 1.0)
            except Exception:
                scale = 1.0
            pool = list(doors_data.get("candidates", []) or [])

            def _sample_pool_for_sink(
                cands: List[Dict[str, Any]],
                *,
                full_w: Optional[int],
                full_h: Optional[int],
                max_out: int,
                grid: int = 8,
            ) -> List[Dict[str, Any]]:
                """Pick a spatially-diverse subset of candidates for client snapping.

                Using only top-N by confidence can omit low-confidence-but-correct
                candidates elsewhere on the page. A coarse grid keeps coverage.
                """
                if not cands:
                    return []
                if not full_w or not full_h or full_w <= 0 or full_h <= 0:
                    return cands[:max_out]
                g = max(2, min(16, int(grid)))
                per_cell = max(1, int(max_out // (g * g)))
                buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
                seen: set[str] = set()

                def _cell_for_bbox(bb: Tuple[float, float, float, float]) -> Tuple[int, int]:
                    cx = 0.5 * (bb[0] + bb[2])
                    cy = 0.5 * (bb[1] + bb[3])
                    ix = int((cx / float(full_w)) * g)
                    iy = int((cy / float(full_h)) * g)
                    ix = max(0, min(g - 1, ix))
                    iy = max(0, min(g - 1, iy))
                    return (ix, iy)

                # Pool is already sorted by confidence; keep each bucket in that order.
                for cand in cands:
                    cid = cand.get("id")
                    bbox = cand.get("bbox_xyxy")
                    if cid is None or not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    sid = str(cid)
                    if sid in seen:
                        continue
                    nb = _normalize_bbox_xyxy(bbox)
                    if nb is None:
                        continue
                    cell = _cell_for_bbox(nb)
                    buckets.setdefault(cell, []).append(cand)
                    seen.add(sid)

                out: List[Dict[str, Any]] = []
                used: set[str] = set()
                for ix in range(g):
                    for iy in range(g):
                        for cand in buckets.get((ix, iy), [])[:per_cell]:
                            sid = str(cand.get("id"))
                            if sid in used:
                                continue
                            out.append(cand)
                            used.add(sid)
                            if len(out) >= max_out:
                                return out

                # Fill remaining slots with highest-confidence unused candidates.
                if len(out) < max_out:
                    for cand in cands:
                        sid = str(cand.get("id"))
                        if not sid or sid in used:
                            continue
                        out.append(cand)
                        used.add(sid)
                        if len(out) >= max_out:
                            break
                return out[:max_out]

            # Determine full-res dimensions for spatial bucketing.
            full_w = None
            full_h = None
            if isinstance(full_dims, tuple) and len(full_dims) == 2:
                try:
                    full_w = int(full_dims[0]) if full_dims[0] is not None else None
                    full_h = int(full_dims[1]) if full_dims[1] is not None else None
                except Exception:
                    full_w, full_h = None, None

            picked = _sample_pool_for_sink(pool, full_w=full_w, full_h=full_h, max_out=1200, grid=8)
            out_pool: List[Dict[str, Any]] = []
            for cand in picked:
                cid = cand.get("id")
                bbox = cand.get("bbox_xyxy")
                if cid is None or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                nb = _normalize_bbox_xyxy(bbox)
                if nb is None:
                    continue
                # full → preview
                out_pool.append(
                    {
                        "id": str(cid),
                        "type": str(cand.get("type") or ""),
                        "confidence": float(cand.get("confidence", 0.0) or 0.0),
                        "bbox_xyxy": [float(nb[0]) * scale, float(nb[1]) * scale, float(nb[2]) * scale, float(nb[3]) * scale],
                    }
                )
            st.session_state[candidate_pool_sink_label] = json.dumps({"candidates": out_pool}, separators=(",", ":"))
        except Exception:
            pass

        # Iframe → server events (do not overwrite once set by JS).
        if draw_event_sink_label not in st.session_state:
            st.session_state[draw_event_sink_label] = ""

        st.text_input(edit_mode_sink_label, key=edit_mode_sink_label, label_visibility="collapsed")
        st.text_input(draw_event_sink_label, key=draw_event_sink_label, label_visibility="collapsed")
        st.text_input(manual_overlay_sink_label, key=manual_overlay_sink_label, label_visibility="collapsed")
        st.text_input(door_state_sink_label, key=door_state_sink_label, label_visibility="collapsed")
        st.text_input(viewer_display_sink_label, key=viewer_display_sink_label, label_visibility="collapsed")
        st.text_input(unmatched_debug_sink_label, key=unmatched_debug_sink_label, label_visibility="collapsed")
        st.text_input(candidate_pool_sink_label, key=candidate_pool_sink_label, label_visibility="collapsed")

        viewer_width_hint = int(VIEWER_TARGET_WIDTH_PX)
        viewer_width_hint = max(600, min(2000, viewer_width_hint))
        aspect = float(VIEWER_ASPECT_RATIO_HW)
        aspect = max(0.35, min(1.25, aspect))

        # Viewer height derived from width and a fixed aspect ratio.
        viewer_height = int(round(viewer_width_hint * aspect))
        viewer_height = max(450, min(1400, viewer_height))

        # NOTE: Don't wrap this in `st.container(height=...)` because Streamlit makes that
        # container scrollable (adds a scrollbar) which steals scroll/drag interactions.
        rects_svg = _rects_to_svg(
            active_doors=active_doors,
            fstate=fstate,
            scale=float(preview_spec.get("scale", 1.0)),
            img_width=int(preview_spec.get("width", 1)),
            img_height=int(preview_spec.get("height", 1)),
        )

        img_src = _image_path_to_streamlit_url(str(preview_spec.get("path", "")))

        _panzoom_image_viewer(
            img_src=img_src,
            img_width=int(preview_spec.get("width", 1)),
            img_height=int(preview_spec.get("height", 1)),
            rects_svg=rects_svg,
            height=viewer_height,
            key=str(file_id),
            click_sink_aria_label=click_sink_label,
            selected_sink_aria_label=selected_sink_label,
            focus_seq_sink_aria_label=focus_seq_sink_label,
            edit_mode_sink_aria_label=edit_mode_sink_label,
            draw_event_sink_aria_label=draw_event_sink_label,
            manual_overlay_sink_aria_label=manual_overlay_sink_label,
            door_state_sink_aria_label=door_state_sink_label,
            viewer_display_sink_aria_label=viewer_display_sink_label,
            auto_focus_sink_aria_label=auto_focus_sink_label,
            unmatched_debug_sink_aria_label=unmatched_debug_sink_label,
            candidate_pool_sink_aria_label=candidate_pool_sink_label,
        )
        return None, active_doors

    # Still run the sidebar auto-open logic even when the viewer isn't mounted.
    components.html(sidebar_autopen_component_html(), height=0, scrolling=False)
    st.info("Analyze to see results.")
    return None, []

