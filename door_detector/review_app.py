import hashlib
import json
import math
import os
import shutil
import time
import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

def _patch_streamlit_drawable_canvas_image_to_url() -> None:
    """Compatibility shim for streamlit-drawable-canvas.

    streamlit-drawable-canvas<=0.9.x calls streamlit.elements.image.image_to_url,
    but newer Streamlit moved the implementation and changed its signature.
    """
    try:
        import streamlit.elements.image as st_image

        if hasattr(st_image, "image_to_url"):
            return

        from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig

        def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
            return _image_to_url(
                image=image,
                layout_config=LayoutConfig(width=width),
                clamp=clamp,
                channels=channels,
                output_format=output_format,
                image_id=image_id,
            )

        st_image.image_to_url = _image_to_url_compat  # type: ignore[attr-defined]
    except Exception:
        # If Streamlit internals change again, avoid breaking the app on import.
        return


_patch_streamlit_drawable_canvas_image_to_url()

from door_detector.library import Library
from door_detector.step1_pipeline import process_pdf
from door_detector.step2_pipeline import run_step2
from door_detector.reweight_fit import fit_reweighter

# Increase PIL pixel limit
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Door Detector: Door Detection & Review", layout="wide", initial_sidebar_state="expanded")

# --- UI Styling ---
st.markdown("""
<style>
    /* Hide Streamlit chrome (cosmetic only; not a security boundary) */
    header { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    [data-testid="stDeployButton"] { display: none !important; }

    /* Remove Streamlit's default huge bottom spacing in main area */
    html body section.stMain {
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
    }
    html body section.stMain > div {
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
    }
    html body section.stMain .block-container {
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
    }
    /* Some Streamlit versions include an empty bottom container */
    [data-testid="stBottomBlockContainer"] { display: none !important; }

    /* Pull main content to the top (align with sidebar header) */
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 0rem !important;
        margin-top: 0rem !important;
        max-width: 100% !important;
    }

    /* Streamlit sometimes targets this container directly; keep it tight. */
    [data-testid="stMainBlockContainer"] {
        padding-top: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 0rem !important;
        margin-top: 0rem !important;
    }
    [data-testid="stAppViewContainer"] .main h1, 
    [data-testid="stAppViewContainer"] .main h3 {
        margin-top: 0 !important;
        padding-top: 0.5rem !important;
    }

    /* Styling for the file list rows */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        padding: 2px 4px !important;
        transition: background-color 0.2s;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }

    /* Library file list: plain text buttons, single-line, clipped (no wrap) */
    [data-testid="stSidebar"] button[id^="sel_"] {
        background: none !important;
        border: none !important;
        padding: 2px 0px !important;
        margin: 0 !important;
        color: rgba(255, 255, 255, 0.8) !important;
        text-align: left !important;
        width: 100% !important;
        display: block !important;
        box-shadow: none !important;
        min-height: 0 !important;
        line-height: 1.2 !important;
        outline: none !important;
        overflow: hidden !important;
        white-space: nowrap !important;
    }

    [data-testid="stSidebar"] button[id^="sel_"]:hover {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Streamlit renders button labels via a markdown container; force it to never wrap */
    [data-testid="stSidebar"] button[id^="sel_"] [data-testid="stMarkdownContainer"] {
        overflow: hidden !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }

    [data-testid="stSidebar"] button[id^="sel_"] div {
        overflow: hidden !important;
        min-width: 0 !important;
        max-width: 100% !important;
        flex-wrap: nowrap !important;
    }

    [data-testid="stSidebar"] button[id^="sel_"] * {
        white-space: nowrap !important;
    }

    [data-testid="stSidebar"] button[id^="sel_"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] button[id^="sel_"] [data-testid="stMarkdownContainer"] span {
        overflow: hidden !important;
        text-overflow: clip !important;
        display: block !important;
        width: 100% !important;
        font-size: 13px !important;
        margin: 0 !important;
    }

    /* Selected library item: emphasize via text only (no box) */
    [data-testid="stSidebar"] button[id^="sel_"][data-testid="stBaseButton-primary"] {
        color: white !important;
        font-weight: 600 !important;
    }

    /* Streamlit/BaseUI buttons often look like: #bui3__anchor > button
       Force sidebar button labels to NEVER wrap (clip instead). */
    [data-testid="stSidebar"] div[id$="__anchor"] > button {
        max-width: 100% !important;
        width: 100% !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] div[id$="__anchor"] > button > div,
    [data-testid="stSidebar"] div[id$="__anchor"] > button > div > div {
        max-width: 100% !important;
        min-width: 0 !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] div[id$="__anchor"] > button [data-testid="stMarkdownContainer"] {
        max-width: 100% !important;
        min-width: 0 !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] div[id$="__anchor"] > button [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[id$="__anchor"] > button [data-testid="stMarkdownContainer"] span {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: clip !important;
        max-width: 100% !important;
    }

    /* File Uploader styling to look like a simple "Add" button */
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
        border: none !important;
        background: none !important;
    }
    [data-testid="stFileUploader"] section > div {
        display: none !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
        padding: 0px 12px !important;
        border-radius: 4px !important;
        font-size: 14px !important;
        height: 38px !important;
        width: 100% !important;
    }
    /* Replace Streamlit's default "Browse files" label with "Upload" */
    [data-testid="stFileUploader"] button > div > p {
        font-size: 0 !important;
        line-height: 1 !important;
        margin: 0 !important;
    }
    [data-testid="stFileUploader"] button > div > p::before {
        content: "Upload";
        font-size: 14px;
    }
    [data-testid="stFileUploader"] small {
        display: none !important;
    }

    /* Keep Search/Clear buttons looking like buttons */
    [data-testid="stSidebar"] .stButton button#open_search_btn,
    [data-testid="stSidebar"] .stButton button#close_search_btn {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 0px 12px !important;
        text-align: center !important;
        border-radius: 4px !important;
        margin-bottom: 0px !important;
        height: 38px !important;
    }

    /* File list rows: remove column gap and ensure background covers everything */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] [data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        border: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.1rem !important;
    }
    
    [data-testid="stSidebar"] h1 {
        padding-top: 0 !important;
        margin-top: -20px !important;
    }

    /* Make the X button align better with the search input */
    [data-testid="stSidebar"] div[data-testid="column"] .stButton button#close_search_btn {
        margin-top: 0px !important;
        height: 38px !important;
        width: 100% !important;
    }

    /* Make the Search button match Upload sizing */
    [data-testid="stSidebar"] div[data-testid="column"] .stButton button#open_search_btn {
        margin-top: 0px !important;
        height: 38px !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Library ---
if "library" not in st.session_state:
    st.session_state.library = Library(Path("artifacts"))
    # One-time discovery of existing artifacts
    st.session_state.library.discover_existing()

if "search_visible" not in st.session_state:
    st.session_state.search_visible = False

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "viewer_height" not in st.session_state:
    # Larger default to better fill tall viewports (reduces "blank footer" area).
    st.session_state.viewer_height = 1000

lib = st.session_state.library

# --- Session State Helpers ---
def init_file_state(file_id: str, doors_data: Dict, labels_data: Dict):
    if "files" not in st.session_state:
        st.session_state.files = {}
    
    if file_id not in st.session_state.files:
        st.session_state.files[file_id] = {
            "accepted": set(labels_data.get("accepted_ids", [])),
            "rejected": set(labels_data.get("rejected_ids", [])),
            "added_boxes": labels_data.get("added_boxes", []),
            "notes": labels_data.get("notes", ""),
            "selected_door_id": None,
            "viewer_mode": "Highlight All",
        }

# --- Data Loading ---
@st.cache_data
def load_file_artifacts(file_dir_str: str):
    file_dir = Path(file_dir_str)
    image_path = file_dir / "page.png"
    doors_path = file_dir / "doors.json"
    labels_path = file_dir / "labels.json"
    meta_path = file_dir / "meta.json"

    image = Image.open(image_path) if image_path.exists() else None
    
    doors_data = {}
    if doors_path.exists():
        with open(doors_path) as f:
            doors_data = json.load(f)
            
    labels_data = {
        "accepted_ids": [],
        "rejected_ids": [],
        "added_boxes": [],
        "notes": ""
    }
    if labels_path.exists():
        with open(labels_path) as f:
            labels_data = json.load(f)
            
    meta_data = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta_data = json.load(f)

    return image, doors_data, labels_data, meta_data

def save_labels(dir_path: Path, labels_data: Dict[str, Any]):
    labels_path = dir_path / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)

def get_current_signature(config_path: str):
    try:
        with open(config_path, "rb") as f:
            config_bytes = f.read()
        config = json.loads(config_bytes)
        sig_content = config_bytes
        if "reweighter_path" in config:
            re_path = Path(config["reweighter_path"])
            if re_path.exists():
                with open(re_path, "rb") as f:
                    sig_content += b"|" + f.read()
        return hashlib.sha256(sig_content).hexdigest()
    except Exception:
        return None

# --- UI Components ---

def _pil_image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _normalize_bbox_xyxy(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return (x0, y0, x1, y1) with x0<=x1 and y0<=y1, or None if invalid."""
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def _panzoom_image_viewer(img: Image.Image, *, height: int, key: str) -> None:
    data_url = _pil_image_to_data_url(img)
    # This viewer provides:
    # - scrollwheel zoom (centered at cursor)
    # - click+drag pan
    # - initial fit-to-container with letterboxing
    html = f"""
<div id="pz_root_{key}" style="width: 100%; height: {height}px; overflow: hidden; background: #0e1117; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.12);">
  <style>
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
    <img
      id="pz_img_{key}"
      src="{data_url}"
      style="position: absolute; left: 0; top: 0; transform-origin: 0 0; will-change: transform; pointer-events: none;"
    />
  </div>
</div>

<script>
(function() {{
  const root = document.getElementById("pz_root_{key}");
  const stage = document.getElementById("pz_stage_{key}");
  const img = document.getElementById("pz_img_{key}");
  const resetBtn = document.getElementById("pz_reset_{key}");
  if (!root || !stage || !img) return;

  let scale = 1;
  let tx = 0;
  let ty = 0;

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
    img.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
    updateResetVisibility();
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
    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;
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
  if (img.complete) fitToContainer();
  else img.addEventListener("load", fitToContainer, {{ once: true }});

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

  root.addEventListener("pointerdown", (e) => {{
    if (resetBtn && resetBtn.contains(e.target)) return;
    if (e.button !== 0) return;
    dragging = true;
    stage.style.cursor = "grabbing";
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragStartTx = tx;
    dragStartTy = ty;
    try {{ root.setPointerCapture(e.pointerId); }} catch (_) {{}}
  }});

  root.addEventListener("pointermove", (e) => {{
    if (!dragging) return;
    tx = dragStartTx + (e.clientX - dragStartX);
    ty = dragStartTy + (e.clientY - dragStartY);
    applyTransform();
  }});

  function endDrag() {{
    if (!dragging) return;
    dragging = false;
    stage.style.cursor = "grab";
  }}

  root.addEventListener("pointerup", endDrag);
  root.addEventListener("pointercancel", endDrag);
  root.addEventListener("pointerleave", endDrag);
}})();
</script>
"""
    components.html(html, height=height, scrolling=False)

def sidebar_library():
    st.sidebar.title("Library")

    # Search and Add Area
    if not st.session_state.search_visible:
        col_search, col_add = st.sidebar.columns(2)
        with col_search:
            if st.button("Search", key="open_search_btn", help="Open search", use_container_width=True):
                st.session_state.search_visible = True
                st.rerun()
        with col_add:
            uploaded_file = st.file_uploader("Upload", type=["pdf"], label_visibility="collapsed")
            if uploaded_file:
                file_id = lib.add_file(uploaded_file.name, uploaded_file.getvalue())
                st.rerun()
    else:
        col_input, col_close = st.sidebar.columns([5, 1])
        with col_input:
            search_val = st.text_input(
                "Search", 
                value=st.session_state.search_query,
                label_visibility="collapsed",
                key="search_input_widget"
            )
            if search_val != st.session_state.search_query:
                st.session_state.search_query = search_val
                st.rerun()
        with col_close:
            if st.button("X", key="close_search_btn", help="Clear search"):
                st.session_state.search_query = ""
                st.session_state.search_visible = False
                st.rerun()

    st.sidebar.divider()
    
    items = lib.get_items()
    if st.session_state.search_query:
        items = [i for i in items if st.session_state.search_query.lower() in i["original_name"].lower()]
    
    if not items:
        st.sidebar.info("No files in library.")
    else:
        for item in items:
            is_selected = (st.session_state.get("selected_file_id") == item["id"])
            display_name = item["original_name"]
            label = display_name
            
            if st.sidebar.button(
                label,
                key=f"sel_{item['id']}",
                help=item["original_name"],
                type="primary" if is_selected else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected_file_id = item["id"]
                st.rerun()

def main_viewer_canvas(item: Dict, image: Image.Image, doors_data: Dict, fstate: Dict):
    file_id = item["id"]
    file_dir = Path(item["path"])
    
    if image:
        detections = doors_data.get("doors", [])

        # Filter detections: keep undecided and accepted, exclude rejected
        active_doors = [d for d in detections if d["id"] not in fstate["rejected"]]
        # Add user added boxes as pseudo-detections
        for box in fstate["added_boxes"]:
            bbox = box["bbox_xyxy"]
            # Stable ID for added box based on coordinates
            box_id = f"u_{int(bbox[0])}_{int(bbox[1])}"
            active_doors.append({
                "id": box_id,
                "type": "added",
                "bbox_xyxy": bbox,
                "confidence": 1.0,
                "is_user_added": True
            })

        viewer_height = int(st.session_state.get("viewer_height", 1000))
        viewer_width_hint = 1200  # used only for Add Door canvas sizing

        # NOTE: Don't wrap this in `st.container(height=...)` because Streamlit makes that
        # container scrollable (adds a scrollbar) which steals scroll/drag interactions.
        if fstate["viewer_mode"] == "Add Door":
            # Fit-to-container (approx) for first render.
            fit_scale = min(1.0, min(viewer_width_hint / image.width, viewer_height / image.height))
            display_width = max(400, int(image.width * fit_scale))
            display_height = max(400, int(image.height * fit_scale))

            # Resize background for performance.
            bg_img = image.resize((display_width, display_height), Image.LANCZOS)

            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 255, 0.3)",
                stroke_width=2,
                stroke_color="#00ffff",
                background_image=bg_img,
                update_streamlit=True,
                height=display_height,
                width=display_width,
                drawing_mode="rect",
                key=f"canvas_{file_id}",
            )
            return canvas_result, active_doors

        # Normal viewing: pan+zoom image viewer (scrollwheel + click-drag).
        # Downscale large pages for faster rendering, while keeping the right panel
        # crop coming from the full-resolution image.
        max_viewer_width = 2000
        viewer_scale = min(1.0, max_viewer_width / image.width)
        v_w = max(1, int(image.width * viewer_scale))
        v_h = max(1, int(image.height * viewer_scale))
        viewer_img = image.resize((v_w, v_h), Image.LANCZOS)

        draw = ImageDraw.Draw(viewer_img)
        if fstate["viewer_mode"] != "Off":
            for d in active_doors:
                is_selected = d["id"] == fstate["selected_door_id"]
                if fstate["viewer_mode"] == "Highlight Selected" and not is_selected:
                    continue

                bbox = d["bbox_xyxy"]
                nb = _normalize_bbox_xyxy(bbox)
                if nb is None:
                    continue
                bbox_s = [int(round(x * viewer_scale)) for x in nb]

                color = (0, 255, 0) if d["id"] in fstate["accepted"] else (255, 165, 0)
                if d.get("is_user_added"):
                    color = (0, 255, 255)

                width = max(1, int(round(4 * viewer_scale)))
                if is_selected:
                    color = (255, 255, 255)
                    width = max(2, int(round(8 * viewer_scale)))

                draw.rectangle(bbox_s, outline=color, width=width)

        _panzoom_image_viewer(viewer_img, height=viewer_height, key=str(file_id))
        return None, active_doors
    else:
        st.info("Run analysis to see results.")
        return None, []

def main_viewer_controls(item: Dict, image: Image.Image, doors_data: Dict, fstate: Dict, canvas_result: Any):
    file_id = item["id"]
    file_dir = Path(item["path"])
    
    # Grid for main controls
    c1, c2 = st.columns(2)
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
        
        label = "Rerun" if status == "done" else "Run"
        if is_out_of_date:
            label = f"{label} (!)"
        
        if st.button(label, type="primary" if not status == "done" else "secondary", use_container_width=True):
            run_pipeline(file_id, file_dir, config_path)
            st.rerun()
    with c2:
        modes = ["Highlight All", "Highlight Selected", "Off", "Add Door"]
        fstate["viewer_mode"] = st.selectbox(
            "Mode", 
            modes,
            index=modes.index(fstate["viewer_mode"]) if fstate["viewer_mode"] in modes else 0,
            label_visibility="collapsed"
        )

    with st.expander("Layout", expanded=False):
        st.session_state.viewer_height = st.slider(
            "Viewer height (px)",
            min_value=600,
            max_value=1600,
            value=int(st.session_state.get("viewer_height", 1000)),
            step=50,
        )

    c3, c4 = st.columns(2)
    with c3:
        if fstate["viewer_mode"] == "Add Door":
            if st.button("Cancel", use_container_width=True):
                fstate["viewer_mode"] = "Highlight All"
                st.rerun()
        else:
            if st.button("Add Door", use_container_width=True):
                fstate["viewer_mode"] = "Add Door"
                st.rerun()
    with c4:
        if fstate["viewer_mode"] != "Add Door":
            pass

    if fstate["viewer_mode"] == "Add Door":
        st.info("Draw rectangles on the PDF.")
        if st.button("Save Added Doors", type="primary", use_container_width=True):
            if canvas_result and canvas_result.json_data:
                objects = canvas_result.json_data["objects"]
                display_width = canvas_result.image_data.shape[1] if canvas_result.image_data is not None else 1
                display_height = canvas_result.image_data.shape[0] if canvas_result.image_data is not None else 1
                scale_x = image.width / display_width if display_width else 1.0
                scale_y = image.height / display_height if display_height else 1.0
                
                added_count = 0
                for obj in objects:
                    if obj["type"] == "rect":
                        x0 = obj["left"] * scale_x
                        y0 = obj["top"] * scale_y
                        x1 = (obj["left"] + obj["width"]) * scale_x
                        y1 = (obj["top"] + obj["height"]) * scale_y
                        fstate["added_boxes"].append({"bbox_xyxy": [x0, y0, x1, y1]})
                        added_count += 1
                
                if added_count > 0:
                    save_current_labels(file_id, file_dir)
                    fstate["viewer_mode"] = "Highlight All"
                    st.rerun()
                else:
                    st.warning("No rectangles drawn.")

def right_panel_review(item: Dict, image: Image.Image, doors_data: Dict, fstate: Dict, active_doors: List):
    file_id = item["id"]
    file_dir = Path(item["path"])
    
    # Use pre-calculated active_doors (which already includes added_boxes)
    all_visible = active_doors.copy()
    all_visible.sort(key=lambda x: x["confidence"], reverse=True)
    
    st.subheader(f"Doors ({len(all_visible)})")
    
    if not all_visible:
        return

    # Selection sync
    selected_idx = -1
    if fstate["selected_door_id"]:
        for i, d in enumerate(all_visible):
            if d["id"] == fstate["selected_door_id"]:
                selected_idx = i
                break
    
    if selected_idx == -1 and all_visible:
        selected_idx = 0
        fstate["selected_door_id"] = all_visible[0]["id"]

    # Jump-to selector (replaces click-to-select in the main viewer)
    door_ids = [d["id"] for d in all_visible]
    id_to_label = {
        d["id"]: f"{i+1}/{len(all_visible)}  {d['type']}  {d['confidence']:.3f}  {d['id']}"
        for i, d in enumerate(all_visible)
    }
    # Streamlit selectbox keeps its own state keyed by `key=...`.
    # Keep that state in sync with our fstate selection so Prev/Next works.
    jump_key = f"jump_{file_id}"
    if jump_key not in st.session_state or st.session_state.get(jump_key) not in door_ids:
        st.session_state[jump_key] = fstate["selected_door_id"]
    picked = st.selectbox(
        "Jump to door",
        door_ids,
        index=selected_idx,
        format_func=lambda did: id_to_label.get(did, str(did)),
        label_visibility="collapsed",
        key=jump_key,
    )
    fstate["selected_door_id"] = picked
    selected_idx = door_ids.index(picked) if picked in door_ids else selected_idx

    # Prev/Next
    col_p, col_idx, col_n = st.columns([1, 2, 1])
    if col_p.button("Prev", disabled=False, use_container_width=True):
        new_idx = (selected_idx - 1) % len(all_visible)
        new_id = all_visible[new_idx]["id"]
        st.session_state[jump_key] = new_id
        fstate["selected_door_id"] = new_id
        st.rerun()
    col_idx.write(f"<div style='text-align: center; line-height: 38px;'>{selected_idx + 1} / {len(all_visible)}</div>", unsafe_allow_html=True)
    if col_n.button("Next", disabled=False, use_container_width=True):
        new_idx = (selected_idx + 1) % len(all_visible)
        new_id = all_visible[new_idx]["id"]
        st.session_state[jump_key] = new_id
        fstate["selected_door_id"] = new_id
        st.rerun()

    st.divider()
    
    # Details of selected
    selected_door = all_visible[selected_idx]
    did = selected_door["id"]
    
    st.write(f"**ID:** `{did}` | **Type:** {selected_door['type']} | **Conf:** {selected_door['confidence']:.3f}")
    
    # Zoom
    if image:
        bbox = selected_door["bbox_xyxy"]
        nb = _normalize_bbox_xyxy(bbox)
        if nb is None:
            st.warning("Selected door has an invalid bbox; preview unavailable.")
            return
        x0, y0, x1, y1 = nb
        pad = 100
        left = max(0, int(math.floor(x0 - pad)))
        upper = max(0, int(math.floor(y0 - pad)))
        right = min(image.width, int(math.ceil(x1 + pad)))
        lower = min(image.height, int(math.ceil(y1 + pad)))
        if right <= left or lower <= upper:
            st.warning("Selected door bbox is degenerate after clamping; preview unavailable.")
        else:
            st.image(image.crop((left, upper, right, lower)), use_container_width=True)

    # Actions
    c1, c2, c3 = st.columns(3)
    if c1.button("Accept", use_container_width=True):
        fstate["accepted"].add(did)
        fstate["rejected"].discard(did)
        save_current_labels(file_id, file_dir)
        st.rerun()
    if c2.button("Reject", use_container_width=True):
        if selected_door.get("is_user_added"):
            fstate["added_boxes"] = [b for b in fstate["added_boxes"] if f"u_{int(b['bbox_xyxy'][0])}_{int(b['bbox_xyxy'][1])}" != did]
        else:
            fstate["rejected"].add(did)
            fstate["accepted"].discard(did)
        save_current_labels(file_id, file_dir)
        fstate["selected_door_id"] = None # Move to next
        st.rerun()
    if c3.button("Skip", use_container_width=True):
        if selected_idx < len(all_visible) - 1:
            fstate["selected_door_id"] = all_visible[selected_idx + 1]["id"]
            st.rerun()

    st.divider()
    st.write(f"**Stats:** {len(fstate['accepted'])} Acc, {len(fstate['rejected'])} Rej, {len(fstate['added_boxes'])} Add")
    
    # Train badge
    total_overrides = len(fstate["accepted"]) + len(fstate["rejected"]) + len(fstate["added_boxes"])
    if total_overrides >= 5:
        if st.button("Train Model", use_container_width=True):
            with st.spinner("Training..."):
                fit_reweighter(Path("artifacts"), Path("models/reweighter_v1.json"))
                st.success("Model updated!")
                st.cache_data.clear()
                st.rerun()

def run_pipeline(file_id: str, file_dir: Path, config_path: str):
    lib.update_status(file_id, "processing")
    try:
        # If Step 1 artifacts already exist (common for imported folders),
        # don't require `source.pdf` — just run Step 2.
        primitives_path = file_dir / "primitives.json"
        meta_path = file_dir / "meta.json"
        image_path = file_dir / "page.png"

        has_step1_artifacts = primitives_path.exists() and meta_path.exists() and image_path.exists()
        if not has_step1_artifacts:
            pdf_path = file_dir / "source.pdf"
            process_pdf(pdf_path, file_dir, dpi=400, page_index=0)

        run_step2(file_dir, Path(config_path))
        lib.update_status(file_id, "done")
        st.cache_data.clear()
    except Exception as e:
        lib.update_status(file_id, "error", str(e))

def save_current_labels(file_id: str, file_dir: Path):
    fstate = st.session_state.files[file_id]
    labels_to_save = {
        "schema_version": 1,
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accepted_ids": list(fstate["accepted"]),
        "rejected_ids": list(fstate["rejected"]),
        "added_boxes": fstate["added_boxes"],
        "notes": fstate["notes"]
    }
    save_labels(file_dir, labels_to_save)

# --- Layout ---

sidebar_library()

if "selected_file_id" in st.session_state and st.session_state.selected_file_id:
    items = lib.get_items()
    selected_item = next((i for i in items if i["id"] == st.session_state.selected_file_id), None)
    
    if selected_item:
        file_id = selected_item["id"]
        file_dir = Path(selected_item["path"])
        image, doors_data, labels_data, meta_data = load_file_artifacts(str(file_dir))
        init_file_state(file_id, doors_data, labels_data)
        fstate = st.session_state.files[file_id]

        st.markdown(f"### {selected_item['original_name']}")
        
        col_main, col_review = st.columns([2, 1])
        
        with col_main:
            canvas_result, active_doors = main_viewer_canvas(selected_item, image, doors_data, fstate)
            
        with col_review:
            main_viewer_controls(selected_item, image, doors_data, fstate, canvas_result)
            st.divider()
            right_panel_review(selected_item, image, doors_data, fstate, active_doors)
else:
    st.info("Select a file from the library to begin.")
