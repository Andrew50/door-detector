import hashlib
import html
import json
import logging
import math
import os
import shutil
import time
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
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

# Log to Streamlit's server console for debugging.
logger = logging.getLogger("door_detector.review_app")

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

    /* Main viewer: keep PDF title area a fixed two lines tall */
    .door_detector-pdf-title {
        --door_detector-title-font-size: 1.35rem;
        --door_detector-title-line-height: 1.55rem;

        /* Match existing heading spacing, but keep total height stable */
        padding-top: 0.5rem;
        margin: 0 0 0.75rem 0;

        /* Always reserve exactly two lines (plus the fixed top padding) */
        height: calc(0.5rem + (2 * var(--door_detector-title-line-height)));
        overflow: hidden;
    }

    .door_detector-pdf-title > h3 {
        font-size: var(--door_detector-title-font-size) !important;
        line-height: var(--door_detector-title-line-height) !important;

        margin: 0 !important;
        padding: 0 !important; /* override global .main h3 padding-top */

        /* Clamp to two lines; container keeps the reserved space either way */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        word-break: break-word;
    }

    /* Hide internal "door click sink" widgets (used for box click selection) */
    div[data-testid="stTextInput"]:has(input[aria-label^="door_click_sink_"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"] input[aria-label^="door_click_sink_"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
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

if "debug_perf" not in st.session_state:
    st.session_state.debug_perf = False

VIEWER_TARGET_WIDTH_PX = 1200
VIEWER_ASPECT_RATIO_HW = 0.75  # height/width

lib = st.session_state.library

# --- Session State Helpers ---
def _affine_apply(aff: Any, x: float, y: float) -> Tuple[float, float]:
    a, b, c, d, e, f = [float(v) for v in aff]
    px = a * x + c * y + e
    py = b * x + d * y + f
    return px, py


def _invert_affine(aff: Any) -> Optional[List[float]]:
    """Invert [a,b,c,d,e,f] where x' = a*x + c*y + e; y' = b*x + d*y + f."""
    try:
        a, b, c, d, e, f = [float(v) for v in aff]
    except Exception:
        return None
    det = a * d - b * c
    if abs(det) < 1e-12:
        return None
    return [
        d / det,
        -b / det,
        -c / det,
        a / det,
        (c * f - d * e) / det,
        (b * e - a * f) / det,
    ]


def _infer_bbox_origin_shift_from_transform(
    meta_data: Dict[str, Any], transform_data: Dict[str, Any]
) -> Optional[Tuple[float, float]]:
    """
    Infer a (dx, dy) shift that makes the transformed page bbox start at (0, 0).

    This repairs older/buggy transforms where rotation+scale are correct but the
    translation is off (common on rotated PDFs), yielding negative bboxes.
    """
    try:
        full_w = float(meta_data.get("pix_width", 0))
        full_h = float(meta_data.get("pix_height", 0))
        if full_w <= 0 or full_h <= 0:
            return None
    except Exception:
        return None

    page_rect = meta_data.get("page_rect") or transform_data.get("page_rect")
    aff = transform_data.get("pdf_to_pix_affine")
    if not isinstance(page_rect, dict) or not isinstance(aff, list) or len(aff) != 6:
        return None

    try:
        x0 = float(page_rect["x0"])
        y0 = float(page_rect["y0"])
        x1 = float(page_rect["x1"])
        y1 = float(page_rect["y1"])
    except Exception:
        return None

    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    pts = [_affine_apply(aff, x, y) for x, y in corners]
    min_x = min(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_x = max(p[0] for p in pts)
    max_y = max(p[1] for p in pts)
    w = max_x - min_x
    h = max_y - min_y

    # If the transformed page bbox doesn't match the raster dimensions, don't guess.
    tol = 8.0  # pixels (float math + rounding tolerance)
    if abs(w - full_w) > tol or abs(h - full_h) > tol:
        return None

    dx = -min_x
    dy = -min_y
    if abs(dx) < 1e-6:
        dx = 0.0
    if abs(dy) < 1e-6:
        dy = 0.0

    return (dx, dy)


def _count_offscreen_doors(doors: List[Dict[str, Any]], *, w: float, h: float) -> int:
    off = 0
    for d in doors:
        bbox = d.get("bbox_xyxy")
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except Exception:
            continue
        x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
        y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
        if x1 < 0 or y1 < 0 or x0 > w or y0 > h:
            off += 1
    return off


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
    doors_path = file_dir / "doors.json"
    labels_path = file_dir / "labels.json"
    meta_path = file_dir / "meta.json"
    transform_path = file_dir / "transform.json"
    
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

    transform_data = {}
    if transform_path.exists():
        try:
            with open(transform_path) as f:
                transform_data = json.load(f)
        except Exception:
            transform_data = {}

    # Repair common transform issues so bboxes align to (0,0)-(pix_width,pix_height).
    #
    # Background:
    # - Older artifacts stored bboxes in a pixel space that effectively applied a rotated
    #   PDF→pix affine. In our pipeline, vector primitives from PyMuPDF are in cropbox
    #   coordinates, so the "correct" pixel space for visualization is simply:
    #     pix = pdf * scale  (no extra rotation terms).
    # - For those older artifacts, we can reproject:
    #     old_pix -> pdf (via inverse affine) -> correct_pix (via scale).
    try:
        if isinstance(doors_data.get("doors"), list) and transform_data:
            doors_list = doors_data["doors"]
            full_w = float(meta_data.get("pix_width", 0) or 0)
            full_h = float(meta_data.get("pix_height", 0) or 0)

            aff = transform_data.get("pdf_to_pix_affine")
            inv = _invert_affine(aff) if aff is not None else None
            scale = float(transform_data.get("scale", 1.0) or 1.0)

            did_reproject = False
            if isinstance(aff, list) and len(aff) == 6 and inv and scale > 0:
                # If the affine includes rotation/shear terms, it's a strong signal of a legacy
                # (misaligned) artifact. Reproject to the canonical cropbox-pixel space.
                a, b, c, d, e, f = [float(v) for v in aff]
                needs_reproject = (abs(b) > 1e-6) or (abs(c) > 1e-6)
                if needs_reproject:
                    for door in doors_list:
                        bbox = door.get("bbox_xyxy")
                        try:
                            x0, y0, x1, y1 = [float(v) for v in bbox]
                        except Exception:
                            continue

                        # Transform both corners back to PDF(cropbox) coords, then rescale.
                        px0, py0 = _affine_apply(inv, x0, y0)
                        px1, py1 = _affine_apply(inv, x1, y1)
                        door["bbox_xyxy"] = [px0 * scale, py0 * scale, px1 * scale, py1 * scale]

                    did_reproject = True

            # If we didn't reproject, fall back to a pure origin shift if it looks like just
            # a translation issue.
            if not did_reproject and full_w > 0 and full_h > 0:
                dx_dy = _infer_bbox_origin_shift_from_transform(meta_data, transform_data)
                if dx_dy:
                    dx, dy = dx_dy
                    if dx != 0.0 or dy != 0.0:
                        for door in doors_list:
                            bbox = door.get("bbox_xyxy")
                            try:
                                x0, y0, x1, y1 = [float(v) for v in bbox]
                            except Exception:
                                continue
                            door["bbox_xyxy"] = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]

            pre_off = _count_offscreen_doors(doors_list, w=full_w, h=full_h) if (full_w > 0 and full_h > 0) else 0
            if did_reproject:
                doors_data["_bbox_transform_fix"] = {"mode": "reproject_inv_affine_rescale"}
                logger.info("Reprojected door bboxes for %s (remaining offscreen=%d)", file_dir_str, pre_off)
    except Exception as e:
        logger.info("BBox origin shift repair skipped for %s (%s)", file_dir_str, e)

    return doors_data, labels_data, meta_data


def _get_full_page_dims(meta_data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Return (width, height) for the full-resolution page (page.png), if known."""
    try:
        w = int(meta_data.get("pix_width"))
        h = int(meta_data.get("pix_height"))
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        return None
    return None


def _get_preview_path(file_dir: Path) -> Path:
    # Stable filename so existing artifacts benefit after first view.
    return file_dir / "page_view.jpg"


@st.cache_resource
def get_or_create_page_preview(
    file_dir_str: str,
    *,
    full_width: Optional[int],
    full_height: Optional[int],
    page_png_mtime_ns: int,
    preview_max_width: int = 2400,
) -> Optional[Dict[str, Any]]:
    """Return a lightweight preview image spec for UI rendering.

    Creates `page_view.jpg` on disk if missing by downscaling `page.png` once.
    Returns:
      { path, url, width, height, scale }
    Where `scale` maps full-res pixel coords -> preview coords.
    """
    file_dir = Path(file_dir_str)
    page_png_path = file_dir / "page.png"
    if not page_png_path.exists():
        return None

    preview_path = _get_preview_path(file_dir)

    # Create preview lazily (once) to avoid re-decoding huge PNG on every rerun.
    preview_is_stale = False
    if preview_path.exists():
        try:
            preview_is_stale = preview_path.stat().st_mtime_ns < int(page_png_mtime_ns)
        except Exception:
            preview_is_stale = False

    if (not preview_path.exists()) or preview_is_stale:
        src = Image.open(page_png_path)
        try:
            src_w, src_h = src.size
            # If meta is missing, fall back to the source image dimensions.
            if not full_width or not full_height:
                full_width, full_height = src_w, src_h

            scale_create = min(1.0, float(preview_max_width) / float(src_w)) if src_w else 1.0
            if scale_create < 1.0:
                out_w = max(1, int(round(src_w * scale_create)))
                out_h = max(1, int(round(src_h * scale_create)))
                prev = src.resize((out_w, out_h), Image.LANCZOS)
            else:
                # Still write a JPEG so the UI never has to decode the huge PNG.
                prev = src

            if prev.mode != "RGB":
                prev = prev.convert("RGB")

            preview_path.parent.mkdir(parents=True, exist_ok=True)
            prev.save(preview_path, "JPEG", quality=88, optimize=True, progressive=True)
        finally:
            try:
                src.close()
            except Exception:
                pass

    prev_img = Image.open(preview_path)
    try:
        prev_w, prev_h = prev_img.size
    finally:
        try:
            prev_img.close()
        except Exception:
            pass

    # Compute a stable scale factor from full-res → preview coords.
    if full_width and full_width > 0:
        scale = float(prev_w) / float(full_width)
    else:
        # Last-resort: treat preview as full-res (should be rare).
        scale = 1.0

    return {
        "path": str(preview_path),
        "width": int(prev_w),
        "height": int(prev_h),
        "scale": float(scale),
    }

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

def _normalize_bbox_xyxy(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return (x0, y0, x1, y1) with x0<=x1 and y0<=y1, or None if invalid."""
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

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
    if fstate.get("viewer_mode") == "Off":
        return ""

    parts: List[str] = []
    highlight_selected = fstate.get("viewer_mode") == "Highlight Selected"
    selected_id = fstate.get("selected_door_id")

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    for d in active_doors:
        did = d.get("id")
        is_selected = did == selected_id
        if highlight_selected and not is_selected:
            continue

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

        # Match existing color semantics.
        stroke = "#ffa500"  # undecided (orange)
        if did in fstate.get("accepted", set()):
            stroke = "#00ff00"
        if d.get("is_user_added"):
            stroke = "#00ffff"
        if is_selected:
            stroke = "#ff4b4b"  # selected (red)

        stroke_width = 2 if not is_selected else 3
        did_attr = html.escape(str(did), quote=True)
        parts.append(
            f'<rect x="{x0p:.2f}" y="{y0p:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'vector-effect="non-scaling-stroke" '
            f'data-door-id="{did_attr}" style="pointer-events: all; cursor: pointer;" />'
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
) -> None:
    # This viewer provides:
    # - scrollwheel zoom (centered at cursor)
    # - click+drag pan
    # - initial fit-to-container with letterboxing
    click_sink_aria_label_esc = html.escape(click_sink_aria_label, quote=True)
    viewer_html = f"""
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
    content.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
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

  function setSelectedDoorId(doorId) {{
    if (!doorId) return;
    let input = null;
    try {{
      input = window.parent?.document?.querySelector('input[aria-label="{click_sink_aria_label_esc}"]');
    }} catch (_) {{
      input = null;
    }}
    if (!input) return;
    try {{
      input.value = String(doorId);
      // React/Streamlit listens for "input" events.
      input.dispatchEvent(new Event("input", {{ bubbles: true }}));
      input.dispatchEvent(new Event("change", {{ bubbles: true }}));
    }} catch (_) {{}}
  }}

  if (svg) {{
    // If a door bbox is clicked, select it (and don't start a pan drag).
    svg.addEventListener("pointerdown", (e) => {{
      const t = e.target;
      if (t && t.getAttribute && t.getAttribute("data-door-id")) {{
        e.preventDefault();
        e.stopPropagation();
      }}
    }}, true);

    svg.addEventListener("click", (e) => {{
      const t = e.target;
      if (!t || !t.getAttribute) return;
      const did = t.getAttribute("data-door-id");
      if (!did) return;
      e.preventDefault();
      e.stopPropagation();
      setSelectedDoorId(did);
    }});
  }}
}})();
</script>
"""
    components.html(viewer_html, height=height, scrolling=False)

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

    with st.sidebar.expander("Debug", expanded=False):
        st.checkbox("Show perf timings", key="debug_perf")

def main_viewer_canvas(
    item: Dict,
    *,
    preview_spec: Optional[Dict[str, Any]],
    full_dims: Optional[Tuple[int, int]],
    doors_data: Dict,
    fstate: Dict,
):
    file_id = item["id"]
    file_dir = Path(item["path"])
    
    if preview_spec:
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

        # Door click sink: JS writes the clicked door id into this hidden widget,
        # which triggers a rerun; on rerun we sync it into the "Jump to door" control.
        click_sink_label = f"door_click_sink_{file_id}"
        click_sink_key = click_sink_label
        st.text_input(click_sink_label, key=click_sink_key, label_visibility="collapsed")
        clicked_id = st.session_state.get(click_sink_key)
        active_ids = {d.get("id") for d in active_doors}
        if clicked_id:
            if clicked_id in active_ids and clicked_id != fstate.get("_last_clicked_door_id"):
                fstate["_last_clicked_door_id"] = clicked_id
                fstate["selected_door_id"] = clicked_id
                st.session_state[f"jump_{file_id}"] = clicked_id

        # Keep the main viewer highlight in sync with the "Jump to door" selector.
        # Otherwise, the right panel can advance selection in the same run (Prev/Next),
        # but the main viewer would still render using the previous fstate selection.
        jump_key = f"jump_{file_id}"
        jump_id = st.session_state.get(jump_key)
        if jump_id in active_ids and jump_id != fstate.get("selected_door_id"):
            fstate["selected_door_id"] = jump_id

        viewer_width_hint = int(VIEWER_TARGET_WIDTH_PX)
        viewer_width_hint = max(600, min(2000, viewer_width_hint))
        aspect = float(VIEWER_ASPECT_RATIO_HW)
        aspect = max(0.35, min(1.25, aspect))

        # Viewer height derived from width and a fixed aspect ratio.
        viewer_height = int(round(viewer_width_hint * aspect))
        viewer_height = max(450, min(1400, viewer_height))

        # NOTE: Don't wrap this in `st.container(height=...)` because Streamlit makes that
        # container scrollable (adds a scrollbar) which steals scroll/drag interactions.
        if fstate["viewer_mode"] == "Add Door":
            # Use the (smaller) preview for interactive drawing to keep the UI snappy.
            bg_img = Image.open(preview_spec["path"])

            # Fit-to-container (approx) for first render.
            fit_scale = min(1.0, min(viewer_width_hint / bg_img.width, viewer_height / bg_img.height))
            display_width = max(400, int(bg_img.width * fit_scale))
            display_height = max(400, int(bg_img.height * fit_scale))

            # Resize background for performance.
            bg_img = bg_img.resize((display_width, display_height), Image.LANCZOS)

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
        )
        return None, active_doors
    else:
        st.info("Run analysis to see results.")
        return None, []

def main_viewer_controls(
    item: Dict,
    *,
    full_dims: Optional[Tuple[int, int]],
    doors_data: Dict,
    fstate: Dict,
    canvas_result: Any,
):
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

    if fstate["viewer_mode"] == "Add Door":
        st.info("Draw rectangles on the PDF.")
        if st.button("Save Added Doors", type="primary", use_container_width=True):
            if canvas_result and canvas_result.json_data:
                objects = canvas_result.json_data["objects"]
                display_width = canvas_result.image_data.shape[1] if canvas_result.image_data is not None else 1
                display_height = canvas_result.image_data.shape[0] if canvas_result.image_data is not None else 1
                if full_dims:
                    full_w, full_h = full_dims
                else:
                    full_w, full_h = display_width, display_height

                scale_x = full_w / display_width if display_width else 1.0
                scale_y = full_h / display_height if display_height else 1.0
                
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

def right_panel_review(
    item: Dict,
    *,
    preview_spec: Optional[Dict[str, Any]],
    doors_data: Dict,
    fstate: Dict,
    active_doors: List,
):
    file_id = item["id"]
    file_dir = Path(item["path"])
    
    # Use pre-calculated active_doors (which already includes added_boxes)
    all_visible = active_doors.copy()
    all_visible.sort(key=lambda x: x["confidence"], reverse=True)
    
    st.subheader(f"Doors ({len(all_visible)})")
    
    if not all_visible:
        return

    # Jump-to selector (replaces click-to-select in the main viewer)
    door_ids = [d["id"] for d in all_visible]
    id_to_label = {
        d["id"]: f"{i+1}/{len(all_visible)}  {d['type']}  {d['confidence']:.3f}  {d['id']}"
        for i, d in enumerate(all_visible)
    }

    # Streamlit selectbox keeps its own state keyed by `key=...`.
    # Treat that as canonical, but keep it in sync with our per-file fstate.
    jump_key = f"jump_{file_id}"

    current_id = None
    if st.session_state.get(jump_key) in door_ids:
        current_id = st.session_state[jump_key]
    elif fstate.get("selected_door_id") in door_ids:
        current_id = fstate["selected_door_id"]
    else:
        current_id = door_ids[0]

    # Prev/Next (wrap-around).
    #
    # Avoid callbacks: Streamlit may run callbacks after widget instantiation,
    # which can trigger "cannot be modified..." if we touch `st.session_state[jump_key]`.
    col_p, col_idx, col_n = st.columns([1, 2, 1])
    prev_clicked = col_p.button("Prev", use_container_width=True, key=f"{jump_key}__prev")
    next_clicked = col_n.button("Next", use_container_width=True, key=f"{jump_key}__next")

    if prev_clicked or next_clicked:
        delta = -1 if prev_clicked else 1
        idx = door_ids.index(current_id)
        current_id = door_ids[(idx + delta) % len(door_ids)]

    # Must happen before the selectbox is instantiated.
    if st.session_state.get(jump_key) != current_id:
        st.session_state[jump_key] = current_id
    fstate["selected_door_id"] = current_id
    selected_idx = door_ids.index(current_id)

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

    col_idx.write(f"<div style='text-align: center; line-height: 38px;'>{selected_idx + 1} / {len(all_visible)}</div>", unsafe_allow_html=True)

    st.divider()
    
    # Details of selected
    selected_door = all_visible[selected_idx]
    did = selected_door["id"]
    
    st.write(f"**ID:** `{did}` | **Type:** {selected_door['type']} | **Conf:** {selected_door['confidence']:.3f}")
    
    # Zoom
    if preview_spec:
        image = Image.open(preview_spec["path"])
        bbox = selected_door["bbox_xyxy"]
        nb = _normalize_bbox_xyxy(bbox)
        if nb is None:
            st.warning("Selected door has an invalid bbox; preview unavailable.")
            return
        x0, y0, x1, y1 = nb
        scale = float(preview_spec.get("scale", 1.0))
        pad_full = 100.0
        pad = pad_full * scale

        left = max(0, int(math.floor(x0 * scale - pad)))
        upper = max(0, int(math.floor(y0 * scale - pad)))
        right = min(image.width, int(math.ceil(x1 * scale + pad)))
        lower = min(image.height, int(math.ceil(y1 * scale + pad)))
        if right <= left or lower <= upper:
            st.warning("Selected door bbox is degenerate after clamping; preview unavailable.")
            logger.info(
                "Degenerate preview crop file_id=%s door_id=%s bbox=%s nb=%s scale=%.6f pad_full=%.1f crop=(%d,%d,%d,%d) preview=%dx%d shift=%s",
                file_id,
                did,
                bbox,
                nb,
                scale,
                pad_full,
                left,
                upper,
                right,
                lower,
                image.width,
                image.height,
                doors_data.get("_bbox_transform_fix") or doors_data.get("_bbox_origin_shift"),
            )
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
        perf: Dict[str, float] = {}

        t0 = time.perf_counter()
        doors_data, labels_data, meta_data = load_file_artifacts(str(file_dir))
        perf["load_file_artifacts_ms"] = (time.perf_counter() - t0) * 1000.0
        init_file_state(file_id, doors_data, labels_data)
        fstate = st.session_state.files[file_id]

        full_dims = _get_full_page_dims(meta_data)
        t1 = time.perf_counter()
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
        perf["get_or_create_page_preview_ms"] = (time.perf_counter() - t1) * 1000.0

        title = html.escape(str(selected_item.get("original_name", "")))
        st.markdown(f"<div class='door_detector-pdf-title'><h3>{title}</h3></div>", unsafe_allow_html=True)
        
        col_main, col_review = st.columns([2, 1])
        
        with col_main:
            t2 = time.perf_counter()
            canvas_result, active_doors = main_viewer_canvas(
                selected_item,
                preview_spec=preview_spec,
                full_dims=full_dims,
                doors_data=doors_data,
                fstate=fstate,
            )
            perf["render_main_panel_ms"] = (time.perf_counter() - t2) * 1000.0
            
        with col_review:
            t3 = time.perf_counter()
            main_viewer_controls(
                selected_item,
                full_dims=full_dims,
                doors_data=doors_data,
                fstate=fstate,
                canvas_result=canvas_result,
            )
            st.divider()
            right_panel_review(
                selected_item,
                preview_spec=preview_spec,
                doors_data=doors_data,
                fstate=fstate,
                active_doors=active_doors,
            )
            perf["render_right_panel_ms"] = (time.perf_counter() - t3) * 1000.0

        if st.session_state.get("debug_perf"):
            with st.sidebar.expander("Perf timings (server)", expanded=True):
                for k, v in perf.items():
                    st.write(f"**{k}**: {v:.1f} ms")
else:
    st.info("Select a file from the library to begin.")
