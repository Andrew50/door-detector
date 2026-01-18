"""Reading/writing artifacts for the Streamlit UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from PIL import Image

from door_detector.ui.labels import labels_v2_default, validate_labels_v2_or_raise


def load_file_artifacts(file_dir_str: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    file_dir = Path(file_dir_str)
    doors_path = file_dir / "doors.json"
    labels_path = file_dir / "labels.json"
    meta_path = file_dir / "meta.json"

    doors_data: Dict[str, Any] = {}
    if doors_path.exists():
        with open(doors_path) as f:
            doors_data = json.load(f)

    labels_data: Dict[str, Any] = labels_v2_default()
    if labels_path.exists():
        with open(labels_path) as f:
            labels_data = json.load(f)
        validate_labels_v2_or_raise(labels_data, labels_path=labels_path)

    meta_data: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta_data = json.load(f)

    return doors_data, labels_data, meta_data


def get_full_page_dims(meta_data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
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


@st.cache_resource(show_spinner=False)
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
      { path, width, height, scale }
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

