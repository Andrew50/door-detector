"""Reading/writing artifacts for the Streamlit UI."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from PIL import Image, UnidentifiedImageError

from door_detector.ui.labels import (
    LABELS_SCHEMA_VERSION,
    labels_v4_default,
    migrate_labels_v2_to_v4,
    migrate_labels_v3_to_v4,
    validate_labels_v4_or_raise,
)

def _mtime_ns(p: Path) -> int:
    try:
        return int(p.stat().st_mtime_ns)
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def _load_json_cached(path: str, *, mtime_ns: int) -> Dict[str, Any]:
    """Load JSON from disk, cached by mtime.

    IMPORTANT: Callers must treat returned objects as read-only and copy before mutating.
    """
    return json.loads(Path(path).read_bytes())


def _coerce_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        if x is None:
            continue
        s = str(x)
        if s:
            out.append(s)
    return out


def _remap_labels_ids_in_place(labels_data: Dict[str, Any], doors_data: Dict[str, Any]) -> None:
    """Remap label ids to current candidate ids using per-candidate legacy ids.

    This supports a one-time seamless migration when the candidate id scheme changes.
    """
    try:
        candidates = list(doors_data.get("candidates", doors_data.get("doors", [])) or [])
        legacy_to_current: Dict[str, str] = {}
        for c in candidates:
            cid = c.get("id")
            if cid is None:
                continue
            cid_s = str(cid)
            if not cid_s:
                continue
            legacy_to_current[cid_s] = cid_s
            for lid in _coerce_str_list(c.get("legacy_ids")):
                legacy_to_current[lid] = cid_s

        if not legacy_to_current:
            return

        def _remap_list(xs: list[Any]) -> list[str]:
            out: list[str] = []
            for x in xs:
                if x is None:
                    continue
                s = str(x)
                if not s:
                    continue
                out.append(legacy_to_current.get(s, s))
            return out

        # v2: confirmed_ids
        if isinstance(labels_data.get("confirmed_ids"), list):
            labels_data["confirmed_ids"] = _remap_list(labels_data.get("confirmed_ids") or [])
        # v3: confirmed_by_type
        if isinstance(labels_data.get("confirmed_by_type"), dict):
            cbt = labels_data.get("confirmed_by_type") or {}
            for k, v in list(cbt.items()):
                if isinstance(v, list):
                    cbt[k] = _remap_list(v)
            labels_data["confirmed_by_type"] = cbt
        # v4: rejected_by_type
        if isinstance(labels_data.get("rejected_by_type"), dict):
            rbt = labels_data.get("rejected_by_type") or {}
            for k, v in list(rbt.items()):
                if isinstance(v, list):
                    rbt[k] = _remap_list(v)
            labels_data["rejected_by_type"] = rbt
        if isinstance(labels_data.get("deleted_ids"), list):
            labels_data["deleted_ids"] = _remap_list(labels_data.get("deleted_ids") or [])
        if isinstance(labels_data.get("manual_additions"), list):
            for rec in labels_data.get("manual_additions") or []:
                if not isinstance(rec, dict):
                    continue
                sid = rec.get("snapped_candidate_id")
                if sid is None:
                    continue
                s = str(sid)
                if s and s in legacy_to_current:
                    rec["snapped_candidate_id"] = legacy_to_current[s]
    except Exception:
        return


def load_file_artifacts(file_dir_str: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    file_dir = Path(file_dir_str)
    doors_path = file_dir / "doors.json"
    labels_path = file_dir / "labels.json"
    meta_path = file_dir / "meta.json"

    doors_data: Dict[str, Any] = {}
    if doors_path.exists():
        try:
            doors_data = _load_json_cached(str(doors_path), mtime_ns=_mtime_ns(doors_path))
        except Exception:
            doors_data = {}

    labels_data: Dict[str, Any] = labels_v4_default()
    if labels_path.exists():
        try:
            # Copy defensively because we mutate (remap + migrations + validation).
            labels_data = copy.deepcopy(_load_json_cached(str(labels_path), mtime_ns=_mtime_ns(labels_path)))
        except Exception:
            labels_data = labels_v4_default()
        # If `doors.json` uses a newer id scheme, remap old label ids to current
        # candidate ids (without changing the UX).
        _remap_labels_ids_in_place(labels_data, doors_data)

        # Migrate older schemas on load (UI uses v4 internally).
        if isinstance(labels_data, dict) and labels_data.get("schema_version") == 2:
            labels_data = migrate_labels_v2_to_v4(labels_data)
        if isinstance(labels_data, dict) and labels_data.get("schema_version") == 3:
            labels_data = migrate_labels_v3_to_v4(labels_data)

        validate_labels_v4_or_raise(labels_data, labels_path=labels_path)

    meta_data: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta_data = _load_json_cached(str(meta_path), mtime_ns=_mtime_ns(meta_path))
        except Exception:
            meta_data = {}

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


def _delete_invalid_library_item(*, file_dir: Path, err: BaseException) -> None:
    """Best-effort removal of a broken item from the UI library.

    This is intentionally defensive: it should never raise, because it's invoked
    from image loading paths where we want to recover gracefully.
    """
    file_id = str(file_dir.name or "")
    _ = (file_id, err)  # Logging intentionally suppressed.

    # Remove from library index if possible (preferred).
    try:
        lib = st.session_state.get("library")
        if lib is not None and file_id:
            try:
                lib.delete_item(file_id)
            except Exception:
                pass
    except Exception:
        pass

    # Ensure the on-disk folder is gone even if the index entry wasn't found.
    try:
        if file_dir.exists():
            shutil.rmtree(file_dir)
    except Exception:
        pass

    # Reset selection if the deleted item was selected.
    try:
        if file_id and st.session_state.get("selected_file_id") == file_id:
            st.session_state.selected_file_id = None
    except Exception:
        pass

    # Drop any per-file in-memory state (safe to ignore if absent).
    try:
        files = st.session_state.get("files")
        if file_id and isinstance(files, dict):
            files.pop(file_id, None)
    except Exception:
        pass

    # Clear caches so we don't keep trying to use stale image paths.
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass


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
        try:
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
        except (OSError, UnidentifiedImageError) as e:
            # e.g. "OSError: image file is truncated" (corrupt/partial write)
            _delete_invalid_library_item(file_dir=file_dir, err=e)
            return None
        except Exception as e:
            _delete_invalid_library_item(file_dir=file_dir, err=e)
            return None

    if not preview_path.exists():
        return None

    try:
        prev_img = Image.open(preview_path)
        try:
            prev_w, prev_h = prev_img.size
        finally:
            try:
                prev_img.close()
            except Exception:
                pass
    except (OSError, UnidentifiedImageError) as e:
        _delete_invalid_library_item(file_dir=file_dir, err=e)
        return None
    except Exception as e:
        _delete_invalid_library_item(file_dir=file_dir, err=e)
        return None

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

