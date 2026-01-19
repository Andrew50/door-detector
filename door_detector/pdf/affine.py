"""Small helpers for applying 2D affine transforms.

We store transforms as 6-tuples/lists in the common PDF / SVG form:
  [a, b, c, d, e, f]
meaning:
  x' = a*x + c*y + e
  y' = b*x + d*y + f
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple


Affine = Sequence[float]
BBox = Sequence[float]  # [x0,y0,x1,y1]


def apply_affine_xy(m: Affine, x: float, y: float) -> Tuple[float, float]:
    a, b, c, d, e, f = [float(v) for v in m]
    return (a * x + c * y + e, b * x + d * y + f)


def apply_affine_bbox_xyxy(m: Affine, bbox_xyxy: BBox) -> List[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    # Apply to all corners then take axis-aligned bounds in the target space.
    p0 = apply_affine_xy(m, x0, y0)
    p1 = apply_affine_xy(m, x0, y1)
    p2 = apply_affine_xy(m, x1, y0)
    p3 = apply_affine_xy(m, x1, y1)
    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_bbox_xyxy(bbox_xyxy: Iterable[float]) -> List[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def flip_bbox_y_xyxy(bbox_xyxy: Iterable[float], *, y0: float, y1: float) -> List[float]:
    """Flip bbox Y coordinates about the horizontal axis defined by [y0, y1].

    Useful for converting between coordinate systems that differ only by the Y axis
    direction (e.g. PyMuPDF/fitz Y-down vs PDF-spec Y-up).
    """
    x0b, y0b, x1b, y1b = normalize_bbox_xyxy(bbox_xyxy)
    y_sum = float(y0) + float(y1)
    fy0 = y_sum - float(y1b)
    fy1 = y_sum - float(y0b)
    return [float(x0b), float(fy0), float(x1b), float(fy1)]


def _infer_pdfjs_y_bounds_from_fitz_cropbox(cropbox: Any) -> Tuple[float, float]:
    """Infer PDF.js (PDF-spec) Y bounds from a PyMuPDF cropbox.

    PyMuPDF exposes page geometry in a Y-down coordinate system whose Y origin is
    at the *top* of the cropbox. PDF.js expects PDF-spec coords (Y-up) in the
    original PDF page coordinate system, which can be either:

    - Typical pages: y in [0, H]
    - CAD-style centered pages: y in [-H/2, H/2]

    We infer the centered case via X bounds: if x spans negative→positive and is
    symmetric around 0, assume the PDF page coordinate system is centered.
    """
    if not isinstance(cropbox, dict):
        return (0.0, 0.0)
    try:
        x0 = float(cropbox.get("x0", 0.0) or 0.0)
        x1 = float(cropbox.get("x1", 0.0) or 0.0)
        y0 = float(cropbox.get("y0", 0.0) or 0.0)
        y1 = float(cropbox.get("y1", 0.0) or 0.0)
    except Exception:
        return (0.0, 0.0)
    w = float(x1 - x0)
    h = float(y1 - y0)
    if not (w > 0 and h > 0):
        return (0.0, 0.0)

    # Centered-X heuristic (robust to float rounding).
    # Example: x0=-1512.12, x1=1512.12 => centered.
    centered_x = (x0 < 0.0) and (x1 > 0.0) and (abs(x0 + x1) <= max(1.0, 1e-3 * abs(w)))
    if centered_x:
        return (-0.5 * h, 0.5 * h)
    return (0.0, h)


def fitz_bbox_to_pdfjs_bbox_xyxy(bbox_fitz_xyxy: Iterable[float], *, cropbox: Any) -> List[float]:
    """Convert a bbox from PyMuPDF/fitz coords (Y-down) → PDF.js PDF coords (Y-up).

    This accounts for the common case where the PDF page coordinate system is
    centered (negative Y values) even though PyMuPDF reports a 0..H Y range.
    """
    x0, y0, x1, y1 = normalize_bbox_xyxy(bbox_fitz_xyxy)
    if not isinstance(cropbox, dict):
        return [float(x0), float(y0), float(x1), float(y1)]
    try:
        cy0 = float(cropbox.get("y0", 0.0) or 0.0)
    except Exception:
        cy0 = 0.0
    y_min, y_max = _infer_pdfjs_y_bounds_from_fitz_cropbox(cropbox)
    # Interpret fitz y as "down from the cropbox top".
    y0_rel = float(y0) - float(cy0)
    y1_rel = float(y1) - float(cy0)
    # PDF y is "up from the PDF origin", where the top of the page is y_max.
    py0 = float(y_max) - float(y1_rel)
    py1 = float(y_max) - float(y0_rel)
    return [float(x0), float(py0), float(x1), float(py1)]


def pdfjs_bbox_to_fitz_bbox_xyxy(bbox_pdf_xyxy: Iterable[float], *, cropbox: Any) -> List[float]:
    """Convert a bbox from PDF.js PDF coords (Y-up) → PyMuPDF/fitz coords (Y-down)."""
    x0, y0, x1, y1 = normalize_bbox_xyxy(bbox_pdf_xyxy)
    if not isinstance(cropbox, dict):
        return [float(x0), float(y0), float(x1), float(y1)]
    try:
        cy0 = float(cropbox.get("y0", 0.0) or 0.0)
    except Exception:
        cy0 = 0.0
    _y_min, y_max = _infer_pdfjs_y_bounds_from_fitz_cropbox(cropbox)
    # Inverse of fitz_bbox_to_pdfjs_bbox_xyxy:
    # py = y_max - (fitz_y - cy0)  => fitz_y = cy0 + (y_max - py)
    fy0 = float(cy0) + (float(y_max) - float(y1))
    fy1 = float(cy0) + (float(y_max) - float(y0))
    return [float(x0), float(fy0), float(x1), float(fy1)]

