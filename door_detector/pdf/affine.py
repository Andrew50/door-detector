"""Small helpers for applying 2D affine transforms.

We store transforms as 6-tuples/lists in the common PDF / SVG form:
  [a, b, c, d, e, f]
meaning:
  x' = a*x + c*y + e
  y' = b*x + d*y + f
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


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

