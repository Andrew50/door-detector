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

