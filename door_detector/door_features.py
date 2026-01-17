"""Geometry helpers for door detection."""

import math
from typing import List, Tuple

import numpy as np


def sample_bezier(p0, p1, p2, p3, num_points: int = 17) -> List[Tuple[float, float]]:
    """Sample points along a cubic Bezier curve."""
    pts = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # B(t) = (1-t)^3 p0 + 3(1-t)^2 t p1 + 3(1-t) t^2 p2 + t^3 p3
        x = (
            (1 - t) ** 3 * p0["x"]
            + 3 * (1 - t) ** 2 * t * p1["x"]
            + 3 * (1 - t) * t**2 * p2["x"]
            + t**3 * p3["x"]
        )
        y = (
            (1 - t) ** 3 * p0["y"]
            + 3 * (1 - t) ** 2 * t * p1["y"]
            + 3 * (1 - t) * t**2 * p2["y"]
            + t**3 * p3["y"]
        )
        pts.append((x, y))
    return pts


def fit_circle(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], float, float]:
    """
    Fit a circle to a set of points using least squares.
    Returns (center_x, center_y), radius, and RMSE.
    """
    if len(points) < 3:
        return (0.0, 0.0), 0.0, 1e9

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Linear least squares for circle fit
    # (x - xc)^2 + (y - yc)^2 = R^2
    # x^2 - 2x*xc + xc^2 + y^2 - 2y*yc + yc^2 = R^2
    # 2x*xc + 2y*yc + (R^2 - xc^2 - yc^2) = x^2 + y^2
    # A * [xc, yc, k]^T = B
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    B = x**2 + y**2

    try:
        # Use pseudo-inverse or lstsq
        res = np.linalg.lstsq(A, B, rcond=None)[0]
        xc, yc, k = res
        radius = math.sqrt(max(0, k + xc**2 + yc**2))

        # Compute RMSE
        distances = np.sqrt((x - xc)**2 + (y - yc)**2)
        rmse = np.sqrt(np.mean((distances - radius)**2))

        return (xc, yc), radius, rmse
    except Exception:
        return (0.0, 0.0), 0.0, 1e9


def get_arc_angle_span(points: List[Tuple[float, float]], center: Tuple[float, float]) -> float:
    """Compute the angle span (in degrees) of an arc around a center point."""
    if not points:
        return 0.0
    
    p_start = points[0]
    p_end = points[-1]
    
    a_start = math.atan2(p_start[1] - center[1], p_start[0] - center[0])
    a_end = math.atan2(p_end[1] - center[1], p_end[0] - center[0])
    
    # Span is the absolute difference, handled for wrapping
    diff = abs(a_end - a_start)
    if diff > math.pi:
        diff = 2 * math.pi - diff
        
    return math.degrees(diff)


def dist_point_to_point(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_bbox(points: List[Tuple[float, float]]) -> List[float]:
    """Get [x0, y0, x1, y1] bounding box for a set of points."""
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return [float(min(x)), float(min(y)), float(max(x)), float(max(y))]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Intersection over Union for two [x0, y0, x1, y1] boxes."""
    inter_x0 = max(box1[0], box2[0])
    inter_y0 = max(box1[1], box2[1])
    inter_x1 = min(box1[2], box2[2])
    inter_y1 = min(box1[3], box2[3])
    
    if inter_x1 < inter_x0 or inter_y1 < inter_y0:
        return 0.0
        
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
        
    return inter_area / union_area


