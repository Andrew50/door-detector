"""Core door detection logic using vector primitives."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from door_detector.door_features import (
    compute_iou,
    dist_point_to_point,
    fit_circle,
    get_arc_angle_span,
    get_bbox,
    sample_bezier,
)


class SpatialIndex:
    """Simple grid-based spatial index for fast neighborhood searches."""
    def __init__(self, cell_size: float = 100.0):
        self.cell_size = cell_size
        self.grid = {}

    def add(self, item_id: int, bbox: List[float]):
        x0, y0, x1, y1 = bbox
        for ix in range(int(x0 // self.cell_size), int(x1 // self.cell_size) + 1):
            for iy in range(int(y0 // self.cell_size), int(y1 // self.cell_size) + 1):
                self.grid.setdefault((ix, iy), []).append(item_id)

    def query(self, bbox: List[float]) -> List[int]:
        x0, y0, x1, y1 = bbox
        results = set()
        for ix in range(int(x0 // self.cell_size), int(x1 // self.cell_size) + 1):
            for iy in range(int(y0 // self.cell_size), int(y1 // self.cell_size) + 1):
                results.update(self.grid.get((ix, iy), []))
        return list(results)


def apply_reweighter(candidates: List[Dict[str, Any]], model_path: str) -> List[Dict[str, Any]]:
    """Apply a learned reweighter to update candidate confidence scores."""
    try:
        with open(model_path) as f:
            model = json.load(f)
        
        weights = np.array(model["weights"])
        bias = model["bias"]
        feature_order = model["feature_order"]
        scaler = model["scaler"]
        
        means = np.array(scaler["mean"])
        stds = np.array(scaler["std"])
        
        for cand in candidates:
            # Build feature vector
            x = []
            for feat_name in feature_order:
                val = cand["features"].get(feat_name, 0.0)
                x.append(val)
            
            x = np.array(x)
            # Scale
            x_scaled = (x - means) / (stds + 1e-8)
            
            # Predict (Logistic Regression)
            z = np.dot(x_scaled, weights) + bias
            prob = 1.0 / (1.0 + math.exp(-z))
            
            # Combine or replace confidence
            # For now, let's just replace it
            cand["confidence"] = float(prob)
            
        return candidates
    except Exception as e:
        print(f"Warning: Failed to apply reweighter: {e}")
        return candidates


def detect_doors(
    primitives: Dict[str, Any],
    meta: Dict[str, Any],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Detect doors in a floor plan using vector primitives and rule-based logic."""
    
    if meta["mode"] == "scan" and config["mode_policy"].get("scan") == "empty_with_message":
        # For scans in v1, we skip detection
        return []

    lines = primitives.get("lines", [])
    beziers = primitives.get("beziers", [])
    
    # 1. Build spatial index for lines
    line_index = SpatialIndex(cell_size=200.0)
    for i, line in enumerate(lines):
        bbox = [
            min(line["p0"]["x"], line["p1"]["x"]),
            min(line["p0"]["y"], line["p1"]["y"]),
            max(line["p0"]["x"], line["p1"]["x"]),
            max(line["p0"]["y"], line["p1"]["y"]),
        ]
        line_index.add(i, bbox)

    candidates = []

    # 2. Swing door detection
    if config["swing"]["enabled"]:
        swing_conf = config["swing"]
        for b_idx, bez in enumerate(beziers):
            # Sample and fit arc
            pts = sample_bezier(
                bez["p0"], bez["p1"], bez["p2"], bez["p3"], 
                num_points=swing_conf["bezier_sampling_points"]
            )
            center, radius, rmse = fit_circle(pts)
            
            # Check arc thresholds
            arc_conf = swing_conf["arc"]
            if not (arc_conf["min_radius_px"] <= radius <= arc_conf["max_radius_px"]):
                continue
            if rmse > arc_conf["max_circle_fit_rmse"]:
                continue
                
            angle_span = get_arc_angle_span(pts, center)
            if not (arc_conf["min_angle_deg"] <= angle_span <= arc_conf["max_angle_deg"]):
                continue

            # Look for matching leaf lines
            leaf_conf = swing_conf["leaf"]
            # Query area: arc bbox padded
            arc_bbox = get_bbox(pts)
            query_bbox = [
                arc_bbox[0] - radius * 0.5, arc_bbox[1] - radius * 0.5,
                arc_bbox[2] + radius * 0.5, arc_bbox[3] + radius * 0.5
            ]
            
            nearby_line_indices = line_index.query(query_bbox)
            for l_idx in nearby_line_indices:
                line = lines[l_idx]
                p0 = (line["p0"]["x"], line["p0"]["y"])
                p1 = (line["p1"]["x"], line["p1"]["y"])
                l_len = dist_point_to_point(p0, p1)
                
                # Length ratio
                len_ratio = l_len / radius
                if not (leaf_conf["min_length_ratio"] <= len_ratio <= leaf_conf["max_length_ratio"]):
                    continue
                
                # Hinge proximity: one line endpoint near one arc endpoint or center
                arc_start = pts[0]
                arc_end = pts[-1]
                
                d0_start = dist_point_to_point(p0, arc_start)
                d0_end = dist_point_to_point(p0, arc_end)
                d1_start = dist_point_to_point(p1, arc_start)
                d1_end = dist_point_to_point(p1, arc_end)
                
                min_hinge_dist = min(d0_start, d0_end, d1_start, d1_end)
                if min_hinge_dist > radius * leaf_conf["max_hinge_dist_ratio"]:
                    # Also check center if it's a wedge style
                    d0_center = dist_point_to_point(p0, center)
                    d1_center = dist_point_to_point(p1, center)
                    if min(d0_center, d1_center) > radius * leaf_conf["max_hinge_dist_ratio"]:
                        continue

                # Basic score
                score_conf = swing_conf["scoring"]
                fit_score = max(0, 1.0 - (rmse / arc_conf["max_circle_fit_rmse"]))
                angle_score = max(0, 1.0 - abs(angle_span - 90.0) / 45.0)
                prox_score = max(0, 1.0 - (min_hinge_dist / (radius * leaf_conf["max_hinge_dist_ratio"])))
                
                confidence = (
                    score_conf["w_fit"] * fit_score +
                    score_conf["w_angle"] * angle_score +
                    score_conf["w_proximity"] * prox_score
                )
                
                if confidence >= config["output"]["min_confidence"]:
                    # Create a stable ID based on the primitives used
                    stable_key = f"swing|b={b_idx}|l={l_idx}"
                    door_id = "d_" + hashlib.sha1(stable_key.encode()).hexdigest()[:10]
                    
                    candidates.append({
                        "id": door_id,
                        "type": "swing",
                        "bbox_xyxy": get_bbox(pts + [p0, p1]),
                        "confidence": float(confidence),
                        "features": {
                            "rmse": float(rmse),
                            "radius": float(radius),
                            "angle_span": float(angle_span),
                            "hinge_dist": float(min_hinge_dist),
                            "len_ratio": float(len_ratio),
                        },
                        "primitives": {"beziers": [b_idx], "lines": [l_idx]}
                    })

    # 2.5 Apply reweighter if available
    if "reweighter_path" in config and Path(config["reweighter_path"]).exists():
        candidates = apply_reweighter(candidates, config["reweighter_path"])

    # 3. Simple NMS
    if not candidates:
        return []
        
    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    keep = []
    nms_iou = config["output"]["nms_iou"]
    
    for cand in candidates:
        overlap = False
        for kept in keep:
            if compute_iou(cand["bbox_xyxy"], kept["bbox_xyxy"]) > nms_iou:
                overlap = True
                break
        if not overlap:
            keep.append(cand)
            
    return keep[:config["output"]["max_doors"]]

