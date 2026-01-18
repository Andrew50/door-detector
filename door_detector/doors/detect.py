"""Core door detection logic using vector primitives."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from door_detector.doors.geometry import (
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
            x = []
            for feat_name in feature_order:
                val = cand["features"].get(feat_name, 0.0)
                x.append(val)

            x = np.array(x)
            x_scaled = (x - means) / (stds + 1e-8)

            z = np.dot(x_scaled, weights) + bias
            prob = 1.0 / (1.0 + math.exp(-z))
            cand["confidence"] = float(prob)

        return candidates
    except Exception as e:
        print(f"Warning: Failed to apply reweighter: {e}")
        return candidates


def detect_doors(primitives: Dict[str, Any], meta: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Detect doors in a floor plan using vector primitives and rule-based logic."""

    if meta["mode"] == "scan" and config["mode_policy"].get("scan") == "empty_with_message":
        return {"doors": [], "candidates": []}

    lines = primitives.get("lines", [])
    beziers = primitives.get("beziers", [])

    # Build spatial index for lines
    line_index = SpatialIndex(cell_size=200.0)
    for i, line in enumerate(lines):
        bbox = [
            min(line["p0"]["x"], line["p1"]["x"]),
            min(line["p0"]["y"], line["p1"]["y"]),
            max(line["p0"]["x"], line["p1"]["x"]),
            max(line["p0"]["y"], line["p1"]["y"]),
        ]
        line_index.add(i, bbox)

    # Two pools:
    # - `doors`: strict, post-threshold/post-NMS list (what the model “predicts”)
    # - `candidates`: broader pool for snapping/training (looser leaf len_ratio only)
    strict_candidates: List[Dict[str, Any]] = []
    candidate_pool: List[Dict[str, Any]] = []
    arc_cluster_counts: Dict[Tuple[int, int, int], int] = {}
    arc_cluster_sum_angle: Dict[Tuple[int, int, int], float] = {}

    if config["swing"]["enabled"]:
        swing_conf = config["swing"]
        out_conf = config.get("output", {})
        min_conf = float(out_conf.get("min_confidence", 0.55) or 0.55)
        # Looser leaf ratio for candidate pool (does NOT affect final doors).
        pool_min_len_ratio = 0.22
        pool_max_len_ratio = 2.20
        pool_max_hinge_dist_ratio = 0.55
        pool_require_endpoint_near_center = False
        pool_max_center_dist_ratio = 0.60
        pool_max_radial_angle_deg = 50.0
        pool_max_tip_to_arc_ratio = 0.70
        for b_idx, bez in enumerate(beziers):
            pts = sample_bezier(
                bez["p0"], bez["p1"], bez["p2"], bez["p3"], num_points=swing_conf["bezier_sampling_points"]
            )
            center, radius, rmse = fit_circle(pts)

            arc_conf = swing_conf["arc"]
            if not (arc_conf["min_radius_px"] <= radius <= arc_conf["max_radius_px"]):
                continue
            if rmse > arc_conf["max_circle_fit_rmse"]:
                continue

            angle_span = get_arc_angle_span(pts, center)
            if not (arc_conf["min_angle_deg"] <= angle_span <= arc_conf["max_angle_deg"]):
                continue

            circle_key = None
            if arc_conf.get("suppress_circle_clusters", False):
                cbin = float(arc_conf.get("circle_cluster_center_bin_px", 4.0))
                rbin = float(arc_conf.get("circle_cluster_radius_bin_px", 4.0))
                cbin = cbin if cbin > 0 else 4.0
                rbin = rbin if rbin > 0 else 4.0
                circle_key = (
                    int(round(center[0] / cbin)),
                    int(round(center[1] / cbin)),
                    int(round(radius / rbin)),
                )
                arc_cluster_counts[circle_key] = arc_cluster_counts.get(circle_key, 0) + 1
                arc_cluster_sum_angle[circle_key] = arc_cluster_sum_angle.get(circle_key, 0.0) + float(angle_span)

            leaf_conf = swing_conf["leaf"]
            arc_bbox = get_bbox(pts)
            query_bbox = [
                arc_bbox[0] - radius * 0.5,
                arc_bbox[1] - radius * 0.5,
                arc_bbox[2] + radius * 0.5,
                arc_bbox[3] + radius * 0.5,
            ]

            nearby_line_indices = line_index.query(query_bbox)
            for l_idx in nearby_line_indices:
                line = lines[l_idx]
                p0 = (line["p0"]["x"], line["p0"]["y"])
                p1 = (line["p1"]["x"], line["p1"]["y"])
                l_len = dist_point_to_point(p0, p1)

                len_ratio = l_len / radius if radius > 1e-6 else 0.0
                in_strict_len_ratio = bool(leaf_conf["min_length_ratio"] <= len_ratio <= leaf_conf["max_length_ratio"])
                in_pool_len_ratio = bool(pool_min_len_ratio <= len_ratio <= pool_max_len_ratio)
                if not in_pool_len_ratio:
                    continue

                arc_start = pts[0]
                arc_end = pts[-1]

                d0_start = dist_point_to_point(p0, arc_start)
                d0_end = dist_point_to_point(p0, arc_end)
                d1_start = dist_point_to_point(p1, arc_start)
                d1_end = dist_point_to_point(p1, arc_end)

                d0_center = dist_point_to_point(p0, center)
                d1_center = dist_point_to_point(p1, center)

                min_hinge_dist = min(d0_start, d0_end, d1_start, d1_end)
                strict_hinge_ok = True
                if min_hinge_dist > radius * leaf_conf["max_hinge_dist_ratio"]:
                    if min(d0_center, d1_center) > radius * leaf_conf["max_hinge_dist_ratio"]:
                        strict_hinge_ok = False

                pool_hinge_ok = True
                if min_hinge_dist > radius * pool_max_hinge_dist_ratio:
                    if min(d0_center, d1_center) > radius * pool_max_hinge_dist_ratio:
                        pool_hinge_ok = False
                if not pool_hinge_ok:
                    continue

                strict_center_ok = True
                if leaf_conf.get("require_endpoint_near_center", False):
                    max_center_ratio = float(
                        leaf_conf.get("max_center_dist_ratio", leaf_conf.get("max_hinge_dist_ratio", 0.25))
                    )
                    if min(d0_center, d1_center) > radius * max_center_ratio:
                        strict_center_ok = False

                pool_center_ok = True
                if pool_require_endpoint_near_center:
                    if min(d0_center, d1_center) > radius * pool_max_center_dist_ratio:
                        pool_center_ok = False
                if not pool_center_ok:
                    continue

                max_radial_angle = leaf_conf.get("max_radial_angle_deg", None)
                if max_radial_angle is not None:
                    try:
                        max_radial_angle = float(max_radial_angle)
                    except Exception:
                        max_radial_angle = None
                radial_angle_deg = None
                tip_to_arc_dist = None
                pool_radial_ok = True
                strict_radial_ok = True
                strict_tip_ok = True
                pool_tip_ok = True

                if (max_radial_angle is not None and max_radial_angle > 0) or (
                    pool_max_radial_angle_deg and pool_max_radial_angle_deg > 0
                ):
                    hinge_pt = p0 if d0_center <= d1_center else p1
                    tip_pt = p1 if hinge_pt == p0 else p0
                    if dist_point_to_point(tip_pt, arc_start) <= dist_point_to_point(tip_pt, arc_end):
                        target_pt = arc_start
                    else:
                        target_pt = arc_end
                    tip_to_arc_dist = dist_point_to_point(tip_pt, target_pt)

                    lx, ly = (tip_pt[0] - hinge_pt[0], tip_pt[1] - hinge_pt[1])
                    rx, ry = (target_pt[0] - center[0], target_pt[1] - center[1])
                    ln = math.hypot(lx, ly)
                    rn = math.hypot(rx, ry)
                    if ln > 1e-6 and rn > 1e-6:
                        dot = (lx * rx + ly * ry) / (ln * rn)
                        dot = max(-1.0, min(1.0, dot))
                        radial_angle_deg = math.degrees(math.acos(dot))
                        if max_radial_angle is not None and max_radial_angle > 0 and radial_angle_deg > max_radial_angle:
                            strict_radial_ok = False
                        if pool_max_radial_angle_deg and pool_max_radial_angle_deg > 0 and radial_angle_deg > pool_max_radial_angle_deg:
                            pool_radial_ok = False

                    max_tip_ratio = leaf_conf.get("max_tip_to_arc_ratio", None)
                    if max_tip_ratio is not None and tip_to_arc_dist is not None:
                        try:
                            max_tip_ratio = float(max_tip_ratio)
                        except Exception:
                            max_tip_ratio = None
                        if max_tip_ratio is not None and max_tip_ratio > 0:
                            if tip_to_arc_dist > radius * max_tip_ratio:
                                strict_tip_ok = False
                    if pool_max_tip_to_arc_ratio is not None and tip_to_arc_dist is not None:
                        if pool_max_tip_to_arc_ratio > 0 and tip_to_arc_dist > radius * pool_max_tip_to_arc_ratio:
                            pool_tip_ok = False

                if not pool_radial_ok or not pool_tip_ok:
                    continue

                score_conf = swing_conf["scoring"]
                fit_score = max(0, 1.0 - (rmse / arc_conf["max_circle_fit_rmse"]))
                min_a = float(arc_conf.get("min_angle_deg", 55.0))
                denom = max(1.0, 90.0 - min_a)
                angle_score = (float(angle_span) - min_a) / denom
                angle_score = max(0.0, min(1.0, angle_score))
                prox_score_strict = max(0, 1.0 - (min_hinge_dist / (radius * leaf_conf["max_hinge_dist_ratio"])))
                prox_score_pool = max(0, 1.0 - (min_hinge_dist / (radius * pool_max_hinge_dist_ratio)))

                conf_strict = score_conf["w_fit"] * fit_score + score_conf["w_angle"] * angle_score + score_conf["w_proximity"] * prox_score_strict
                conf_pool = score_conf["w_fit"] * fit_score + score_conf["w_angle"] * angle_score + score_conf["w_proximity"] * prox_score_pool

                stable_key = f"swing|b={b_idx}|l={l_idx}"
                door_id = "d_" + hashlib.sha1(stable_key.encode()).hexdigest()[:10]

                base = {
                    "id": door_id,
                    "type": "swing",
                    "bbox_xyxy": get_bbox(pts + [p0, p1]),
                    "features": {
                        "rmse": float(rmse),
                        "radius": float(radius),
                        "angle_span": float(angle_span),
                        "hinge_dist": float(min_hinge_dist),
                        "len_ratio": float(len_ratio),
                        "center_dist": float(min(d0_center, d1_center)),
                        "radial_angle_deg": float(radial_angle_deg) if radial_angle_deg is not None else 0.0,
                        "tip_to_arc_dist": float(tip_to_arc_dist) if tip_to_arc_dist is not None else 0.0,
                    },
                    "primitives": {"beziers": [b_idx], "lines": [l_idx]},
                    "_circle_key": circle_key,
                }

                cand_pool = dict(base)
                cand_pool["confidence"] = float(conf_pool)
                cand_pool["pool"] = True
                candidate_pool.append(cand_pool)

                strict_ok = (
                    in_strict_len_ratio
                    and strict_hinge_ok
                    and strict_center_ok
                    and strict_radial_ok
                    and strict_tip_ok
                )
                if strict_ok and float(conf_strict) >= min_conf:
                    cand_strict = dict(base)
                    cand_strict["confidence"] = float(conf_strict)
                    strict_candidates.append(cand_strict)

    if strict_candidates and config.get("swing", {}).get("enabled") and config.get("swing", {}).get("arc", {}).get("suppress_circle_clusters", False):
        arc_conf = config["swing"]["arc"]
        min_arcs = int(arc_conf.get("circle_cluster_min_arcs", 3))
        min_total_angle = float(arc_conf.get("circle_cluster_min_total_angle_deg", 250.0))
        filtered: List[Dict[str, Any]] = []
        for cand in strict_candidates:
            key = cand.get("_circle_key", None)
            if key is not None:
                if arc_cluster_counts.get(key, 0) >= min_arcs and arc_cluster_sum_angle.get(key, 0.0) >= min_total_angle:
                    continue
            filtered.append(cand)
        strict_candidates = filtered

    if candidate_pool and config.get("swing", {}).get("enabled") and config.get("swing", {}).get("arc", {}).get("suppress_circle_clusters", False):
        arc_conf = config["swing"]["arc"]
        min_arcs = int(arc_conf.get("circle_cluster_min_arcs", 3))
        min_total_angle = float(arc_conf.get("circle_cluster_min_total_angle_deg", 250.0))
        filtered_pool: List[Dict[str, Any]] = []
        for cand in candidate_pool:
            key = cand.get("_circle_key", None)
            if key is not None:
                if arc_cluster_counts.get(key, 0) >= min_arcs and arc_cluster_sum_angle.get(key, 0.0) >= min_total_angle:
                    continue
            filtered_pool.append(cand)
        candidate_pool = filtered_pool

    for cand in strict_candidates:
        cand.pop("_circle_key", None)
    for cand in candidate_pool:
        cand.pop("_circle_key", None)

    if "reweighter_path" in config and Path(config["reweighter_path"]).exists():
        strict_candidates = apply_reweighter(strict_candidates, config["reweighter_path"])
        candidate_pool = apply_reweighter(candidate_pool, config["reweighter_path"])

    if not strict_candidates:
        candidate_pool.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return {"doors": [], "candidates": candidate_pool[:5000]}

    strict_candidates.sort(key=lambda x: x["confidence"], reverse=True)

    keep = []
    nms_iou = config["output"]["nms_iou"]
    for cand in strict_candidates:
        overlap = False
        for kept in keep:
            if compute_iou(cand["bbox_xyxy"], kept["bbox_xyxy"]) > nms_iou:
                overlap = True
                break
        if not overlap:
            keep.append(cand)

    candidate_pool.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return {"doors": keep[: config["output"]["max_doors"]], "candidates": candidate_pool[:5000]}

