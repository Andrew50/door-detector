"""Core door detection logic using vector primitives."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _q(v: float, *, step: float) -> int:
    """Quantize a float deterministically to an integer bin."""
    if step <= 0:
        step = 1.0
    try:
        return int(round(float(v) / float(step)))
    except Exception:
        return 0


def _stable_swing_candidate_id(
    *,
    center: Tuple[float, float],
    radius: float,
    angle_span: float,
    arc_start: Tuple[float, float],
    arc_end: Tuple[float, float],
    hinge_pt: Tuple[float, float],
    tip_pt: Tuple[float, float],
    bbox_xyxy: List[float],
    quant_step_px: float = 1.0,
) -> str:
    """Return a stable candidate id derived from quantized geometry (not indices)."""
    # Canonicalize arc endpoints so reversing curve direction doesn't change the id.
    a0 = (_q(arc_start[0], step=quant_step_px), _q(arc_start[1], step=quant_step_px))
    a1 = (_q(arc_end[0], step=quant_step_px), _q(arc_end[1], step=quant_step_px))
    if a1 < a0:
        a0, a1 = a1, a0

    payload = {
        "id_version": "swing_geom_v1",
        "type": "swing",
        "center": (_q(center[0], step=quant_step_px), _q(center[1], step=quant_step_px)),
        "radius": _q(radius, step=quant_step_px),
        "angle_span": _q(angle_span, step=1.0),
        "arc_endpoints": (a0, a1),
        "hinge": (_q(hinge_pt[0], step=quant_step_px), _q(hinge_pt[1], step=quant_step_px)),
        "tip": (_q(tip_pt[0], step=quant_step_px), _q(tip_pt[1], step=quant_step_px)),
        "bbox": tuple(_q(float(v), step=quant_step_px) for v in (bbox_xyxy or [0, 0, 0, 0])),
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "d_" + hashlib.sha1(stable).hexdigest()[:12]


def _stable_swing_arc_candidate_id(
    *,
    center: Tuple[float, float],
    radius: float,
    angle_span: float,
    arc_start: Tuple[float, float],
    arc_end: Tuple[float, float],
    bbox_xyxy: List[float],
    quant_step_px: float = 1.0,
) -> str:
    """Stable id for arc-only swing candidates (no leaf/hinge line)."""
    a0 = (_q(arc_start[0], step=quant_step_px), _q(arc_start[1], step=quant_step_px))
    a1 = (_q(arc_end[0], step=quant_step_px), _q(arc_end[1], step=quant_step_px))
    if a1 < a0:
        a0, a1 = a1, a0
    payload = {
        "id_version": "swing_arc_geom_v1",
        "type": "swing_arc",
        "center": (_q(center[0], step=quant_step_px), _q(center[1], step=quant_step_px)),
        "radius": _q(radius, step=quant_step_px),
        "angle_span": _q(angle_span, step=1.0),
        "arc_endpoints": (a0, a1),
        "bbox": tuple(_q(float(v), step=quant_step_px) for v in (bbox_xyxy or [0, 0, 0, 0])),
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "d_" + hashlib.sha1(stable).hexdigest()[:12]


def apply_reweighter(candidates: List[Dict[str, Any]], model_path: str) -> List[Dict[str, Any]]:
    """Apply a learned reweighter to update candidate confidence scores."""
    try:
        with open(model_path) as f:
            model = json.load(f)

        weights = np.array(model["weights"], dtype=float)
        bias = float(model["bias"])
        feature_order = list(model["feature_order"])
        scaler = dict(model["scaler"])

        means = np.array(scaler["mean"], dtype=float)
        stds = np.array(scaler["std"], dtype=float)

        if weights.ndim != 1:
            raise ValueError("invalid model weights shape")
        if len(feature_order) != int(weights.shape[0]):
            raise ValueError("feature_order length does not match weights")
        if means.shape != weights.shape or stds.shape != weights.shape:
            raise ValueError("scaler mean/std shapes do not match weights")

        # Guard against pathological scalers (e.g. tiny std from near-constant features),
        # which can explode z-scores and collapse probabilities to ~0 or ~1.
        # This is especially important for very small training sets.
        try:
            std_floor = 1e-3
            stds_safe = np.where(stds < std_floor, std_floor, stds)
        except Exception:
            stds_safe = stds

        for cand in candidates:
            x = []
            for feat_name in feature_order:
                val = (cand.get("features") or {}).get(feat_name, 0.0)
                x.append(val)

            x = np.array(x, dtype=float)
            x_scaled = (x - means) / (stds_safe + 1e-8)
            try:
                x_scaled = np.clip(x_scaled, -10.0, 10.0)
            except Exception:
                pass

            z = np.dot(x_scaled, weights) + bias
            # Numerically-stable sigmoid (avoid overflow for large negative z).
            zf = float(z)
            if zf >= 0:
                ez = math.exp(-zf)  # safe: exp(-positive) in (0, 1]
                prob = 1.0 / (1.0 + ez)
            else:
                ez = math.exp(zf)  # safe: exp(negative) in (0, 1]
                prob = ez / (1.0 + ez)
            cand["confidence"] = float(prob)

        return candidates
    except Exception as e:
        print(f"Warning: Failed to apply reweighter: {e}")
        return candidates


def apply_reweighters_by_type(
    candidates: List[Dict[str, Any]],
    *,
    model_paths_by_type: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Apply per-type reweighters in-place (best-effort)."""
    if not candidates:
        return candidates
    if not isinstance(model_paths_by_type, dict) or not model_paths_by_type:
        return candidates

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        t = str(c.get("type") or "").strip().lower()
        if not t:
            continue
        buckets.setdefault(t, []).append(c)

    for t, path in model_paths_by_type.items():
        if not path or not isinstance(path, str):
            continue
        try:
            if not Path(path).exists():
                continue
        except Exception:
            continue
        if t not in buckets:
            continue
        apply_reweighter(buckets[t], path)

    return candidates


def _normalize_bbox_xyxy(bbox: Any) -> Optional[List[float]]:
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _bbox_union(a: List[float], b: List[float]) -> List[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _stable_double_candidate_id(*, swing_ids: Tuple[str, str], bbox_xyxy: List[float], quant_step_px: float = 1.0) -> str:
    a, b = sorted([str(swing_ids[0]), str(swing_ids[1])])
    nb = _normalize_bbox_xyxy(bbox_xyxy) or [0.0, 0.0, 0.0, 0.0]
    payload = {
        "id_version": "double_pair_v1",
        "type": "double",
        "components": [a, b],
        "bbox": tuple(_q(float(v), step=quant_step_px) for v in nb),
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "d_" + hashlib.sha1(stable).hexdigest()[:12]


def _stable_line_candidate_id(
    *, cand_type: str, p0: Tuple[float, float], p1: Tuple[float, float], bbox_xyxy: List[float], quant_step_px: float = 1.0
) -> str:
    a0 = (_q(p0[0], step=quant_step_px), _q(p0[1], step=quant_step_px))
    a1 = (_q(p1[0], step=quant_step_px), _q(p1[1], step=quant_step_px))
    if a1 < a0:
        a0, a1 = a1, a0
    nb = _normalize_bbox_xyxy(bbox_xyxy) or [0.0, 0.0, 0.0, 0.0]
    payload = {
        "id_version": "line_geom_v1",
        "type": str(cand_type),
        "p": (a0, a1),
        "bbox": tuple(_q(float(v), step=quant_step_px) for v in nb),
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "d_" + hashlib.sha1(stable).hexdigest()[:12]


def _stable_bifold_candidate_id(*, line_bins: List[Tuple[int, int, int, int]], bbox_xyxy: List[float]) -> str:
    # line_bins contains quantized endpoints for each segment; sort for stability.
    nb = _normalize_bbox_xyxy(bbox_xyxy) or [0.0, 0.0, 0.0, 0.0]
    payload = {
        "id_version": "bifold_chain_v1",
        "type": "bifold",
        "lines": sorted(list(line_bins)),
        "bbox": tuple(_q(float(v), step=1.0) for v in nb),
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "d_" + hashlib.sha1(stable).hexdigest()[:12]


def _is_dashed_primitive(p: Dict[str, Any]) -> bool:
    v = p.get("is_dashed")
    if isinstance(v, bool):
        return v
    # Fallback: infer from dash_pattern if present.
    dp = p.get("dash_pattern")
    if isinstance(dp, (list, tuple)) and len(dp) >= 2:
        try:
            return any(float(x) > 0 for x in dp if x is not None)
        except Exception:
            return False
    return False


def _line_len_px(line: Dict[str, Any]) -> float:
    p0 = (line["p0"]["x"], line["p0"]["y"])
    p1 = (line["p1"]["x"], line["p1"]["y"])
    return float(dist_point_to_point(p0, p1))


def _extract_polyline_arcs_from_lines(
    *,
    lines: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract arc-like polylines from short line segments.

    Many PDFs approximate circular arcs (door swings) as chains of short straight
    segments rather than cubic beziers. This helper groups short segments by
    snapped endpoints, fits a circle, and returns ordered point chains.
    """
    swing_conf = (config.get("swing") or {}) if isinstance(config, dict) else {}
    arc_conf = (swing_conf.get("arc") or {}) if isinstance(swing_conf, dict) else {}
    poly_conf = (swing_conf.get("polyline_arc") or {}) if isinstance(swing_conf, dict) else {}
    if poly_conf.get("enabled") is False:
        return []

    # Defaults chosen to be conservative and roughly match bezier arc thresholds.
    endpoint_snap_px = float(poly_conf.get("endpoint_snap_px", 3.0) or 3.0)
    min_segments = int(poly_conf.get("min_segments", 4) or 4)
    max_segments = int(poly_conf.get("max_segments", 36) or 36)
    max_seg_len = float(poly_conf.get("max_segment_length_px", 85.0) or 85.0)
    allow_branches = bool(poly_conf.get("allow_branches", True))
    max_paths_per_component = int(poly_conf.get("max_paths_per_component", 200) or 200)
    allow_branches = bool(poly_conf.get("allow_branches", True))
    max_paths_per_component = int(poly_conf.get("max_paths_per_component", 200) or 200)
    # Fit tolerance: reuse arc's circle fit constraint if present, else a mild default.
    max_circle_fit_rmse = float(arc_conf.get("max_circle_fit_rmse", 2.5) or 2.5)

    # Use the same arc thresholds as bezier arcs (in pixel space).
    min_radius_px = float(arc_conf.get("min_radius_px", 0.0) or 0.0)
    max_radius_px = float(arc_conf.get("max_radius_px", 1e9) or 1e9)
    min_angle_deg = float(arc_conf.get("min_angle_deg", 0.0) or 0.0)
    max_angle_deg = float(arc_conf.get("max_angle_deg", 1e9) or 1e9)

    if not (endpoint_snap_px > 0):
        endpoint_snap_px = 3.0
    if max_seg_len <= 0:
        max_seg_len = 85.0

    def _bin_pt(p: Tuple[float, float]) -> Tuple[int, int]:
        return (_q(p[0], step=endpoint_snap_px), _q(p[1], step=endpoint_snap_px))

    # Only consider short, solid segments as arc pieces.
    usable: List[int] = []
    ends: Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], Tuple[int, int]]] = {}
    end_to_lines: Dict[Tuple[int, int], List[int]] = {}
    for i, ln in enumerate(lines):
        if _is_dashed_primitive(ln):
            continue
        try:
            p0 = (float(ln["p0"]["x"]), float(ln["p0"]["y"]))
            p1 = (float(ln["p1"]["x"]), float(ln["p1"]["y"]))
        except Exception:
            continue
        seg_len = float(dist_point_to_point(p0, p1))
        if not (0.5 <= seg_len <= max_seg_len):
            continue
        b0 = _bin_pt(p0)
        b1 = _bin_pt(p1)
        ends[i] = (p0, p1, b0, b1)
        end_to_lines.setdefault(b0, []).append(i)
        end_to_lines.setdefault(b1, []).append(i)
        usable.append(i)

    if not usable:
        return []

    # Build adjacency via shared binned endpoints.
    adj: Dict[int, set[int]] = {i: set() for i in usable}
    for node, inc in end_to_lines.items():
        if len(inc) <= 1:
            continue
        # Link all incident short segments together.
        for a in inc:
            for b in inc:
                if a != b:
                    adj[a].add(b)

    # Connected components (over usable short segments only).
    seen: set[int] = set()
    comps: List[List[int]] = []
    for i in usable:
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)

    out: List[Dict[str, Any]] = []
    for comp in comps:
        if not (min_segments <= len(comp) <= max_segments):
            continue

        # Node degrees within this component.
        node_deg: Dict[Tuple[int, int], int] = {}
        for li in comp:
            _, _, b0, b1 = ends[li]
            node_deg[b0] = node_deg.get(b0, 0) + 1
            node_deg[b1] = node_deg.get(b1, 0) + 1

        end_nodes = [n for n, d in node_deg.items() if d == 1]
        if len(end_nodes) != 2:
            continue  # ignore loops/branches

        # Walk the chain in order from one endpoint.
        start_node = end_nodes[0]
        ordered: List[int] = []
        used_lines: set[int] = set()
        current_node = start_node
        prev_line: Optional[int] = None

        for _ in range(len(comp)):
            candidates = [li for li in end_to_lines.get(current_node, []) if li in comp and li not in used_lines]
            if not candidates:
                break
            nxt = candidates[0]
            if prev_line is not None and len(candidates) > 1:
                # Prefer not to immediately backtrack.
                for cand in candidates:
                    if cand != prev_line:
                        nxt = cand
                        break
            ordered.append(nxt)
            used_lines.add(nxt)
            p0, p1, b0, b1 = ends[nxt]
            current_node = b1 if current_node == b0 else b0
            prev_line = nxt

        if len(ordered) != len(comp):
            continue

        # Build ordered point chain (dedupe consecutive identical points).
        pts: List[Tuple[float, float]] = []
        current_node = start_node
        for li in ordered:
            p0, p1, b0, b1 = ends[li]
            if current_node == b0:
                a, b = p0, p1
                current_node = b1
            else:
                a, b = p1, p0
                current_node = b0
            if not pts:
                pts.append(a)
            if pts[-1] != b:
                pts.append(b)

        if len(pts) < 3:
            continue

        center, radius, rmse = fit_circle(pts)
        if not (min_radius_px <= radius <= max_radius_px):
            continue
        if rmse > max_circle_fit_rmse:
            continue

        angle_span = get_arc_angle_span(pts, center)
        if not (min_angle_deg <= angle_span <= max_angle_deg):
            continue

        out.append(
            {
                "source": "polyline",
                "pts": pts,
                "center": center,
                "radius": float(radius),
                "rmse": float(rmse),
                "angle_span": float(angle_span),
                "arc_lines": ordered,
            }
        )

    return out


def _debug_polyline_arcs_from_lines_subset(
    *,
    lines: List[Dict[str, Any]],
    line_indices: List[int],
    config: Dict[str, Any],
    max_rejected_examples: int = 25,
) -> Dict[str, Any]:
    """Debug-only polyline-arc extraction within a subset of line indices.

    Returns a dict with:
    - usable_short_segments: int
    - component_sizes: List[int]
    - rejected_components: List[Dict[str, Any]] (sampled)
    - arc_candidates: List[Dict[str, Any]] where each item includes pts/fit metrics and fails list

    This mirrors `_extract_polyline_arcs_from_lines` but keeps *global* line indices and
    records why components/arcs were rejected, so the unmatched debug report can be
    fully explanatory.
    """
    swing_conf = (config.get("swing") or {}) if isinstance(config, dict) else {}
    arc_conf = (swing_conf.get("arc") or {}) if isinstance(swing_conf, dict) else {}
    poly_conf = (swing_conf.get("polyline_arc") or {}) if isinstance(swing_conf, dict) else {}
    if poly_conf.get("enabled") is False:
        return {
            "enabled": False,
            "usable_short_segments": 0,
            "component_sizes": [],
            "rejected_components": [],
            "arc_candidates": [],
        }

    endpoint_snap_px = float(poly_conf.get("endpoint_snap_px", 3.0) or 3.0)
    min_segments = int(poly_conf.get("min_segments", 4) or 4)
    max_segments = int(poly_conf.get("max_segments", 36) or 36)
    max_seg_len = float(poly_conf.get("max_segment_length_px", 85.0) or 85.0)
    allow_branches = bool(poly_conf.get("allow_branches", True))
    max_paths_per_component = int(poly_conf.get("max_paths_per_component", 200) or 200)

    # Fit/arc thresholds (same as swing.arc).
    max_circle_fit_rmse = float(arc_conf.get("max_circle_fit_rmse", 2.5) or 2.5)
    min_radius_px = float(arc_conf.get("min_radius_px", 0.0) or 0.0)
    max_radius_px = float(arc_conf.get("max_radius_px", 1e9) or 1e9)
    min_angle_deg = float(arc_conf.get("min_angle_deg", 0.0) or 0.0)
    max_angle_deg = float(arc_conf.get("max_angle_deg", 1e9) or 1e9)

    if not (endpoint_snap_px > 0):
        endpoint_snap_px = 3.0
    if max_seg_len <= 0:
        max_seg_len = 85.0

    def _bin_pt(p: Tuple[float, float]) -> Tuple[int, int]:
        return (_q(p[0], step=endpoint_snap_px), _q(p[1], step=endpoint_snap_px))

    usable: List[int] = []
    ends: Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], Tuple[int, int]]] = {}
    end_to_lines: Dict[Tuple[int, int], List[int]] = {}
    for i in list(line_indices or []):
        try:
            ln = lines[int(i)]
        except Exception:
            continue
        if _is_dashed_primitive(ln):
            continue
        try:
            p0 = (float(ln["p0"]["x"]), float(ln["p0"]["y"]))
            p1 = (float(ln["p1"]["x"]), float(ln["p1"]["y"]))
        except Exception:
            continue
        seg_len = float(dist_point_to_point(p0, p1))
        if not (0.5 <= seg_len <= max_seg_len):
            continue
        b0 = _bin_pt(p0)
        b1 = _bin_pt(p1)
        ends[int(i)] = (p0, p1, b0, b1)
        end_to_lines.setdefault(b0, []).append(int(i))
        end_to_lines.setdefault(b1, []).append(int(i))
        usable.append(int(i))

    if not usable:
        return {
            "enabled": True,
            "usable_short_segments": 0,
            "component_sizes": [],
            "rejected_components": [],
            "arc_candidates": [],
        }

    # Build adjacency via shared binned endpoints.
    adj: Dict[int, set[int]] = {i: set() for i in usable}
    for node, inc in end_to_lines.items():
        if len(inc) <= 1:
            continue
        for a in inc:
            for b in inc:
                if a != b:
                    adj[a].add(b)

    # Connected components (over usable short segments only).
    seen: set[int] = set()
    comps: List[List[int]] = []
    for i in usable:
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)

    component_sizes = [int(len(c)) for c in comps]
    rejected_components: List[Dict[str, Any]] = []
    arc_candidates: List[Dict[str, Any]] = []

    def _reject(*, comp: List[int], reason: str, detail: Optional[Dict[str, Any]] = None) -> None:
        if len(rejected_components) >= int(max_rejected_examples):
            return
        out = {"reason": str(reason), "segments": int(len(comp))}
        if detail and isinstance(detail, dict):
            out.update(detail)
        # include a small sample for reproducibility
        out["sample_line_idxs"] = [int(x) for x in list(comp)[: min(8, len(comp))]]
        rejected_components.append(out)

    # Walk each component and attempt circle fit + arc validation, recording failures.
    for ci, comp in enumerate(comps):
        if not (min_segments <= len(comp) <= max_segments):
            _reject(
                comp=comp,
                reason="component.size",
                detail={"min_segments": int(min_segments), "max_segments": int(max_segments)},
            )
            continue

        # Node degrees within this component.
        node_deg: Dict[Tuple[int, int], int] = {}
        # Also store the component edges so we can explore paths in branched components.
        comp_edges: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
        for li in comp:
            try:
                _, _, b0, b1 = ends[li]
            except Exception:
                continue
            node_deg[b0] = node_deg.get(b0, 0) + 1
            node_deg[b1] = node_deg.get(b1, 0) + 1
            comp_edges.append((int(li), b0, b1))

        end_nodes = [n for n, d in node_deg.items() if d == 1]
        if len(end_nodes) != 2:
            # Many door arcs get "branched" where a wall/leaf line touches the arc polyline,
            # producing 3+ endpoints. In that case, try to recover a simple path that still
            # fits a plausible arc. Keep loops (end_nodes==0) rejected to avoid label bubbles.
            if not allow_branches or len(end_nodes) < 2:
                _reject(comp=comp, reason="component.topology", detail={"end_nodes": int(len(end_nodes))})
                continue  # ignore loops/branches unless enabled

            # Build node -> incident component lines map.
            node_to_lines: Dict[Tuple[int, int], List[int]] = {}
            for li, b0, b1 in comp_edges:
                node_to_lines.setdefault(b0, []).append(li)
                node_to_lines.setdefault(b1, []).append(li)

            # Helper: given (node, line) return the other node.
            other_node: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {li: (b0, b1) for li, b0, b1 in comp_edges}

            best_path: Optional[List[int]] = None
            best_start: Optional[Tuple[int, int]] = None
            best_score = -1e18
            paths_tried = 0

            def _score_path(path_lines: List[int], start_node: Tuple[int, int]) -> Optional[Tuple[float, List[Tuple[float, float]]]]:
                # Build ordered point chain for this path and compute circle fit.
                pts: List[Tuple[float, float]] = []
                cur_node = start_node
                for li in path_lines:
                    try:
                        p0, p1, b0, b1 = ends[li]
                    except Exception:
                        return None
                    if cur_node == b0:
                        a, b = p0, p1
                        cur_node = b1
                    else:
                        a, b = p1, p0
                        cur_node = b0
                    if not pts:
                        pts.append((float(a[0]), float(a[1])))
                    if pts[-1] != (float(b[0]), float(b[1])):
                        pts.append((float(b[0]), float(b[1])))
                if len(pts) < 3:
                    return None
                try:
                    center, radius, rmse = fit_circle(pts)
                    angle_span = float(get_arc_angle_span(pts, center))
                except Exception:
                    return None
                radius = float(radius)
                rmse = float(rmse)
                # Apply the same arc thresholds used for normal components.
                if not (min_radius_px <= radius <= max_radius_px):
                    return None
                if rmse > max_circle_fit_rmse:
                    return None
                if not (min_angle_deg <= angle_span <= max_angle_deg):
                    return None
                # Score: prefer larger angle spans and better fits; mild preference for longer paths.
                fit_score = max(0.0, 1.0 - (rmse / max(1e-6, max_circle_fit_rmse)))
                score = float(angle_span) * 1.0 + float(fit_score) * 50.0 + float(len(path_lines)) * 0.5
                return score, pts

            def _dfs(cur_node: Tuple[int, int], target: Tuple[int, int], used: set[int], path: List[int]) -> None:
                nonlocal best_path, best_start, best_score, paths_tried
                if paths_tried >= max_paths_per_component:
                    return
                if len(path) > max_segments:
                    return
                if cur_node == target:
                    paths_tried += 1
                    if len(path) < min_segments:
                        return
                    res = _score_path(path, start_node=path_start)
                    if res is None:
                        return
                    score, _pts = res
                    if score > best_score:
                        best_score = score
                        best_path = list(path)
                        best_start = path_start
                    return
                for li in node_to_lines.get(cur_node, []):
                    if li in used:
                        continue
                    used.add(li)
                    path.append(li)
                    b0, b1 = other_node[li]
                    nxt_node = b1 if cur_node == b0 else b0
                    _dfs(nxt_node, target, used, path)
                    path.pop()
                    used.remove(li)

            # Enumerate simple paths between endpoint pairs, keep the best arc-like one.
            for i in range(len(end_nodes)):
                for j in range(i + 1, len(end_nodes)):
                    path_start = end_nodes[i]
                    target = end_nodes[j]
                    _dfs(path_start, target, set(), [])
                    if paths_tried >= max_paths_per_component:
                        break
                if paths_tried >= max_paths_per_component:
                    break

            if best_path is None or best_start is None:
                _reject(
                    comp=comp,
                    reason="component.topology",
                    detail={"end_nodes": int(len(end_nodes)), "paths_tried": int(paths_tried)},
                )
                continue

            # Use the recovered simple path as the ordered chain.
            ordered = best_path
            start_node = best_start
        else:
            # Walk the chain in order from one endpoint.
            start_node = end_nodes[0]
            ordered = []
            used_lines: set[int] = set()
            current_node = start_node
            prev_line: Optional[int] = None

            for _ in range(len(comp)):
                candidates = [li for li in end_to_lines.get(current_node, []) if li in comp and li not in used_lines]
                if not candidates:
                    break
                nxt = candidates[0]
                if prev_line is not None and len(candidates) > 1:
                    for cand in candidates:
                        if cand != prev_line:
                            nxt = cand
                            break
                ordered.append(nxt)
                used_lines.add(nxt)
                p0, p1, b0, b1 = ends[nxt]
                current_node = b1 if current_node == b0 else b0
                prev_line = nxt

        # For non-branched components, ensure we walked everything.
        if len(end_nodes) == 2 and len(ordered) != len(comp):
            _reject(comp=comp, reason="component.walk_failed", detail={"walked": int(len(ordered)), "expected": int(len(comp))})
            continue

        # Build ordered point chain.
        pts: List[Tuple[float, float]] = []
        current_node = start_node
        for li in ordered:
            p0, p1, b0, b1 = ends[li]
            if current_node == b0:
                a, b = p0, p1
                current_node = b1
            else:
                a, b = p1, p0
                current_node = b0
            if not pts:
                pts.append(a)
            if pts[-1] != b:
                pts.append(b)

        if len(pts) < 3:
            _reject(comp=comp, reason="arc.too_few_points", detail={"points": int(len(pts))})
            continue

        fit = None
        try:
            center, radius, rmse = fit_circle(pts)
            angle_span = get_arc_angle_span(pts, center)
            fit = (center, float(radius), float(rmse), float(angle_span))
        except Exception:
            fit = None

        if fit is None:
            _reject(comp=comp, reason="arc.fit_failed")
            continue

        center, radius, rmse, angle_span = fit
        fails: List[str] = []
        if not (min_radius_px <= radius <= max_radius_px):
            fails.append("arc.radius")
        if rmse > max_circle_fit_rmse:
            fails.append("arc.rmse")
        if not (min_angle_deg <= angle_span <= max_angle_deg):
            fails.append("arc.angle_span")

        arc_candidates.append(
            {
                "source": "polyline",
                "comp_idx": int(ci),
                "line_idxs": [int(x) for x in ordered],
                "radius": float(radius),
                "rmse": float(rmse),
                "angle_span_deg": float(angle_span),
                "center_xy": [float(center[0]), float(center[1])],
                "fails": fails,
                "arc_conf": {
                    "min_radius_px": float(min_radius_px),
                    "max_radius_px": float(max_radius_px),
                    "max_rmse": float(max_circle_fit_rmse),
                    "min_angle_deg": float(min_angle_deg),
                    "max_angle_deg": float(max_angle_deg),
                },
            }
        )

    return {
        "enabled": True,
        "endpoint_snap_px": float(endpoint_snap_px),
        "min_segments": int(min_segments),
        "max_segments": int(max_segments),
        "max_segment_length_px": float(max_seg_len),
        "usable_short_segments": int(len(usable)),
        "component_sizes": [int(x) for x in component_sizes],
        "rejected_components": rejected_components,
        "arc_candidates": arc_candidates,
    }


def detect_swing_candidates(
    *,
    lines: List[Dict[str, Any]],
    beziers: List[Dict[str, Any]],
    line_index: SpatialIndex,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (strict_candidates, candidate_pool) for swing doors."""
    strict_candidates: List[Dict[str, Any]] = []
    candidate_pool: List[Dict[str, Any]] = []
    arc_cluster_counts: Dict[Tuple[int, int, int], int] = {}
    arc_cluster_sum_angle: Dict[Tuple[int, int, int], float] = {}

    swing_conf = config["swing"]
    # Looser leaf ratio for candidate pool (does NOT affect final doors).
    pool_min_len_ratio = 0.22
    pool_max_len_ratio = 2.20
    pool_max_hinge_dist_ratio = 0.55
    pool_require_endpoint_near_center = False
    pool_max_center_dist_ratio = 0.60
    pool_max_radial_angle_deg = 50.0
    pool_max_tip_to_arc_ratio = 0.70

    arc_conf = swing_conf["arc"]
    arc_items: List[Dict[str, Any]] = []

    # Bezier arcs (native PDF curves).
    for b_idx, bez in enumerate(beziers):
        try:
            pts = sample_bezier(
                bez["p0"],
                bez["p1"],
                bez["p2"],
                bez["p3"],
                num_points=swing_conf["bezier_sampling_points"],
            )
            center, radius, rmse = fit_circle(pts)
            angle_span = get_arc_angle_span(pts, center)
        except Exception:
            continue
        arc_items.append(
            {
                "source": "bezier",
                "b_idx": int(b_idx),
                "arc_lines": [],
                "pts": pts,
                "center": center,
                "radius": float(radius),
                "rmse": float(rmse),
                "angle_span": float(angle_span),
            }
        )

    # Polyline arcs (short line-segment chains approximating arcs).
    for a in _extract_polyline_arcs_from_lines(lines=lines, config=config):
        arc_items.append(a)

    for arc in arc_items:
        pts = arc.get("pts") or []
        if not pts:
            continue
        center = arc.get("center") or (0.0, 0.0)
        try:
            radius = float(arc.get("radius", 0.0) or 0.0)
            rmse = float(arc.get("rmse", 1e9) or 1e9)
            angle_span = float(arc.get("angle_span", 0.0) or 0.0)
        except Exception:
            continue

        if not (arc_conf["min_radius_px"] <= radius <= arc_conf["max_radius_px"]):
            continue
        if rmse > arc_conf["max_circle_fit_rmse"]:
            continue
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
        arc_line_set: set[int] = set()
        try:
            arc_line_set = set(int(i) for i in (arc.get("arc_lines") or []))
        except Exception:
            arc_line_set = set()

        had_any_pool_candidate_for_arc = False
        for l_idx in nearby_line_indices:
            if arc_line_set and int(l_idx) in arc_line_set:
                continue
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
                max_center_ratio = float(leaf_conf.get("max_center_dist_ratio", leaf_conf.get("max_hinge_dist_ratio", 0.25)))
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

            if (max_radial_angle is not None and max_radial_angle > 0) or (pool_max_radial_angle_deg and pool_max_radial_angle_deg > 0):
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

            conf_strict = (
                score_conf["w_fit"] * fit_score + score_conf["w_angle"] * angle_score + score_conf["w_proximity"] * prox_score_strict
            )
            conf_pool = score_conf["w_fit"] * fit_score + score_conf["w_angle"] * angle_score + score_conf["w_proximity"] * prox_score_pool

            # Canonicalize hinge/tip for stable IDs and consistent features.
            # Also compute a legacy index-derived id for one-time migration of old labels.
            legacy_ids: List[str] = []
            try:
                if str(arc.get("source") or "") == "bezier" and arc.get("b_idx") is not None:
                    legacy_key = f"swing|b={int(arc.get('b_idx'))}|l={l_idx}"
                    legacy_ids = ["d_" + hashlib.sha1(legacy_key.encode()).hexdigest()[:10]]
            except Exception:
                legacy_ids = []
            hinge_pt = p0 if d0_center <= d1_center else p1
            tip_pt = p1 if hinge_pt == p0 else p0
            bbox_xyxy = get_bbox(pts + [p0, p1])
            door_id = _stable_swing_candidate_id(
                center=center,
                radius=float(radius),
                angle_span=float(angle_span),
                arc_start=tuple(arc_start),
                arc_end=tuple(arc_end),
                hinge_pt=hinge_pt,
                tip_pt=tip_pt,
                bbox_xyxy=bbox_xyxy,
                quant_step_px=1.0,
            )

            base: Dict[str, Any] = {
                "id": door_id,
                "legacy_ids": legacy_ids,
                "type": "swing",
                "bbox_xyxy": bbox_xyxy,
                "geom": {
                    "center_xy": [float(center[0]), float(center[1])],
                    "hinge_xy": [float(hinge_pt[0]), float(hinge_pt[1])],
                    "tip_xy": [float(tip_pt[0]), float(tip_pt[1])],
                    "arc_endpoints_xy": [[float(arc_start[0]), float(arc_start[1])], [float(arc_end[0]), float(arc_end[1])]],
                },
                "features": {
                    "rmse": float(rmse),
                    "radius": float(radius),
                    "angle_span": float(angle_span),
                    "hinge_dist": float(min_hinge_dist),
                    "len_ratio": float(len_ratio),
                    "center_dist": float(min(d0_center, d1_center)),
                    "radial_angle_deg": float(radial_angle_deg) if radial_angle_deg is not None else 0.0,
                    "tip_to_arc_dist": float(tip_to_arc_dist) if tip_to_arc_dist is not None else 0.0,
                    "arc_source": 1.0 if str(arc.get("source") or "") == "polyline" else 0.0,
                },
                "primitives": {
                    "beziers": [int(arc.get("b_idx"))] if str(arc.get("source") or "") == "bezier" and arc.get("b_idx") is not None else [],
                    "lines": [l_idx],
                    "arc_lines": sorted(list(arc_line_set)) if arc_line_set else [],
                },
                "_circle_key": circle_key,
            }

            cand_pool = dict(base)
            cand_pool["heuristic_confidence"] = float(conf_pool)
            cand_pool["confidence"] = float(conf_pool)
            cand_pool["pool"] = True
            candidate_pool.append(cand_pool)
            had_any_pool_candidate_for_arc = True

            strict_ok = in_strict_len_ratio and strict_hinge_ok and strict_center_ok and strict_radial_ok and strict_tip_ok
            if strict_ok:
                cand_strict = dict(base)
                cand_strict["heuristic_confidence"] = float(conf_strict)
                cand_strict["confidence"] = float(conf_strict)
                cand_strict["pool"] = False
                strict_candidates.append(cand_strict)

        # If we couldn't find any reasonable "leaf" line for this arc, still emit an
        # arc-only candidate into the pool to enable snapping/interactive review.
        # This candidate type is excluded from final door selection logic.
        if not had_any_pool_candidate_for_arc:
            try:
                arc_start = pts[0]
                arc_end = pts[-1]
                arc_bbox = get_bbox(pts)
                pad = max(4.0, float(radius) * 0.15)
                bbox_xyxy = [arc_bbox[0] - pad, arc_bbox[1] - pad, arc_bbox[2] + pad, arc_bbox[3] + pad]

                # Use a conservative heuristic confidence from arc quality only.
                fit_score = max(0.0, 1.0 - (float(rmse) / float(arc_conf["max_circle_fit_rmse"])))
                min_a = float(arc_conf.get("min_angle_deg", 55.0))
                denom = max(1.0, 90.0 - min_a)
                angle_score = (float(angle_span) - min_a) / denom
                angle_score = max(0.0, min(1.0, float(angle_score)))
                conf_arc = 0.65 * float(fit_score) + 0.35 * float(angle_score)
                conf_arc = max(0.0, min(1.0, float(conf_arc)))

                cid = _stable_swing_arc_candidate_id(
                    center=center,
                    radius=float(radius),
                    angle_span=float(angle_span),
                    arc_start=tuple(arc_start),
                    arc_end=tuple(arc_end),
                    bbox_xyxy=bbox_xyxy,
                    quant_step_px=1.0,
                )
                candidate_pool.append(
                    {
                        "id": cid,
                        "legacy_ids": [],
                        "type": "swing_arc",
                        "bbox_xyxy": bbox_xyxy,
                        "geom": {
                            "center_xy": [float(center[0]), float(center[1])],
                            "arc_endpoints_xy": [
                                [float(arc_start[0]), float(arc_start[1])],
                                [float(arc_end[0]), float(arc_end[1])],
                            ],
                        },
                        "heuristic_confidence": float(conf_arc),
                        "confidence": float(conf_arc),
                        "pool": True,
                        "features": {
                            "rmse": float(rmse),
                            "radius": float(radius),
                            "angle_span": float(angle_span),
                            "arc_only": 1.0,
                            "arc_source": 1.0 if str(arc.get("source") or "") == "polyline" else 0.0,
                        },
                        "primitives": {
                            "beziers": [int(arc.get("b_idx"))] if str(arc.get("source") or "") == "bezier" and arc.get("b_idx") is not None else [],
                            "lines": [],
                            "arc_lines": sorted(list(arc_line_set)) if arc_line_set else [],
                        },
                        "_circle_key": circle_key,
                    }
                )
            except Exception:
                pass

    # Apply swing-only circle-cluster suppression (if enabled) on both pools.
    arc_conf = swing_conf.get("arc", {}) if isinstance(swing_conf, dict) else {}
    if arc_conf.get("suppress_circle_clusters", False):
        min_arcs = int(arc_conf.get("circle_cluster_min_arcs", 3))
        min_total_angle = float(arc_conf.get("circle_cluster_min_total_angle_deg", 250.0))
        filtered_strict: List[Dict[str, Any]] = []
        for cand in strict_candidates:
            key = cand.get("_circle_key", None)
            if key is not None:
                if arc_cluster_counts.get(key, 0) >= min_arcs and arc_cluster_sum_angle.get(key, 0.0) >= min_total_angle:
                    continue
            filtered_strict.append(cand)
        strict_candidates = filtered_strict

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

    return strict_candidates, candidate_pool


def detect_double_candidates(*, swing_candidates: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Propose double-door candidates by pairing swing candidates."""
    conf = (config.get("double") or {}) if isinstance(config, dict) else {}
    pairing = conf.get("pairing") or {}
    scoring = conf.get("scoring") or {}

    max_center_dist_ratio = float(pairing.get("max_center_dist_ratio", 3.0) or 3.0)
    max_radius_ratio = float(pairing.get("max_radius_ratio", 1.35) or 1.35)
    max_bbox_iou = float(pairing.get("max_bbox_iou", 0.15) or 0.15)
    max_pairs = int(pairing.get("max_pairs", 600) or 600)

    w_pair = float(scoring.get("w_pair", 0.40) or 0.40)
    w_avg_conf = float(scoring.get("w_avg_conf", 0.60) or 0.60)

    out: List[Dict[str, Any]] = []
    n = len(swing_candidates)
    if n <= 1:
        return out

    # Consider high-confidence swing candidates first to keep pairing cost reasonable.
    swings = list(swing_candidates)
    swings.sort(key=lambda c: float(c.get("confidence", 0.0) or 0.0), reverse=True)
    swings = swings[: min(len(swings), 500)]

    for i in range(len(swings)):
        a = swings[i]
        a_id = a.get("id")
        if a_id is None:
            continue
        a_geom = (a.get("geom") or {}) if isinstance(a.get("geom"), dict) else {}
        bba = _normalize_bbox_xyxy(a.get("bbox_xyxy"))
        if bba is None:
            continue
        try:
            ca = a_geom.get("center_xy") or []
            ax, ay = float(ca[0]), float(ca[1])
        except Exception:
            continue
        ra = float((a.get("features") or {}).get("radius", 0.0) or 0.0)
        if not (ra > 0):
            continue

        for j in range(i + 1, len(swings)):
            b = swings[j]
            b_id = b.get("id")
            if b_id is None:
                continue
            b_geom = (b.get("geom") or {}) if isinstance(b.get("geom"), dict) else {}
            bbb = _normalize_bbox_xyxy(b.get("bbox_xyxy"))
            if bbb is None:
                continue
            try:
                cb = b_geom.get("center_xy") or []
                bx, by = float(cb[0]), float(cb[1])
            except Exception:
                continue
            rb = float((b.get("features") or {}).get("radius", 0.0) or 0.0)
            if not (rb > 0):
                continue

            rmax = max(ra, rb)
            rmin = max(1e-6, min(ra, rb))
            if (rmax / rmin) > max_radius_ratio:
                continue

            center_dist = math.hypot(ax - bx, ay - by)
            if center_dist > (max_center_dist_ratio * rmax):
                continue

            if compute_iou(bba, bbb) > max_bbox_iou:
                continue

            union = _bbox_union(bba, bbb)
            pair_score = 1.0 - (center_dist / max(1e-6, (max_center_dist_ratio * rmax)))
            pair_score = max(0.0, min(1.0, float(pair_score)))
            avg_conf = 0.5 * (float(a.get("confidence", 0.0) or 0.0) + float(b.get("confidence", 0.0) or 0.0))
            conf_pair = w_pair * pair_score + w_avg_conf * avg_conf
            conf_pair = max(0.0, min(1.0, float(conf_pair)))

            cid = _stable_double_candidate_id(swing_ids=(str(a_id), str(b_id)), bbox_xyxy=union, quant_step_px=1.0)
            out.append(
                {
                    "id": cid,
                    "legacy_ids": [],
                    "type": "double",
                    "bbox_xyxy": union,
                    "components": {"swing_ids": sorted([str(a_id), str(b_id)])},
                    "heuristic_confidence": float(conf_pair),
                    "confidence": float(conf_pair),
                    "pool": True,
                    "features": {
                        "center_dist": float(center_dist),
                        "radius_ratio": float(rmax / rmin),
                        "avg_swing_conf": float(avg_conf),
                        "pair_score": float(pair_score),
                    },
                }
            )

    out.sort(key=lambda c: float(c.get("confidence", 0.0) or 0.0), reverse=True)
    return out[: max(0, max_pairs)]


def detect_pocket_candidates(*, lines: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """First-pass pocket door candidates (dashed track-like lines)."""
    conf = (config.get("pocket") or {}) if isinstance(config, dict) else {}
    dashed_conf = conf.get("dashed") or {}
    geom_conf = conf.get("geometry") or {}
    scoring = conf.get("scoring") or {}

    require_dashed = bool(dashed_conf.get("require_dashed", True))
    min_len = float(geom_conf.get("min_track_length_px", 25.0) or 25.0)
    max_len = float(geom_conf.get("max_track_length_px", 600.0) or 600.0)
    pad = float(geom_conf.get("pad_px", 8.0) or 8.0)

    w_dashed = float(scoring.get("w_dashed", 0.40) or 0.40)
    w_length = float(scoring.get("w_length", 0.60) or 0.60)

    out: List[Dict[str, Any]] = []
    for l_idx, line in enumerate(lines):
        p0 = (float(line["p0"]["x"]), float(line["p0"]["y"]))
        p1 = (float(line["p1"]["x"]), float(line["p1"]["y"]))
        is_dashed = _is_dashed_primitive(line)
        if require_dashed and (not is_dashed):
            continue
        length = float(dist_point_to_point(p0, p1))
        if not (min_len <= length <= max_len):
            continue

        x0, y0 = min(p0[0], p1[0]), min(p0[1], p1[1])
        x1, y1 = max(p0[0], p1[0]), max(p0[1], p1[1])
        bbox = [x0 - pad, y0 - pad, x1 + pad, y1 + pad]

        # Simple heuristics: prefer dashed and mid-length.
        len_norm = (length - min_len) / max(1e-6, (max_len - min_len))
        len_norm = max(0.0, min(1.0, float(len_norm)))
        dashed_score = 1.0 if is_dashed else 0.0
        conf_h = w_dashed * dashed_score + w_length * len_norm
        conf_h = max(0.0, min(1.0, float(conf_h)))

        dash_pattern = line.get("dash_pattern") if isinstance(line.get("dash_pattern"), list) else []
        dash_len = float(dash_pattern[0]) if len(dash_pattern) >= 1 else 0.0
        gap_len = float(dash_pattern[1]) if len(dash_pattern) >= 2 else 0.0
        try:
            stroke_width = float(line.get("stroke_width") or 0.0)
        except Exception:
            stroke_width = 0.0

        cid = _stable_line_candidate_id(cand_type="pocket", p0=p0, p1=p1, bbox_xyxy=bbox, quant_step_px=1.0)
        out.append(
            {
                "id": cid,
                "legacy_ids": [],
                "type": "pocket",
                "bbox_xyxy": bbox,
                "heuristic_confidence": float(conf_h),
                "confidence": float(conf_h),
                "pool": True,
                "features": {
                    "track_length_px": float(length),
                    "is_dashed": 1.0 if is_dashed else 0.0,
                    "dash_len": float(dash_len),
                    "gap_len": float(gap_len),
                    "stroke_width": float(stroke_width),
                },
                "primitives": {"lines": [l_idx]},
            }
        )

    out.sort(key=lambda c: float(c.get("confidence", 0.0) or 0.0), reverse=True)
    return out


def detect_bifold_candidates(*, lines: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """First-pass bifold candidates (small connected zig-zag chains)."""
    conf = (config.get("bifold") or {}) if isinstance(config, dict) else {}
    zz = conf.get("zigzag") or {}
    geom_conf = conf.get("geometry") or {}
    scoring = conf.get("scoring") or {}

    min_segments = int(zz.get("min_segments", 3) or 3)
    max_segments = int(zz.get("max_segments", 8) or 8)
    snap_px = float(zz.get("endpoint_snap_px", 3.0) or 3.0)
    min_turn = float(zz.get("min_turn_angle_deg", 30.0) or 30.0)
    pad = float(geom_conf.get("pad_px", 6.0) or 6.0)

    w_segments = float(scoring.get("w_segments", 0.55) or 0.55)
    w_compactness = float(scoring.get("w_compactness", 0.45) or 0.45)

    # Only consider solid lines for bifold (avoid pocket dashed tracks).
    usable_idx: List[int] = []
    for i, ln in enumerate(lines):
        if _is_dashed_primitive(ln):
            continue
        usable_idx.append(i)
    if not usable_idx:
        return []

    def _bin_pt(p: Tuple[float, float]) -> Tuple[int, int]:
        return (_q(p[0], step=snap_px), _q(p[1], step=snap_px))

    # Build endpoint → incident lines.
    end_to_lines: Dict[Tuple[int, int], List[int]] = {}
    line_ends: Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], Tuple[int, int]]] = {}
    for i in usable_idx:
        ln = lines[i]
        p0 = (float(ln["p0"]["x"]), float(ln["p0"]["y"]))
        p1 = (float(ln["p1"]["x"]), float(ln["p1"]["y"]))
        b0 = _bin_pt(p0)
        b1 = _bin_pt(p1)
        line_ends[i] = (p0, p1, b0, b1)
        end_to_lines.setdefault(b0, []).append(i)
        end_to_lines.setdefault(b1, []).append(i)

    # Build line adjacency via shared binned endpoints.
    adj: Dict[int, set[int]] = {i: set() for i in usable_idx}
    for node, inc in end_to_lines.items():
        for a in inc:
            for b in inc:
                if a != b:
                    adj[a].add(b)

    # Connected components over lines.
    seen: set[int] = set()
    comps: List[List[int]] = []
    for i in usable_idx:
        if i in seen:
            continue
        stack = [i]
        comp: List[int] = []
        seen.add(i)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)

    out: List[Dict[str, Any]] = []
    for comp in comps:
        if not (min_segments <= len(comp) <= max_segments):
            continue

        # Node degrees within this component.
        node_deg: Dict[Tuple[int, int], int] = {}
        for li in comp:
            _, _, b0, b1 = line_ends[li]
            node_deg[b0] = node_deg.get(b0, 0) + 1
            node_deg[b1] = node_deg.get(b1, 0) + 1

        end_nodes = [n for n, d in node_deg.items() if d == 1]
        if len(end_nodes) != 2:
            continue  # not a path-like chain

        # Walk the chain in order from one endpoint.
        start_node = end_nodes[0]
        ordered: List[int] = []
        used_lines: set[int] = set()
        current_node = start_node
        prev_line: Optional[int] = None
        for _ in range(len(comp)):
            # pick the next unused line incident to current_node
            candidates = [li for li in end_to_lines.get(current_node, []) if li in comp and li not in used_lines]
            if not candidates:
                break
            nxt = candidates[0]
            if prev_line is not None and len(candidates) > 1:
                # Prefer not to immediately backtrack if possible.
                try:
                    candidates2 = [li for li in candidates if li != prev_line]
                    if candidates2:
                        nxt = candidates2[0]
                except Exception:
                    pass
            ordered.append(nxt)
            used_lines.add(nxt)
            p0, p1, b0, b1 = line_ends[nxt]
            current_node = b1 if current_node == b0 else b0
            prev_line = nxt

        if len(ordered) != len(comp):
            continue

        # Compute average turn angle.
        vecs: List[Tuple[float, float]] = []
        current_node = start_node
        for li in ordered:
            p0, p1, b0, b1 = line_ends[li]
            if current_node == b0:
                vx, vy = (p1[0] - p0[0], p1[1] - p0[1])
                current_node = b1
            else:
                vx, vy = (p0[0] - p1[0], p0[1] - p1[1])
                current_node = b0
            vecs.append((vx, vy))

        turns: List[float] = []
        for k in range(1, len(vecs)):
            ax, ay = vecs[k - 1]
            bx, by = vecs[k]
            an = math.hypot(ax, ay)
            bn = math.hypot(bx, by)
            if an <= 1e-6 or bn <= 1e-6:
                continue
            dot = (ax * bx + ay * by) / (an * bn)
            dot = max(-1.0, min(1.0, dot))
            ang = math.degrees(math.acos(dot))
            turns.append(float(ang))

        avg_turn = float(sum(turns) / max(1, len(turns))) if turns else 0.0
        if avg_turn < min_turn:
            continue

        # Bounding box + chain length.
        xs: List[float] = []
        ys: List[float] = []
        total_len = 0.0
        line_bins: List[Tuple[int, int, int, int]] = []
        for li in ordered:
            p0, p1, b0, b1 = line_ends[li]
            xs.extend([p0[0], p1[0]])
            ys.extend([p0[1], p1[1]])
            total_len += float(dist_point_to_point(p0, p1))
            line_bins.append((b0[0], b0[1], b1[0], b1[1]))

        bbox = [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]
        area = max(1e-6, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        compactness = float(total_len / math.sqrt(area))

        # Heuristic: prefer 3-5 segments and reasonably compact shapes.
        seg_score = max(0.0, min(1.0, (len(ordered) - min_segments) / max(1.0, (max_segments - min_segments))))
        comp_score = max(0.0, min(1.0, compactness / 6.0))
        conf_h = w_segments * seg_score + w_compactness * comp_score
        conf_h = max(0.0, min(1.0, float(conf_h)))

        cid = _stable_bifold_candidate_id(line_bins=line_bins, bbox_xyxy=bbox)
        out.append(
            {
                "id": cid,
                "legacy_ids": [],
                "type": "bifold",
                "bbox_xyxy": bbox,
                "heuristic_confidence": float(conf_h),
                "confidence": float(conf_h),
                "pool": True,
                "features": {
                    "num_segments": float(len(ordered)),
                    "avg_turn_angle_deg": float(avg_turn),
                    "path_length_px": float(total_len),
                    "compactness": float(compactness),
                },
                "primitives": {"lines": sorted(list(comp))},
            }
        )

    out.sort(key=lambda c: float(c.get("confidence", 0.0) or 0.0), reverse=True)
    return out


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

    strict_candidates_all: List[Dict[str, Any]] = []
    candidate_pool_all: List[Dict[str, Any]] = []

    # --- Swing ---
    # Run swing detection whenever enabled and we have *either* beziers or lines.
    # This is important because many PDFs approximate arcs as polylines (line chains).
    if bool((config.get("swing") or {}).get("enabled")) and (beziers or lines):
        strict_swing, pool_swing = detect_swing_candidates(lines=lines, beziers=beziers, line_index=line_index, config=config)
        strict_candidates_all.extend(strict_swing)
        candidate_pool_all.extend(pool_swing)

    # --- Double (pair swing candidates) ---
    if bool((config.get("double") or {}).get("enabled")) and candidate_pool_all:
        double_cands = detect_double_candidates(swing_candidates=[c for c in candidate_pool_all if c.get("type") == "swing"], config=config)
        # For now, treat these as both pool + strict (no strict-vs-pool distinction).
        strict_candidates_all.extend(double_cands)
        candidate_pool_all.extend(double_cands)

    # --- Pocket (dashed tracks) ---
    if bool((config.get("pocket") or {}).get("enabled")) and lines:
        pocket_cands = detect_pocket_candidates(lines=lines, config=config)
        strict_candidates_all.extend(pocket_cands)
        candidate_pool_all.extend(pocket_cands)

    # --- Bi-fold (zig-zag chains) ---
    if bool((config.get("bifold") or {}).get("enabled")) and lines:
        bifold_cands = detect_bifold_candidates(lines=lines, config=config)
        strict_candidates_all.extend(bifold_cands)
        candidate_pool_all.extend(bifold_cands)

    out_conf = config.get("output", {}) or {}
    legacy_min_conf = float(out_conf.get("min_confidence", 0.55) or 0.55)
    min_candidate_conf = float(out_conf.get("min_candidate_confidence", 0.0) or 0.0)
    min_keep_conf = float(out_conf.get("min_confidence_after_reweight", legacy_min_conf) or legacy_min_conf)
    max_candidates_out = int(out_conf.get("max_candidates", 5000) or 5000)
    max_candidates_before_nms = out_conf.get("max_candidates_before_nms", None)
    try:
        max_candidates_before_nms_int: Optional[int] = int(max_candidates_before_nms) if max_candidates_before_nms is not None else None
    except Exception:
        max_candidates_before_nms_int = None

    nms_iou = float(out_conf.get("nms_iou", 0.35) or 0.35)
    max_doors = int(out_conf.get("max_doors", 500) or 500)

    # Per-type reweighters (preferred) + backward compatibility with single `reweighter_path`.
    model_paths_by_type: Dict[str, str] = {}
    cfg_reweighters = config.get("reweighters")
    if isinstance(cfg_reweighters, dict):
        for k, v in cfg_reweighters.items():
            if isinstance(v, str) and v.strip():
                model_paths_by_type[str(k).strip().lower()] = v.strip()
    legacy_model_path = config.get("reweighter_path")
    if isinstance(legacy_model_path, str) and legacy_model_path.strip():
        model_paths_by_type.setdefault("swing", legacy_model_path.strip())

    has_any_model = False
    for p in list(model_paths_by_type.values()):
        try:
            if p and Path(p).exists():
                has_any_model = True
                break
        except Exception:
            continue

    if has_any_model:
        strict_candidates_all = apply_reweighters_by_type(strict_candidates_all, model_paths_by_type=model_paths_by_type)
        candidate_pool_all = apply_reweighters_by_type(candidate_pool_all, model_paths_by_type=model_paths_by_type)

    # Deduplicate candidate pool by stable id (keep highest-confidence record).
    # This matters for snapping (UI pool transport) and prevents duplicates from
    # crowding out spatial diversity.
    if candidate_pool_all:
        by_id: Dict[str, Dict[str, Any]] = {}
        for c in candidate_pool_all:
            cid = c.get("id")
            if cid is None:
                continue
            sid = str(cid)
            prev = by_id.get(sid)
            if prev is None or float(c.get("confidence", 0.0) or 0.0) > float(prev.get("confidence", 0.0) or 0.0):
                by_id[sid] = c
        if by_id:
            candidate_pool_all = list(by_id.values())

    # Always export candidates (for snapping/training), sorted by current confidence.
    candidate_pool_all.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)
    exported_candidates = candidate_pool_all[: max(0, max_candidates_out)]

    # Final selection:
    # - If a model exists, select from the broad pool (post-reweight decisioning).
    # - Otherwise, keep the conservative strict selection behavior.
    selection_src = candidate_pool_all if has_any_model else strict_candidates_all
    if not selection_src:
        return {"doors": [], "candidates": exported_candidates}

    # Only select final doors from canonical door types; other candidate-only types
    # (e.g. swing_arc) should remain available for snapping/review but not appear
    # as auto-detected doors.
    allowed_final_types = {"swing", "double", "pocket", "bifold"}
    selection_src = [c for c in selection_src if str(c.get("type") or "").strip().lower() in allowed_final_types]
    if not selection_src:
        return {"doors": [], "candidates": exported_candidates}

    # Pre-filter using the heuristic score (candidate volume control).
    filtered = [
        c
        for c in selection_src
        if float(c.get("heuristic_confidence", c.get("confidence", 0.0)) or 0.0) >= min_candidate_conf
    ]
    if not filtered:
        return {"doors": [], "candidates": exported_candidates}

    filtered.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)
    if isinstance(max_candidates_before_nms_int, int) and max_candidates_before_nms_int > 0:
        filtered = filtered[:max_candidates_before_nms_int]

    # Post-reweight threshold (actual keep/drop decision).
    kept = [c for c in filtered if float(c.get("confidence", 0.0) or 0.0) >= min_keep_conf]
    if not kept:
        return {"doors": [], "candidates": exported_candidates}

    # NMS on kept candidates, highest confidence first.
    final: List[Dict[str, Any]] = []
    for cand in kept:
        cb = cand.get("bbox_xyxy")
        if not isinstance(cb, list) or len(cb) != 4:
            continue
        overlap = False
        for prev in final:
            if compute_iou(cb, prev["bbox_xyxy"]) > nms_iou:
                overlap = True
                break
        if not overlap:
            final.append(cand)
        if len(final) >= max_doors:
            break

    return {"doors": final, "candidates": exported_candidates}


def debug_explain_unmatched_box(
    *,
    primitives: Dict[str, Any],
    bbox_full_xyxy: List[float],
    config: Dict[str, Any],
    pad_px: float = 20.0,
    max_examples: int = 12,
) -> Dict[str, Any]:
    """Return a structured report explaining why a region produced no snap candidate.

    This is designed for interactive debugging: given a user-drawn selection bbox,
    report whether there are any nearby primitives and which swing-door "arc/leaf"
    qualifications are failing.
    """

    def _nb(b: Any) -> Optional[List[float]]:
        return _normalize_bbox_xyxy(b)

    def _bbox_intersects(a: List[float], b: List[float]) -> bool:
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    nb = _nb(bbox_full_xyxy) or [0.0, 0.0, 0.0, 0.0]
    roi = [nb[0] - pad_px, nb[1] - pad_px, nb[2] + pad_px, nb[3] + pad_px]

    lines: List[Dict[str, Any]] = list(primitives.get("lines", []) or [])
    beziers: List[Dict[str, Any]] = list(primitives.get("beziers", []) or [])

    # Cheap "nearby" filtering using control-point bbox.
    near_lines: List[int] = []
    for i, ln in enumerate(lines):
        try:
            x0 = float(min(ln["p0"]["x"], ln["p1"]["x"]))
            y0 = float(min(ln["p0"]["y"], ln["p1"]["y"]))
            x1 = float(max(ln["p0"]["x"], ln["p1"]["x"]))
            y1 = float(max(ln["p0"]["y"], ln["p1"]["y"]))
        except Exception:
            continue
        if _bbox_intersects([x0, y0, x1, y1], roi):
            near_lines.append(i)

    near_beziers: List[int] = []
    for i, bz in enumerate(beziers):
        try:
            xs = [float(bz["p0"]["x"]), float(bz["p1"]["x"]), float(bz["p2"]["x"]), float(bz["p3"]["x"])]
            ys = [float(bz["p0"]["y"]), float(bz["p1"]["y"]), float(bz["p2"]["y"]), float(bz["p3"]["y"])]
        except Exception:
            continue
        bb = [min(xs), min(ys), max(xs), max(ys)]
        if _bbox_intersects(bb, roi):
            near_beziers.append(i)

    swing_conf = (config.get("swing") or {}) if isinstance(config, dict) else {}
    arc_conf = (swing_conf.get("arc") or {}) if isinstance(swing_conf, dict) else {}
    leaf_conf = (swing_conf.get("leaf") or {}) if isinstance(swing_conf, dict) else {}
    sampling_points = int(swing_conf.get("bezier_sampling_points", 17) or 17)

    try:
        min_r = float(arc_conf.get("min_radius_px", 0.0) or 0.0)
        max_r = float(arc_conf.get("max_radius_px", 1e9) or 1e9)
        max_rmse = float(arc_conf.get("max_circle_fit_rmse", 1e9) or 1e9)
        min_a = float(arc_conf.get("min_angle_deg", 0.0) or 0.0)
        max_a = float(arc_conf.get("max_angle_deg", 1e9) or 1e9)
    except Exception:
        min_r, max_r, max_rmse, min_a, max_a = 0.0, 1e9, 1e9, 0.0, 1e9

    arc_fail_counts: Dict[str, int] = {"radius": 0, "rmse": 0, "angle": 0}
    arc_pass: List[Dict[str, Any]] = []
    arc_examples: List[Dict[str, Any]] = []
    arc_suppression: Dict[str, Any] = {
        "enabled": bool(arc_conf.get("suppress_circle_clusters", False)),
        "near_suppressed_count": 0,
        "near_examples": [],
    }

    # If circle-cluster suppression is enabled, compute cluster membership for arcs
    # near the ROI so we can explain why an obvious arc may yield no candidates.
    circle_counts: Dict[Tuple[int, int, int], int] = {}
    circle_sum_angle: Dict[Tuple[int, int, int], float] = {}
    circle_key_for_arc: Dict[int, Tuple[int, int, int]] = {}
    if bool(arc_conf.get("suppress_circle_clusters", False)):
        cbin = float(arc_conf.get("circle_cluster_center_bin_px", 4.0) or 4.0)
        rbin = float(arc_conf.get("circle_cluster_radius_bin_px", 4.0) or 4.0)
        cbin = cbin if cbin > 0 else 4.0
        rbin = rbin if rbin > 0 else 4.0
        for i in near_beziers:
            bz = beziers[i]
            try:
                pts = sample_bezier(bz["p0"], bz["p1"], bz["p2"], bz["p3"], num_points=sampling_points)
                center, radius, rmse = fit_circle(pts)
                angle_span = get_arc_angle_span(pts, center)
            except Exception:
                continue
            # Only count arcs that pass the arc filters (same as detection).
            if not (min_r <= radius <= max_r):
                continue
            if rmse > max_rmse:
                continue
            if not (min_a <= angle_span <= max_a):
                continue
            key = (int(round(center[0] / cbin)), int(round(center[1] / cbin)), int(round(float(radius) / rbin)))
            circle_key_for_arc[int(i)] = key
            circle_counts[key] = circle_counts.get(key, 0) + 1
            circle_sum_angle[key] = circle_sum_angle.get(key, 0.0) + float(angle_span)

    # For arcs that pass, attempt leaf pairing checks (pool + strict).
    pair_stats = {
        "pairs_tested": 0,
        "pool_pass": 0,
        "strict_pass": 0,
        "pool_fail_counts": {
            "len_ratio": 0,
            "hinge_dist": 0,
            "center_dist": 0,
            "radial_angle": 0,
            "tip_to_arc": 0,
        },
        "strict_fail_counts": {
            "len_ratio": 0,
            "hinge_dist": 0,
            "center_dist": 0,
            "radial_angle": 0,
            "tip_to_arc": 0,
        },
        "examples": [],
    }

    # Always-verbose additions (bounded with truncation guards).
    MAX_NEAR_LINES = 2000
    MAX_NEAR_BEZIERS = 2000
    MAX_PAIRINGS_VERBOSE = 20000
    MAX_POLY_ARCS_VERBOSE = 300
    verbose_pairings: List[Dict[str, Any]] = []
    verbose_truncated = False

    # Mirror the candidate-pool looseners in detect_swing_candidates (so debug matches reality).
    pool_min_len_ratio = 0.22
    pool_max_len_ratio = 2.20
    pool_max_hinge_dist_ratio = 0.55
    pool_require_endpoint_near_center = False
    pool_max_center_dist_ratio = 0.60
    pool_max_radial_angle_deg = 50.0
    pool_max_tip_to_arc_ratio = 0.70

    strict_min_len_ratio = float(leaf_conf.get("min_length_ratio", 0.0) or 0.0)
    strict_max_len_ratio = float(leaf_conf.get("max_length_ratio", 1e9) or 1e9)
    strict_max_hinge_ratio = float(leaf_conf.get("max_hinge_dist_ratio", 0.25) or 0.25)
    strict_req_center = bool(leaf_conf.get("require_endpoint_near_center", False))
    strict_max_center_ratio = float(leaf_conf.get("max_center_dist_ratio", strict_max_hinge_ratio) or strict_max_hinge_ratio)
    strict_max_radial = leaf_conf.get("max_radial_angle_deg", None)
    try:
        strict_max_radial = float(strict_max_radial) if strict_max_radial is not None else None
    except Exception:
        strict_max_radial = None
    strict_max_tip_ratio = leaf_conf.get("max_tip_to_arc_ratio", None)
    try:
        strict_max_tip_ratio = float(strict_max_tip_ratio) if strict_max_tip_ratio is not None else None
    except Exception:
        strict_max_tip_ratio = None

    # Build a line spatial index so we can test pairings near each arc quickly.
    line_index = SpatialIndex(cell_size=200.0)
    for i, ln in enumerate(lines):
        try:
            x0 = float(min(ln["p0"]["x"], ln["p1"]["x"]))
            y0 = float(min(ln["p0"]["y"], ln["p1"]["y"]))
            x1 = float(max(ln["p0"]["x"], ln["p1"]["x"]))
            y1 = float(max(ln["p0"]["y"], ln["p1"]["y"]))
        except Exception:
            continue
        line_index.add(i, [x0, y0, x1, y1])

    # Verbose listing of near primitives.
    near_lines_truncated = False
    near_beziers_truncated = False
    near_lines_list = list(near_lines)
    near_beziers_list = list(near_beziers)
    if len(near_lines_list) > MAX_NEAR_LINES:
        near_lines_list = near_lines_list[:MAX_NEAR_LINES]
        near_lines_truncated = True
    if len(near_beziers_list) > MAX_NEAR_BEZIERS:
        near_beziers_list = near_beziers_list[:MAX_NEAR_BEZIERS]
        near_beziers_truncated = True

    near_primitives = {
        "lines": [],
        "beziers": [],
        "truncated": bool(near_lines_truncated or near_beziers_truncated),
        "counts": {"lines": int(len(near_lines)), "beziers": int(len(near_beziers))},
        "limits": {"max_lines": int(MAX_NEAR_LINES), "max_beziers": int(MAX_NEAR_BEZIERS)},
    }
    if near_lines_truncated:
        near_primitives["lines_truncated"] = True
    if near_beziers_truncated:
        near_primitives["beziers_truncated"] = True
    for l_idx in near_lines_list:
        ln = lines[l_idx]
        try:
            p0 = {"x": float(ln["p0"]["x"]), "y": float(ln["p0"]["y"])}
            p1 = {"x": float(ln["p1"]["x"]), "y": float(ln["p1"]["y"])}
            x0, y0 = min(p0["x"], p1["x"]), min(p0["y"], p1["y"])
            x1, y1 = max(p0["x"], p1["x"]), max(p0["y"], p1["y"])
            rec = {
                "l_idx": int(l_idx),
                "p0": p0,
                "p1": p1,
                "len_px": float(dist_point_to_point((p0["x"], p0["y"]), (p1["x"], p1["y"]))),
                "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                "is_dashed": bool(_is_dashed_primitive(ln)),
            }
            near_primitives["lines"].append(rec)
        except Exception:
            continue

    for b_idx in near_beziers_list:
        bz = beziers[b_idx]
        try:
            p0 = {"x": float(bz["p0"]["x"]), "y": float(bz["p0"]["y"])}
            p1 = {"x": float(bz["p1"]["x"]), "y": float(bz["p1"]["y"])}
            p2 = {"x": float(bz["p2"]["x"]), "y": float(bz["p2"]["y"])}
            p3 = {"x": float(bz["p3"]["x"]), "y": float(bz["p3"]["y"])}
            xs = [p0["x"], p1["x"], p2["x"], p3["x"]]
            ys = [p0["y"], p1["y"], p2["y"], p3["y"]]
            rec = {
                "b_idx": int(b_idx),
                "p0": p0,
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "bbox_xyxy": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
            }
            near_primitives["beziers"].append(rec)
        except Exception:
            continue

    # Polyline-arc diagnostics within the ROI subset of lines (always verbose).
    poly_dbg = _debug_polyline_arcs_from_lines_subset(lines=lines, line_indices=near_lines, config=config)
    poly_arc_candidates = list(poly_dbg.get("arc_candidates", []) or [])
    if len(poly_arc_candidates) > MAX_POLY_ARCS_VERBOSE:
        poly_dbg["arc_candidates"] = poly_arc_candidates[:MAX_POLY_ARCS_VERBOSE]
        poly_dbg["truncated"] = True
        poly_dbg["total_arc_candidates"] = int(len(poly_arc_candidates))
        poly_dbg["limits"] = {"max_arc_candidates": int(MAX_POLY_ARCS_VERBOSE)}
    else:
        poly_dbg["truncated"] = False
        poly_dbg["total_arc_candidates"] = int(len(poly_arc_candidates))
        poly_dbg["limits"] = {"max_arc_candidates": int(MAX_POLY_ARCS_VERBOSE)}

    # Include polyline-arc candidates (even failed) as part of arc examples for debugging.
    # We also compute how many pass arc filters (fails == []) for quick triage.
    poly_pass = [a for a in list(poly_dbg.get("arc_candidates", []) or []) if not (a.get("fails") or [])]
    poly_dbg["arc_pass_near_count"] = int(len(poly_pass))
    poly_fail_counts = {"radius": 0, "rmse": 0, "angle": 0}
    for a in list(poly_dbg.get("arc_candidates", []) or []):
        fails = list(a.get("fails") or [])
        if "arc.radius" in fails:
            poly_fail_counts["radius"] += 1
        if "arc.rmse" in fails:
            poly_fail_counts["rmse"] += 1
        if "arc.angle_span" in fails:
            poly_fail_counts["angle"] += 1
    poly_dbg["arc_fail_counts_near"] = poly_fail_counts

    # Compute circle-cluster suppression over *near* arcs (bezier + polyline) so the debug report
    # can explain why an arc was filtered. (Scope is near-ROI; this is sufficient for label bubbles.)
    circle_counts_all: Dict[Tuple[int, int, int], int] = {}
    circle_sum_angle_all: Dict[Tuple[int, int, int], float] = {}
    circle_key_for_any_arc: Dict[str, Tuple[int, int, int]] = {}
    cbin = float(arc_conf.get("circle_cluster_center_bin_px", 4.0) or 4.0)
    rbin = float(arc_conf.get("circle_cluster_radius_bin_px", 4.0) or 4.0)
    cbin = cbin if cbin > 0 else 4.0
    rbin = rbin if rbin > 0 else 4.0

    def _circle_key(center_xy: Tuple[float, float], radius: float) -> Tuple[int, int, int]:
        return (int(round(center_xy[0] / cbin)), int(round(center_xy[1] / cbin)), int(round(float(radius) / rbin)))

    # Add bezier-pass arcs (already computed in arc_pass).
    for ex in list(arc_pass):
        try:
            key = _circle_key((float(ex.get("center_xy")[0]), float(ex.get("center_xy")[1])), float(ex.get("radius")))
        except Exception:
            continue
        arc_id = f"bezier:{int(ex.get('b_idx'))}"
        circle_key_for_any_arc[arc_id] = key
        circle_counts_all[key] = circle_counts_all.get(key, 0) + 1
        circle_sum_angle_all[key] = circle_sum_angle_all.get(key, 0.0) + float(ex.get("angle_span_deg", 0.0) or 0.0)

    # Add polyline-pass arcs (from poly_dbg).
    for ex in poly_pass:
        try:
            key = _circle_key((float(ex.get("center_xy")[0]), float(ex.get("center_xy")[1])), float(ex.get("radius")))
        except Exception:
            continue
        arc_id = f"polyline:{int(ex.get('comp_idx'))}"
        circle_key_for_any_arc[arc_id] = key
        circle_counts_all[key] = circle_counts_all.get(key, 0) + 1
        circle_sum_angle_all[key] = circle_sum_angle_all.get(key, 0.0) + float(ex.get("angle_span_deg", 0.0) or 0.0)

    # Update suppression examples to include both sources.
    if bool(arc_conf.get("suppress_circle_clusters", False)):
        min_arcs = int(arc_conf.get("circle_cluster_min_arcs", 3) or 3)
        min_total_angle = float(arc_conf.get("circle_cluster_min_total_angle_deg", 250.0) or 250.0)
        # Reset and rebuild near suppression details to cover both bezier and polyline.
        arc_suppression["near_suppressed_count"] = 0
        arc_suppression["near_examples"] = []
        arc_suppression["scope"] = "near_roi"
        arc_suppression["min_arcs"] = int(min_arcs)
        arc_suppression["min_total_angle_deg"] = float(min_total_angle)
        arc_suppression["bin_px"] = {"center": float(cbin), "radius": float(rbin)}

        def _add_supp_example(*, arc_id: str, b_idx: Optional[int] = None, comp_idx: Optional[int] = None) -> None:
            if len(arc_suppression["near_examples"]) >= int(max_examples):
                return
            key = circle_key_for_any_arc.get(arc_id)
            if key is None:
                return
            cnt = int(circle_counts_all.get(key, 0))
            tot = float(circle_sum_angle_all.get(key, 0.0))
            suppressed = bool(cnt >= min_arcs and tot >= min_total_angle)
            if suppressed:
                arc_suppression["near_suppressed_count"] = int(arc_suppression.get("near_suppressed_count", 0) or 0) + 1
            rec = {
                "source": "bezier" if b_idx is not None else "polyline",
                "circle_key": list(key),
                "cluster_count": cnt,
                "cluster_total_angle_deg": tot,
                "suppressed": suppressed,
            }
            if b_idx is not None:
                rec["b_idx"] = int(b_idx)
            if comp_idx is not None:
                rec["comp_idx"] = int(comp_idx)
            arc_suppression["near_examples"].append(rec)

        for ex in list(arc_pass):
            try:
                _add_supp_example(arc_id=f"bezier:{int(ex.get('b_idx'))}", b_idx=int(ex.get("b_idx")))
            except Exception:
                continue
        for ex in poly_pass:
            try:
                _add_supp_example(arc_id=f"polyline:{int(ex.get('comp_idx'))}", comp_idx=int(ex.get("comp_idx")))
            except Exception:
                continue

    # Augment bezier arc examples to include center_xy (needed for suppression clustering above).
    # (Backward compatible: just adds a field.)
    arc_pass_by_idx: Dict[int, Tuple[float, float]] = {}
    for ex in list(arc_pass):
        try:
            arc_pass_by_idx[int(ex["b_idx"])] = (float(ex.get("radius")), float(ex.get("angle_span_deg")))
        except Exception:
            continue

    # Evaluate bezier arcs near ROI.
    arc_pass = []  # recompute pass list with center info included
    arc_examples = []
    arc_fail_counts = {"radius": 0, "rmse": 0, "angle": 0}

    for b_idx in near_beziers:
        bz = beziers[b_idx]
        try:
            pts = sample_bezier(bz["p0"], bz["p1"], bz["p2"], bz["p3"], num_points=sampling_points)
        except Exception:
            continue
        center, radius, rmse = fit_circle(pts)
        angle_span = get_arc_angle_span(pts, center)

        fails: List[str] = []
        if not (min_r <= radius <= max_r):
            fails.append("arc.radius")
            arc_fail_counts["radius"] += 1
        if rmse > max_rmse:
            fails.append("arc.rmse")
            arc_fail_counts["rmse"] += 1
        if not (min_a <= angle_span <= max_a):
            fails.append("arc.angle_span")
            arc_fail_counts["angle"] += 1

        ex = {
            "b_idx": int(b_idx),
            "radius": float(radius),
            "rmse": float(rmse),
            "angle_span_deg": float(angle_span),
            "center_xy": [float(center[0]), float(center[1])],
            "fails": fails,
            "arc_conf": {"min_radius_px": min_r, "max_radius_px": max_r, "max_rmse": max_rmse, "min_angle_deg": min_a, "max_angle_deg": max_a},
        }
        if len(arc_examples) < max_examples:
            arc_examples.append(ex)

        if fails:
            continue
        arc_pass.append(ex)

        # Leaf pairing checks.
        arc_bbox = get_bbox(pts)
        query_bbox = [
            arc_bbox[0] - radius * 0.5,
            arc_bbox[1] - radius * 0.5,
            arc_bbox[2] + radius * 0.5,
            arc_bbox[3] + radius * 0.5,
        ]
        nearby_line_indices = line_index.query(query_bbox)

        arc_start = pts[0]
        arc_end = pts[-1]

        for l_idx in nearby_line_indices:
            ln = lines[l_idx]
            try:
                p0 = (float(ln["p0"]["x"]), float(ln["p0"]["y"]))
                p1 = (float(ln["p1"]["x"]), float(ln["p1"]["y"]))
            except Exception:
                continue
            l_len = float(dist_point_to_point(p0, p1))
            len_ratio = (l_len / float(radius)) if float(radius) > 1e-6 else 0.0

            # Distances used for hinge/center checks.
            d0_start = float(dist_point_to_point(p0, arc_start))
            d0_end = float(dist_point_to_point(p0, arc_end))
            d1_start = float(dist_point_to_point(p1, arc_start))
            d1_end = float(dist_point_to_point(p1, arc_end))
            d0_center = float(dist_point_to_point(p0, center))
            d1_center = float(dist_point_to_point(p1, center))
            min_hinge_dist = float(min(d0_start, d0_end, d1_start, d1_end))

            # Identify hinge/tip consistently (closer endpoint to circle center = hinge).
            hinge_pt = p0 if d0_center <= d1_center else p1
            tip_pt = p1 if hinge_pt == p0 else p0
            target_pt = arc_start if dist_point_to_point(tip_pt, arc_start) <= dist_point_to_point(tip_pt, arc_end) else arc_end
            tip_to_arc_dist = float(dist_point_to_point(tip_pt, target_pt))

            # Radial angle check (leaf direction vs radius direction).
            radial_angle_deg = None
            lx, ly = (tip_pt[0] - hinge_pt[0], tip_pt[1] - hinge_pt[1])
            rx, ry = (target_pt[0] - center[0], target_pt[1] - center[1])
            lnrm = math.hypot(lx, ly)
            rnrm = math.hypot(rx, ry)
            if lnrm > 1e-6 and rnrm > 1e-6:
                dot = (lx * rx + ly * ry) / (lnrm * rnrm)
                dot = max(-1.0, min(1.0, float(dot)))
                radial_angle_deg = float(math.degrees(math.acos(dot)))

            pair_stats["pairs_tested"] += 1

            # Pool checks.
            pool_fail: List[str] = []
            if not (pool_min_len_ratio <= len_ratio <= pool_max_len_ratio):
                pool_fail.append("leaf.len_ratio")
                pair_stats["pool_fail_counts"]["len_ratio"] += 1
            pool_hinge_ok = True
            if min_hinge_dist > float(radius) * pool_max_hinge_dist_ratio:
                if min(d0_center, d1_center) > float(radius) * pool_max_hinge_dist_ratio:
                    pool_hinge_ok = False
            if not pool_hinge_ok:
                pool_fail.append("leaf.hinge_dist")
                pair_stats["pool_fail_counts"]["hinge_dist"] += 1
            if pool_require_endpoint_near_center:
                if min(d0_center, d1_center) > float(radius) * pool_max_center_dist_ratio:
                    pool_fail.append("leaf.center_dist")
                    pair_stats["pool_fail_counts"]["center_dist"] += 1
            if radial_angle_deg is not None and pool_max_radial_angle_deg and pool_max_radial_angle_deg > 0:
                if radial_angle_deg > pool_max_radial_angle_deg:
                    pool_fail.append("leaf.radial_angle")
                    pair_stats["pool_fail_counts"]["radial_angle"] += 1
            if pool_max_tip_to_arc_ratio is not None and pool_max_tip_to_arc_ratio > 0:
                if tip_to_arc_dist > float(radius) * pool_max_tip_to_arc_ratio:
                    pool_fail.append("leaf.tip_to_arc")
                    pair_stats["pool_fail_counts"]["tip_to_arc"] += 1

            pool_ok = len(pool_fail) == 0
            if pool_ok:
                pair_stats["pool_pass"] += 1

            # Strict checks.
            strict_fail: List[str] = []
            if not (strict_min_len_ratio <= len_ratio <= strict_max_len_ratio):
                strict_fail.append("leaf.len_ratio")
                pair_stats["strict_fail_counts"]["len_ratio"] += 1
            strict_hinge_ok = True
            if min_hinge_dist > float(radius) * strict_max_hinge_ratio:
                if min(d0_center, d1_center) > float(radius) * strict_max_hinge_ratio:
                    strict_hinge_ok = False
            if not strict_hinge_ok:
                strict_fail.append("leaf.hinge_dist")
                pair_stats["strict_fail_counts"]["hinge_dist"] += 1
            if strict_req_center:
                if min(d0_center, d1_center) > float(radius) * strict_max_center_ratio:
                    strict_fail.append("leaf.center_dist")
                    pair_stats["strict_fail_counts"]["center_dist"] += 1
            if radial_angle_deg is not None and strict_max_radial is not None and strict_max_radial > 0:
                if radial_angle_deg > strict_max_radial:
                    strict_fail.append("leaf.radial_angle")
                    pair_stats["strict_fail_counts"]["radial_angle"] += 1
            if strict_max_tip_ratio is not None and strict_max_tip_ratio > 0:
                if tip_to_arc_dist > float(radius) * strict_max_tip_ratio:
                    strict_fail.append("leaf.tip_to_arc")
                    pair_stats["strict_fail_counts"]["tip_to_arc"] += 1

            strict_ok = len(strict_fail) == 0
            if strict_ok:
                pair_stats["strict_pass"] += 1

            # Always-verbose per-pair record (bounded).
            if len(verbose_pairings) < MAX_PAIRINGS_VERBOSE:
                verbose_pairings.append(
                    {
                        "arc_source": "bezier",
                        "b_idx": int(b_idx),
                        "l_idx": int(l_idx),
                        "radius": float(radius),
                        "rmse": float(rmse),
                        "angle_span_deg": float(angle_span),
                        "len_ratio": float(len_ratio),
                        "hinge_dist": float(min_hinge_dist),
                        "center_dist": float(min(d0_center, d1_center)),
                        "radial_angle_deg": float(radial_angle_deg) if radial_angle_deg is not None else None,
                        "tip_to_arc_dist": float(tip_to_arc_dist),
                        "pool_fail": pool_fail,
                        "strict_fail": strict_fail,
                    }
                )
            else:
                verbose_truncated = True

            if len(pair_stats["examples"]) < max_examples and (not pool_ok or not strict_ok):
                pair_stats["examples"].append(
                    {
                        "b_idx": int(b_idx),
                        "l_idx": int(l_idx),
                        "radius": float(radius),
                        "rmse": float(rmse),
                        "angle_span_deg": float(angle_span),
                        "len_ratio": float(len_ratio),
                        "hinge_dist": float(min_hinge_dist),
                        "center_dist": float(min(d0_center, d1_center)),
                        "radial_angle_deg": float(radial_angle_deg) if radial_angle_deg is not None else None,
                        "tip_to_arc_dist": float(tip_to_arc_dist),
                        "pool_fail": pool_fail,
                        "strict_fail": strict_fail,
                    }
                )

    # Leaf pairing for polyline arcs that pass arc filters (always-verbose).
    # We mimic detect_swing_candidates by skipping leaf lines that are part of the arc polyline itself.
    for ex in list(poly_pass):
        try:
            comp_idx = int(ex.get("comp_idx"))
            line_idxs = [int(x) for x in list(ex.get("line_idxs") or [])]
            center_xy = ex.get("center_xy") or [0.0, 0.0]
            center = (float(center_xy[0]), float(center_xy[1]))
            radius = float(ex.get("radius", 0.0) or 0.0)
        except Exception:
            continue
        if not (radius > 0):
            continue
        # Approximate arc bbox using just the endpoints of the chain (cheap) would be too weak;
        # instead, use the bbox of all segment endpoints.
        pts: List[Tuple[float, float]] = []
        for li in line_idxs:
            try:
                ln = lines[int(li)]
                pts.append((float(ln["p0"]["x"]), float(ln["p0"]["y"])))
                pts.append((float(ln["p1"]["x"]), float(ln["p1"]["y"])))
            except Exception:
                continue
        if not pts:
            continue
        arc_bbox = get_bbox(pts)
        query_bbox = [
            arc_bbox[0] - radius * 0.5,
            arc_bbox[1] - radius * 0.5,
            arc_bbox[2] + radius * 0.5,
            arc_bbox[3] + radius * 0.5,
        ]
        nearby_line_indices = line_index.query(query_bbox)
        arc_line_set = set(int(x) for x in line_idxs)

        # Use first/last endpoints as arc endpoints for hinge/tip checks.
        arc_start = pts[0]
        arc_end = pts[-1]
        for l_idx in nearby_line_indices:
            if int(l_idx) in arc_line_set:
                continue
            ln = lines[l_idx]
            try:
                p0 = (float(ln["p0"]["x"]), float(ln["p0"]["y"]))
                p1 = (float(ln["p1"]["x"]), float(ln["p1"]["y"]))
            except Exception:
                continue
            l_len = float(dist_point_to_point(p0, p1))
            len_ratio = (l_len / float(radius)) if float(radius) > 1e-6 else 0.0

            d0_start = float(dist_point_to_point(p0, arc_start))
            d0_end = float(dist_point_to_point(p0, arc_end))
            d1_start = float(dist_point_to_point(p1, arc_start))
            d1_end = float(dist_point_to_point(p1, arc_end))
            d0_center = float(dist_point_to_point(p0, center))
            d1_center = float(dist_point_to_point(p1, center))
            min_hinge_dist = float(min(d0_start, d0_end, d1_start, d1_end))

            hinge_pt = p0 if d0_center <= d1_center else p1
            tip_pt = p1 if hinge_pt == p0 else p0
            target_pt = arc_start if dist_point_to_point(tip_pt, arc_start) <= dist_point_to_point(tip_pt, arc_end) else arc_end
            tip_to_arc_dist = float(dist_point_to_point(tip_pt, target_pt))

            radial_angle_deg = None
            lx, ly = (tip_pt[0] - hinge_pt[0], tip_pt[1] - hinge_pt[1])
            rx, ry = (target_pt[0] - center[0], target_pt[1] - center[1])
            lnrm = math.hypot(lx, ly)
            rnrm = math.hypot(rx, ry)
            if lnrm > 1e-6 and rnrm > 1e-6:
                dot = (lx * rx + ly * ry) / (lnrm * rnrm)
                dot = max(-1.0, min(1.0, float(dot)))
                radial_angle_deg = float(math.degrees(math.acos(dot)))

            pair_stats["pairs_tested"] += 1

            pool_fail: List[str] = []
            if not (pool_min_len_ratio <= len_ratio <= pool_max_len_ratio):
                pool_fail.append("leaf.len_ratio")
                pair_stats["pool_fail_counts"]["len_ratio"] += 1
            pool_hinge_ok = True
            if min_hinge_dist > float(radius) * pool_max_hinge_dist_ratio:
                if min(d0_center, d1_center) > float(radius) * pool_max_hinge_dist_ratio:
                    pool_hinge_ok = False
            if not pool_hinge_ok:
                pool_fail.append("leaf.hinge_dist")
                pair_stats["pool_fail_counts"]["hinge_dist"] += 1
            if pool_require_endpoint_near_center:
                if min(d0_center, d1_center) > float(radius) * pool_max_center_dist_ratio:
                    pool_fail.append("leaf.center_dist")
                    pair_stats["pool_fail_counts"]["center_dist"] += 1
            if radial_angle_deg is not None and pool_max_radial_angle_deg and pool_max_radial_angle_deg > 0:
                if radial_angle_deg > pool_max_radial_angle_deg:
                    pool_fail.append("leaf.radial_angle")
                    pair_stats["pool_fail_counts"]["radial_angle"] += 1
            if pool_max_tip_to_arc_ratio is not None and pool_max_tip_to_arc_ratio > 0:
                if tip_to_arc_dist > float(radius) * pool_max_tip_to_arc_ratio:
                    pool_fail.append("leaf.tip_to_arc")
                    pair_stats["pool_fail_counts"]["tip_to_arc"] += 1

            pool_ok = len(pool_fail) == 0
            if pool_ok:
                pair_stats["pool_pass"] += 1

            strict_fail: List[str] = []
            if not (strict_min_len_ratio <= len_ratio <= strict_max_len_ratio):
                strict_fail.append("leaf.len_ratio")
                pair_stats["strict_fail_counts"]["len_ratio"] += 1
            strict_hinge_ok = True
            if min_hinge_dist > float(radius) * strict_max_hinge_ratio:
                if min(d0_center, d1_center) > float(radius) * strict_max_hinge_ratio:
                    strict_hinge_ok = False
            if not strict_hinge_ok:
                strict_fail.append("leaf.hinge_dist")
                pair_stats["strict_fail_counts"]["hinge_dist"] += 1
            if strict_req_center:
                if min(d0_center, d1_center) > float(radius) * strict_max_center_ratio:
                    strict_fail.append("leaf.center_dist")
                    pair_stats["strict_fail_counts"]["center_dist"] += 1
            if radial_angle_deg is not None and strict_max_radial is not None and strict_max_radial > 0:
                if radial_angle_deg > strict_max_radial:
                    strict_fail.append("leaf.radial_angle")
                    pair_stats["strict_fail_counts"]["radial_angle"] += 1
            if strict_max_tip_ratio is not None and strict_max_tip_ratio > 0:
                if tip_to_arc_dist > float(radius) * strict_max_tip_ratio:
                    strict_fail.append("leaf.tip_to_arc")
                    pair_stats["strict_fail_counts"]["tip_to_arc"] += 1

            strict_ok = len(strict_fail) == 0
            if strict_ok:
                pair_stats["strict_pass"] += 1

            if len(verbose_pairings) < MAX_PAIRINGS_VERBOSE:
                verbose_pairings.append(
                    {
                        "arc_source": "polyline",
                        "comp_idx": int(comp_idx),
                        "l_idx": int(l_idx),
                        "radius": float(radius),
                        "len_ratio": float(len_ratio),
                        "hinge_dist": float(min_hinge_dist),
                        "center_dist": float(min(d0_center, d1_center)),
                        "radial_angle_deg": float(radial_angle_deg) if radial_angle_deg is not None else None,
                        "tip_to_arc_dist": float(tip_to_arc_dist),
                        "pool_fail": pool_fail,
                        "strict_fail": strict_fail,
                    }
                )
            else:
                verbose_truncated = True

    # Pocket-line quick check (useful when swing arcs are absent).
    pocket_conf = (config.get("pocket") or {}) if isinstance(config, dict) else {}
    pocket_geom = (pocket_conf.get("geometry") or {}) if isinstance(pocket_conf, dict) else {}
    dashed_conf = (pocket_conf.get("dashed") or {}) if isinstance(pocket_conf, dict) else {}
    require_dashed = bool(dashed_conf.get("require_dashed", True))
    min_track_len = float(pocket_geom.get("min_track_length_px", 25.0) or 25.0)
    max_track_len = float(pocket_geom.get("max_track_length_px", 600.0) or 600.0)
    pocket_hits = 0
    pocket_examples: List[Dict[str, Any]] = []
    for l_idx in near_lines[: max(50, max_examples * 5)]:
        ln = lines[l_idx]
        is_dashed = _is_dashed_primitive(ln)
        if require_dashed and not is_dashed:
            continue
        try:
            length = float(_line_len_px(ln))
        except Exception:
            continue
        if not (min_track_len <= length <= max_track_len):
            continue
        pocket_hits += 1
        if len(pocket_examples) < max_examples:
            pocket_examples.append({"l_idx": int(l_idx), "length_px": float(length), "is_dashed": bool(is_dashed)})

    report: Dict[str, Any] = {
        "kind": "unmatched_box_debug_v1",
        "verbose": True,
        "truncation": {
            "near_primitives_truncated": bool(near_primitives.get("truncated")),
            "polyline_arc_near_truncated": bool(poly_dbg.get("truncated")),
            "leaf_pairings_truncated": bool(verbose_truncated),
            "limits": {
                "max_near_lines": int(MAX_NEAR_LINES),
                "max_near_beziers": int(MAX_NEAR_BEZIERS),
                "max_polyline_arc_candidates": int(MAX_POLY_ARCS_VERBOSE),
                "max_leaf_pairings": int(MAX_PAIRINGS_VERBOSE),
            },
        },
        "bbox_full_xyxy": [float(v) for v in nb],
        "roi_full_xyxy": [float(v) for v in roi],
        "thresholds": {
            "swing": {
                "arc": {
                    "min_radius_px": float(min_r),
                    "max_radius_px": float(max_r),
                    "max_circle_fit_rmse": float(max_rmse),
                    "min_angle_deg": float(min_a),
                    "max_angle_deg": float(max_a),
                    "suppress_circle_clusters": bool(arc_conf.get("suppress_circle_clusters", False)),
                    "circle_cluster_center_bin_px": float(arc_conf.get("circle_cluster_center_bin_px", 4.0) or 4.0),
                    "circle_cluster_radius_bin_px": float(arc_conf.get("circle_cluster_radius_bin_px", 4.0) or 4.0),
                    "circle_cluster_min_arcs": int(arc_conf.get("circle_cluster_min_arcs", 3) or 3),
                    "circle_cluster_min_total_angle_deg": float(arc_conf.get("circle_cluster_min_total_angle_deg", 250.0) or 250.0),
                },
                "leaf_strict": {
                    "min_length_ratio": float(strict_min_len_ratio),
                    "max_length_ratio": float(strict_max_len_ratio),
                    "max_hinge_dist_ratio": float(strict_max_hinge_ratio),
                    "require_endpoint_near_center": bool(strict_req_center),
                    "max_center_dist_ratio": float(strict_max_center_ratio),
                    "max_radial_angle_deg": float(strict_max_radial) if strict_max_radial is not None else None,
                    "max_tip_to_arc_ratio": float(strict_max_tip_ratio) if strict_max_tip_ratio is not None else None,
                },
                "leaf_pool": {
                    "min_length_ratio": float(pool_min_len_ratio),
                    "max_length_ratio": float(pool_max_len_ratio),
                    "max_hinge_dist_ratio": float(pool_max_hinge_dist_ratio),
                    "require_endpoint_near_center": bool(pool_require_endpoint_near_center),
                    "max_center_dist_ratio": float(pool_max_center_dist_ratio),
                    "max_radial_angle_deg": float(pool_max_radial_angle_deg),
                    "max_tip_to_arc_ratio": float(pool_max_tip_to_arc_ratio),
                },
            }
        },
        "near_primitives": near_primitives,
        "counts": {
            "lines_total": int(len(lines)),
            "beziers_total": int(len(beziers)),
            "lines_near_roi": int(len(near_lines)),
            "beziers_near_roi": int(len(near_beziers)),
        },
        "swing": {
            "enabled": bool(swing_conf.get("enabled", False)),
            "arc_near_examples": arc_examples,
            "arc_pass_near_count": int(len(arc_pass)),
            "arc_fail_counts_near": arc_fail_counts,
            "arc_circle_cluster_suppression": arc_suppression,
            "leaf_pair_stats_near": pair_stats,
            "polyline_arc_near": poly_dbg,
            "leaf_pairings_verbose": {
                "count": int(len(verbose_pairings)),
                "truncated": bool(verbose_truncated),
                "max_records": int(MAX_PAIRINGS_VERBOSE),
                "records": verbose_pairings,
            },
        },
        "pocket": {
            "enabled": bool(pocket_conf.get("enabled", False)),
            "require_dashed": require_dashed,
            "near_hits": int(pocket_hits),
            "near_examples": pocket_examples,
        },
        "note": "Verbose report: includes near primitives and per-(arc,line) exclusion reasons. If beziers_near_roi is 0, the door swing may be rasterized or drawn as polylines; if arc_fail_counts are high, loosen swing.arc thresholds (especially max_radius_px).",
    }
    return report

