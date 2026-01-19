"""Core door detection logic using vector primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
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
from door_detector.doors.dedupe import bbox_containment, is_duplicate, suppress_duplicates
from door_detector.perf import enabled as perf_enabled, span as perf_span, log as perf_log


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s not in ("", "0", "false", "f", "no", "n", "off")


def _dedupe_debug_log(enabled: bool, kind: str, **fields: Any) -> None:
    """Best-effort debug logging for dedupe investigation.

    We avoid relying on logging configuration (Step 2 CLI often has none).
    """
    if not enabled:
        return
    try:
        payload = {"kind": str(kind), **fields}
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except Exception:
        s = str({"kind": kind, **fields})
    try:
        sys.stderr.write(f"[door_detector][dedupe] {s}\n")
    except Exception:
        return

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


def _door_detector_base_dirs_from_config(config: Dict[str, Any]) -> List[Path]:
    """Return candidate base dirs for resolving relative model paths (best-effort).

    This is intentionally permissive: the UI/CLI may run from a different CWD than
    the repo root, while configs often reference `models/...` relative to the repo.
    """
    out: List[Path] = []

    base = config.get("_door_detector_base_dir")
    if isinstance(base, str) and base.strip():
        try:
            out.append(Path(base.strip()))
        except Exception:
            pass

    # Fallback: infer repo root from package location (works in this repo checkout).
    # `.../door_detector/doors/detect.py` -> parents[2] == repo root.
    try:
        out.append(Path(__file__).resolve().parents[2])
    except Exception:
        pass

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in out:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    return uniq


def _resolve_existing_path(path_str: str, *, base_dirs: List[Path]) -> Optional[str]:
    """Resolve a model path string to an existing filesystem path if possible."""
    if not isinstance(path_str, str) or not path_str.strip():
        return None
    raw = path_str.strip()

    try:
        p = Path(raw)
    except Exception:
        return None

    try:
        if p.is_absolute():
            return str(p) if p.exists() else None
    except Exception:
        return None

    # Prefer config-provided base dirs over CWD.
    #
    # Rationale:
    # - In Door Detector, configs often reference `models/...` relative to the repo or an
    #   artifacts root, while the process may be launched from an arbitrary CWD.
    # - When multiple `models/...` exist (e.g. a real repo model + a temporary test
    #   model), resolving relative-to-CWD first can silently pick the wrong file.
    for bd in base_dirs:
        try:
            cand = bd / p
            if cand.exists():
                return str(cand)
        except Exception:
            continue

    # Fallback: as-is (relative to CWD).
    try:
        if p.exists():
            return str(p)
    except Exception:
        pass

    return None

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
    allow_dashed = bool(poly_conf.get("allow_dashed", False))
    allow_branches = bool(poly_conf.get("allow_branches", True))
    max_paths_per_component = int(poly_conf.get("max_paths_per_component", 200) or 200)
    # Fit tolerance: reuse arc's circle fit constraint if present, else a mild default.
    max_circle_fit_rmse = float(arc_conf.get("max_circle_fit_rmse", 2.5) or 2.5)
    # Optional relative RMSE tolerance (RMSE / radius). Useful for large-radius polylines where
    # absolute pixel RMSE naturally grows with scale, even when the arc is visually correct.
    # If <=0, disabled.
    max_circle_fit_rmse_ratio = float(arc_conf.get("max_circle_fit_rmse_ratio", 0.0) or 0.0)

    def _rmse_ok(rmse: float, radius: float) -> bool:
        try:
            rmse_f = float(rmse)
            r_f = float(radius)
        except Exception:
            return False
        if rmse_f <= float(max_circle_fit_rmse):
            return True
        if max_circle_fit_rmse_ratio and max_circle_fit_rmse_ratio > 0 and r_f > 1e-6:
            return (rmse_f / r_f) <= float(max_circle_fit_rmse_ratio)
        return False

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
        if _is_dashed_primitive(ln) and not allow_dashed:
            continue
        try:
            p0 = (float(ln["p0"]["x"]), float(ln["p0"]["y"]))
            p1 = (float(ln["p1"]["x"]), float(ln["p1"]["y"]))
        except Exception:
            continue
        # Exclude perfectly axis-aligned segments from polyline-arc extraction.
        # These are overwhelmingly wall/leaf stubs and can connect components into huge
        # graphs, hiding the actual curved arc chain. (The leaf pairing stage still uses
        # all lines, so excluding axis-aligned segments here is safe.)
        try:
            dx = abs(float(p1[0]) - float(p0[0]))
            dy = abs(float(p1[1]) - float(p0[1]))
        except Exception:
            dx, dy = 0.0, 0.0
        if dx < 1e-3 or dy < 1e-3:
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
        comp_edges: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
        for li in comp:
            _, _, b0, b1 = ends[li]
            node_deg[b0] = node_deg.get(b0, 0) + 1
            node_deg[b1] = node_deg.get(b1, 0) + 1
            comp_edges.append((int(li), b0, b1))

        end_nodes = [n for n, d in node_deg.items() if d == 1]
        ordered: List[int] = []
        start_node: Tuple[int, int]
        if len(end_nodes) != 2:
            # In real plans, door swing arcs (polylines) are frequently "attached" to other
            # short geometry (door leaf/wall stubs) near their endpoints. That creates a
            # branched component with != 2 endpoints. When enabled, recover a simple path
            # through the component that still fits a plausible arc.
            #
            # For cycle components with no degree-1 endpoints:
            # - reject loops to avoid label bubbles/circles
            if not allow_branches:
                continue
            if len(end_nodes) == 0:
                continue

            # Performance guardrails: only attempt recovery for small-ish components.
            # Highly-branched components create a combinatorial explosion of paths and
            # are rarely door swing arcs anyway.
            if len(comp) > 20:
                continue
            if len(end_nodes) > 6:
                continue

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
            max_paths = min(int(max_paths_per_component), 160)

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
                if not (min_radius_px <= radius <= max_radius_px):
                    return None
                if not _rmse_ok(rmse, radius):
                    return None
                if not (min_angle_deg <= angle_span <= max_angle_deg):
                    return None
                fit_score = max(0.0, 1.0 - (rmse / max(1e-6, max_circle_fit_rmse)))
                score = float(angle_span) * 1.0 + float(fit_score) * 50.0 + float(len(path_lines)) * 0.5
                return score, pts

            def _dfs(cur_node: Tuple[int, int], target: Tuple[int, int], used: set[int], path: List[int], path_start: Tuple[int, int]) -> None:
                nonlocal best_path, best_start, best_score, paths_tried
                if paths_tried >= max_paths:
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
                    _dfs(nxt_node, target, used, path, path_start)
                    path.pop()
                    used.remove(li)

            # Target nodes:
            # - Prefer other endpoints (when present) or junction-ish nodes (deg != 2).
            #   This helps the common case where one arc endpoint is attached to a wall/leaf stub,
            #   yielding only one true degree-1 endpoint in the short-segment component.
            start_candidates = list(end_nodes)
            for start in start_candidates:
                targets: List[Tuple[int, int]] = []
                if len(end_nodes) >= 2:
                    targets = [t for t in end_nodes if t != start]
                else:
                    targets = [t for t, d in node_deg.items() if t != start and int(d) != 2]
                    if not targets:
                        targets = [t for t in node_deg.keys() if t != start]

                # Cap targets to avoid pathological blow-ups in noisy drawings.
                if len(targets) > 24:
                    targets = targets[:24]

                for target in targets:
                    _dfs(start, target, set(), [], start)
                    if paths_tried >= max_paths:
                        break
                if paths_tried >= max_paths:
                    break

            if best_path is None or best_start is None:
                continue

            ordered = best_path
            start_node = best_start
        else:
            # Walk the chain in order from one endpoint.
            start_node = end_nodes[0]
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

        # First attempt: fit the full chain.
        try:
            center, radius, rmse = fit_circle(pts)
            radius = float(radius)
            rmse = float(rmse)
            angle_span = float(get_arc_angle_span(pts, center))
        except Exception:
            continue

        ok_full = bool((min_radius_px <= radius <= max_radius_px) and _rmse_ok(rmse, radius) and (min_angle_deg <= angle_span <= max_angle_deg))

        # If the full chain fails, try to recover a good sub-arc from a contiguous subpath.
        # This helps for components that concatenate two door arcs (double doors / double-acting doors),
        # or when a small amount of extra geometry is attached at one end.
        if not ok_full and len(ordered) >= max(min_segments + 1, 6):
            best = None  # (score, i0, j0, center, radius, rmse, angle_span)
            paths_tried = 0
            MAX_WINDOWS = 3200
            m = len(ordered)
            # Prefer longer windows first (more stable fits).
            for win_len in range(m, min_segments - 1, -1):
                if paths_tried >= MAX_WINDOWS:
                    break
                for i0 in range(0, m - win_len + 1):
                    j0 = i0 + win_len - 1
                    paths_tried += 1
                    if paths_tried >= MAX_WINDOWS:
                        break
                    pts_w = pts[i0 : j0 + 2]
                    if len(pts_w) < 3:
                        continue
                    try:
                        c2, r2, e2 = fit_circle(pts_w)
                        r2 = float(r2)
                        e2 = float(e2)
                        a2 = float(get_arc_angle_span(pts_w, c2))
                    except Exception:
                        continue
                    if not (min_radius_px <= r2 <= max_radius_px):
                        continue
                    if not _rmse_ok(e2, r2):
                        continue
                    if not (min_angle_deg <= a2 <= max_angle_deg):
                        continue
                    # Score: prefer ~90deg-ish arcs, larger angle spans, and better fits.
                    fit_score = max(0.0, 1.0 - (e2 / max(1e-6, float(max_circle_fit_rmse))))
                    score = (a2 * 1.0) + (fit_score * 40.0) + (float(win_len) * 0.35)
                    if best is None or score > best[0]:
                        best = (score, i0, j0, c2, r2, e2, a2)
                if best is not None and win_len >= max(min_segments + 2, 10):
                    # We found a reasonably long valid sub-arc; don't keep searching tiny windows.
                    break

            if best is not None:
                _, i0, j0, center, radius, rmse, angle_span = best
                ordered = ordered[i0 : j0 + 1]
                pts = pts[i0 : j0 + 2]
                ok_full = True

        if not ok_full:
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
    allow_dashed = bool(poly_conf.get("allow_dashed", False))
    allow_branches = bool(poly_conf.get("allow_branches", True))
    max_paths_per_component = int(poly_conf.get("max_paths_per_component", 200) or 200)

    # Fit/arc thresholds (same as swing.arc).
    max_circle_fit_rmse = float(arc_conf.get("max_circle_fit_rmse", 2.5) or 2.5)
    max_circle_fit_rmse_ratio = float(arc_conf.get("max_circle_fit_rmse_ratio", 0.0) or 0.0)
    min_radius_px = float(arc_conf.get("min_radius_px", 0.0) or 0.0)
    max_radius_px = float(arc_conf.get("max_radius_px", 1e9) or 1e9)
    min_angle_deg = float(arc_conf.get("min_angle_deg", 0.0) or 0.0)
    max_angle_deg = float(arc_conf.get("max_angle_deg", 1e9) or 1e9)

    def _rmse_ok(rmse: float, radius: float) -> bool:
        try:
            rmse_f = float(rmse)
            r_f = float(radius)
        except Exception:
            return False
        if rmse_f <= float(max_circle_fit_rmse):
            return True
        if max_circle_fit_rmse_ratio and max_circle_fit_rmse_ratio > 0 and r_f > 1e-6:
            return (rmse_f / r_f) <= float(max_circle_fit_rmse_ratio)
        return False

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
        if _is_dashed_primitive(ln) and not allow_dashed:
            continue
        try:
            p0 = (float(ln["p0"]["x"]), float(ln["p0"]["y"]))
            p1 = (float(ln["p1"]["x"]), float(ln["p1"]["y"]))
        except Exception:
            continue
        # Match `_extract_polyline_arcs_from_lines`: exclude perfectly axis-aligned segments.
        try:
            dx = abs(float(p1[0]) - float(p0[0]))
            dy = abs(float(p1[1]) - float(p0[1]))
        except Exception:
            dx, dy = 0.0, 0.0
        if dx < 1e-3 or dy < 1e-3:
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
            "allow_dashed": bool(allow_dashed),
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
            # fits a plausible arc.
            #
            # Note: loops (0 endpoints) are intentionally rejected to avoid label bubbles/circles.
            if not allow_branches or len(end_nodes) == 0:
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

            # Search for an arc-like path.
            #
            # - Normal branchy case: try paths between endpoint pairs.
            # - Loop-with-branch case (no endpoints): start from junction nodes (deg != 2) and
            #   try targets within the component (capped) — this recovers door arcs connected
            #   to a leaf line while still rejecting pure loops like circles.
            if len(end_nodes) >= 2:
                for i in range(len(end_nodes)):
                    for j in range(i + 1, len(end_nodes)):
                        path_start = end_nodes[i]
                        target = end_nodes[j]
                        _dfs(path_start, target, set(), [])
                        if paths_tried >= max_paths_per_component:
                            break
                    if paths_tried >= max_paths_per_component:
                        break
            else:
                starts = [n for n, d in node_deg.items() if int(d) != 2]
                if len(starts) > 8:
                    starts = starts[:8]
                for path_start in starts:
                    targets = [t for t, d in node_deg.items() if t != path_start and int(d) != 2]
                    if not targets:
                        targets = [t for t in node_deg.keys() if t != path_start]
                    if len(targets) > 24:
                        targets = targets[:24]
                    for target in targets:
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
        if not _rmse_ok(rmse, radius):
            fails.append("arc.rmse")
        if not (min_angle_deg <= angle_span <= max_angle_deg):
            fails.append("arc.angle_span")

        # Sub-arc recovery: if full path fails, attempt to find a contiguous subpath that passes.
        # This mirrors the non-debug extractor and helps diagnose (and fix) double-door / double-acting symbols.
        if fails and len(ordered) >= max(min_segments + 1, 6):
            best = None  # (score, i0, j0, center, radius, rmse, angle_span)
            paths_tried = 0
            MAX_WINDOWS = 3200
            m = len(ordered)
            # Reconstruct a points list aligned with ordered segments (same as extractor).
            pts_full = pts
            for win_len in range(m, min_segments - 1, -1):
                if paths_tried >= MAX_WINDOWS:
                    break
                for i0 in range(0, m - win_len + 1):
                    j0 = i0 + win_len - 1
                    paths_tried += 1
                    if paths_tried >= MAX_WINDOWS:
                        break
                    pts_w = pts_full[i0 : j0 + 2]
                    if len(pts_w) < 3:
                        continue
                    try:
                        c2, r2, e2 = fit_circle(pts_w)
                        r2 = float(r2)
                        e2 = float(e2)
                        a2 = float(get_arc_angle_span(pts_w, c2))
                    except Exception:
                        continue
                    if not (min_radius_px <= r2 <= max_radius_px):
                        continue
                    if not _rmse_ok(e2, r2):
                        continue
                    if not (min_angle_deg <= a2 <= max_angle_deg):
                        continue
                    fit_score = max(0.0, 1.0 - (e2 / max(1e-6, float(max_circle_fit_rmse))))
                    score = (a2 * 1.0) + (fit_score * 40.0) + (float(win_len) * 0.35)
                    if best is None or score > best[0]:
                        best = (score, i0, j0, c2, r2, e2, a2)
                if best is not None and win_len >= max(min_segments + 2, 10):
                    break

            if best is not None:
                _, i0, j0, c2, r2, e2, a2 = best
                ordered2 = ordered[i0 : j0 + 1]
                center = c2
                radius = float(r2)
                rmse = float(e2)
                angle_span = float(a2)
                fails = []
                if not (min_radius_px <= radius <= max_radius_px):
                    fails.append("arc.radius")
                if not _rmse_ok(rmse, radius):
                    fails.append("arc.rmse")
                if not (min_angle_deg <= angle_span <= max_angle_deg):
                    fails.append("arc.angle_span")
                if not fails:
                    ordered = ordered2

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
                    "max_rmse_ratio": float(max_circle_fit_rmse_ratio),
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
        "allow_dashed": bool(allow_dashed),
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
        # Mirror `_extract_polyline_arcs_from_lines`: allow relative RMSE tolerance for large-radius arcs.
        try:
            max_rmse = float(arc_conf.get("max_circle_fit_rmse", 2.5) or 2.5)
        except Exception:
            max_rmse = float(arc_conf["max_circle_fit_rmse"])
        try:
            max_rmse_ratio = float(arc_conf.get("max_circle_fit_rmse_ratio", 0.0) or 0.0)
        except Exception:
            max_rmse_ratio = 0.0
        rmse_ok = bool(rmse <= max_rmse) or bool(max_rmse_ratio > 0 and radius > 1e-6 and (rmse / radius) <= max_rmse_ratio)
        if not rmse_ok:
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


def detect_swing_leaf_only_candidates(
    *,
    lines: List[Dict[str, Any]],
    line_index: SpatialIndex,
    config: Dict[str, Any],
    line_indices: Optional[List[int]] = None,
    debug_out: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Leaf-only swing candidates for cases where the arc is missing/raster.

    Motivation:
    - Some “vector” PDFs still rasterize the curved swing arc while keeping the leaf line as a vector.
    - Our main swing detector is arc-first; without an arc primitive, no `swing` candidate can exist.
    - This helper proposes **candidate-only** `swing_leaf` boxes from diagonal leaf-like lines whose
      hinge endpoint is near a wall corner (axis-aligned linework).

    These candidates are intended for snapping / interactive labeling, not auto-selection.
    """
    swing_conf = (config.get("swing") or {}) if isinstance(config, dict) else {}
    leaf_only = (swing_conf.get("leaf_only") or {}) if isinstance(swing_conf, dict) else {}
    if not bool(leaf_only.get("enabled", False)):
        return []

    min_len = float(leaf_only.get("min_leaf_length_px", 40.0) or 40.0)
    max_len = float(leaf_only.get("max_leaf_length_px", 420.0) or 420.0)
    corner_probe = float(leaf_only.get("corner_probe_px", 8.0) or 8.0)
    corner_endpoint_snap = float(leaf_only.get("corner_endpoint_snap_px", corner_probe) or corner_probe)
    min_axis_support_len = float(leaf_only.get("min_axis_support_length_px", 40.0) or 40.0)
    # Some plans render thick wall edges as many short axis-aligned segments (often “dashed” visually).
    # To avoid missing hinge support in these cases, allow *thick* short segments to count as wall support.
    wall_support_min_stroke = float(leaf_only.get("wall_support_min_stroke_width", 1.0) or 1.0)
    wall_support_min_len = float(leaf_only.get("wall_support_min_segment_length_px", 12.0) or 12.0)
    axis_ratio = float(leaf_only.get("axis_alignment_ratio", 0.20) or 0.20)  # <=0.2 means "mostly axis-aligned"
    pad_frac = float(leaf_only.get("pad_frac_of_length", 0.22) or 0.22)
    max_out = int(leaf_only.get("max_candidates", 600) or 600)
    # When the hinge lies on the interior of a wall segment (not at a corner),
    # we still want a candidate, but we need extra guardrails to avoid flooding
    # the pool with diagonal annotation lines.
    min_tip_clearance_px = float(leaf_only.get("min_tip_clearance_px", 3.0) or 3.0)
    min_leaf_to_wall_angle_deg = float(leaf_only.get("min_leaf_to_wall_angle_deg", 18.0) or 18.0)

    snap_strict = float(corner_endpoint_snap)
    # Soft snap tolerates wall thickness / quantization / slight drafting offsets.
    # Use a larger neighborhood than the "corner probe" so we don't miss hinges that sit
    # on the interior of thick walls (common in drafted plans).
    snap_soft = max(float(corner_probe) * 2.0, float(corner_endpoint_snap) * 2.0, 12.0)

    def _is_axis_aligned(dx: float, dy: float) -> Tuple[bool, bool]:
        """Return (mostly_horizontal, mostly_vertical)."""
        adx = abs(dx)
        ady = abs(dy)
        if adx <= 1e-6 and ady <= 1e-6:
            return False, False
        mostly_h = ady <= axis_ratio * max(1e-6, adx)
        mostly_v = adx <= axis_ratio * max(1e-6, ady)
        return bool(mostly_h), bool(mostly_v)

    def _dist_point_to_segment(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance from p to segment ab."""
        px, py = float(p[0]), float(p[1])
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        c2 = vx * vx + vy * vy
        if c2 <= 1e-12:
            return float(dist_point_to_point((px, py), (ax, ay)))
        t = (wx * vx + wy * vy) / c2
        if t <= 0.0:
            return float(dist_point_to_point((px, py), (ax, ay)))
        if t >= 1.0:
            return float(dist_point_to_point((px, py), (bx, by)))
        proj = (ax + t * vx, ay + t * vy)
        return float(dist_point_to_point((px, py), proj))

    def _axis_support(pt: Tuple[float, float], *, skip_idx: int) -> Dict[str, Any]:
        """Return nearby axis-aligned support evidence for a hinge point.

        We treat this as a *candidate-pool* heuristic:
        - Strong (corner-like) evidence: both horizontal and vertical support nearby.
        - Weak (mid-wall) evidence: one axis support nearby, with extra guardrails later.
        """
        x, y = float(pt[0]), float(pt[1])
        probe = max(float(corner_probe), float(snap_soft))
        q = [x - probe, y - probe, x + probe, y + probe]
        neigh = line_index.query(q)
        # Aggregate support length (more robust than per-segment gates).
        # Some plans represent walls as many short segments (often thin), so requiring
        # a single long segment can miss real hinges.
        sum_h_len_soft = 0.0
        sum_v_len_soft = 0.0
        sum_h_len_strict = 0.0
        sum_v_len_strict = 0.0
        min_dist_any = float("inf")
        min_dist_support = float("inf")
        for j in neigh:
            if int(j) == int(skip_idx):
                continue
            ln = lines[int(j)]
            if _is_dashed_primitive(ln):
                continue
            p0 = (float(ln["p0"]["x"]), float(ln["p0"]["y"]))
            p1 = (float(ln["p1"]["x"]), float(ln["p1"]["y"]))
            L = float(dist_point_to_point(p0, p1))
            # Ignore extremely short segments; they are too noisy as "wall support".
            # We still allow many short-ish segments to accumulate into support evidence.
            if L < float(wall_support_min_len):
                continue
            # Weight thicker segments higher: thick wall edges are more reliable support
            # than thin annotation strokes.
            try:
                sw = float(ln.get("stroke_width") or 0.0)
            except Exception:
                sw = 0.0
            w = 1.0 if sw >= float(wall_support_min_stroke) else 0.5
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            is_h, is_v = _is_axis_aligned(dx, dy)
            if not (is_h or is_v):
                continue

            dseg = float(_dist_point_to_segment((x, y), p0, p1))
            min_dist_any = min(min_dist_any, dseg)
            # The hinge may land on the interior of a long wall segment (not just endpoints).
            # Use a "soft" snap distance to tolerate wall thickness and rounding.
            if dseg > snap_soft:
                continue
            min_dist_support = min(min_dist_support, dseg)

            if is_h:
                sum_h_len_soft += w * L
                if dseg <= snap_strict:
                    sum_h_len_strict += w * L
            if is_v:
                sum_v_len_soft += w * L
                if dseg <= snap_strict:
                    sum_v_len_strict += w * L

            # Early exit once we have strong support in both directions.
            if sum_h_len_strict >= float(min_axis_support_len) and sum_v_len_strict >= float(min_axis_support_len):
                break

        has_h = bool(sum_h_len_soft >= float(min_axis_support_len))
        has_v = bool(sum_v_len_soft >= float(min_axis_support_len))
        has_h_strict = bool(sum_h_len_strict >= float(min_axis_support_len))
        has_v_strict = bool(sum_v_len_strict >= float(min_axis_support_len))

        score = int(has_h) + int(has_v)
        strength = "none"
        if score > 0:
            strength = "strict" if (has_h_strict or has_v_strict) else "soft"
        return {
            "score": int(score),
            "strength": str(strength),
            "has_h": bool(has_h),
            "has_v": bool(has_v),
            "has_h_strict": bool(has_h_strict),
            "has_v_strict": bool(has_v_strict),
            "sum_h_len_soft": float(sum_h_len_soft),
            "sum_v_len_soft": float(sum_v_len_soft),
            "sum_h_len_strict": float(sum_h_len_strict),
            "sum_v_len_strict": float(sum_v_len_strict),
            "min_dist_any": float(min_dist_any) if math.isfinite(min_dist_any) else None,
            "min_dist_support": float(min_dist_support) if math.isfinite(min_dist_support) else None,
        }

    out: List[Dict[str, Any]] = []
    # Optional debug counters (for unmatched debug reports).
    dbg_counts: Dict[str, int] = {
        "scanned_lines": 0,
        "skipped_dashed": 0,
        "skipped_len": 0,
        "skipped_axis_aligned": 0,
        "skipped_no_support": 0,
        "skipped_midwall_tip_clearance": 0,
        "skipped_midwall_angle_to_wall": 0,
        "candidates_emitted": 0,
    }
    dbg_examples: Dict[str, List[Dict[str, Any]]] = {"no_support": [], "midwall_tip_clearance": [], "midwall_angle_to_wall": []}
    scan = line_indices if isinstance(line_indices, list) else list(range(len(lines)))
    for l_idx in scan:
        dbg_counts["scanned_lines"] += 1
        try:
            ln = lines[int(l_idx)]
        except Exception:
            continue
        try:
            if _is_dashed_primitive(ln):
                dbg_counts["skipped_dashed"] += 1
                continue
            p0 = (float(ln["p0"]["x"]), float(ln["p0"]["y"]))
            p1 = (float(ln["p1"]["x"]), float(ln["p1"]["y"]))
        except Exception:
            continue
        L = float(dist_point_to_point(p0, p1))
        if not (min_len <= L <= max_len):
            dbg_counts["skipped_len"] += 1
            continue
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        # Leaf line should be noticeably non-axis-aligned.
        is_h, is_v = _is_axis_aligned(dx, dy)
        if is_h or is_v:
            dbg_counts["skipped_axis_aligned"] += 1
            continue

        sup0 = _axis_support(p0, skip_idx=l_idx)
        sup1 = _axis_support(p1, skip_idx=l_idx)
        s0 = int(sup0.get("score") or 0)
        s1 = int(sup1.get("score") or 0)
        if s0 <= 0 and s1 <= 0:
            dbg_counts["skipped_no_support"] += 1
            try:
                if len(dbg_examples["no_support"]) < 6:
                    dbg_examples["no_support"].append(
                        {
                            "l_idx": int(l_idx),
                            "leaf_len_px": float(L),
                            "p0": [float(p0[0]), float(p0[1])],
                            "p1": [float(p1[0]), float(p1[1])],
                            "sup0": sup0,
                            "sup1": sup1,
                        }
                    )
            except Exception:
                pass
            continue

        # Prefer a hinge endpoint with stronger axis support. Tie-break by proximity to support.
        d0 = float(sup0.get("min_dist_support") or 1e18)
        d1 = float(sup1.get("min_dist_support") or 1e18)
        if s0 > s1 or (s0 == s1 and d0 <= d1):
            hinge = p0
            tip = p1
            hinge_sup = sup0
            tip_sup = sup1
        else:
            hinge = p1
            tip = p0
            hinge_sup = sup1
            tip_sup = sup0
        corner_score = int(hinge_sup.get("score") or 0)
        support_strength = str(hinge_sup.get("strength") or "none")

        # Guardrails for mid-wall hinges: allow score==1 (single axis-aligned wall support),
        # but require the leaf to "stick out" away from wall geometry and not be near-parallel
        # to the supporting wall direction.
        if corner_score < 2:
            try:
                hinge_any = hinge_sup.get("min_dist_any")
                hinge_any_f = float(hinge_any) if hinge_any is not None else float("inf")
            except Exception:
                hinge_any_f = float("inf")
            try:
                tip_any = tip_sup.get("min_dist_any")
                tip_any_f = float(tip_any) if tip_any is not None else float("inf")
            except Exception:
                tip_any_f = float("inf")
            if not (tip_any_f >= hinge_any_f + float(min_tip_clearance_px)):
                dbg_counts["skipped_midwall_tip_clearance"] += 1
                try:
                    if len(dbg_examples["midwall_tip_clearance"]) < 6:
                        dbg_examples["midwall_tip_clearance"].append(
                            {
                                "l_idx": int(l_idx),
                                "leaf_len_px": float(L),
                                "hinge_any_dist": float(hinge_any_f) if math.isfinite(hinge_any_f) else None,
                                "tip_any_dist": float(tip_any_f) if math.isfinite(tip_any_f) else None,
                                "min_tip_clearance_px": float(min_tip_clearance_px),
                                "hinge_sup": hinge_sup,
                                "tip_sup": tip_sup,
                            }
                        )
                except Exception:
                    pass
                continue

            # Angle-to-wall gating.
            ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            has_h_wall = bool(hinge_sup.get("has_h"))
            has_v_wall = bool(hinge_sup.get("has_v"))
            if has_h_wall and not has_v_wall:
                # Horizontal wall: leaf must not be near-horizontal.
                parallel_err = min(ang, 180.0 - ang)
                if parallel_err < float(min_leaf_to_wall_angle_deg):
                    dbg_counts["skipped_midwall_angle_to_wall"] += 1
                    try:
                        if len(dbg_examples["midwall_angle_to_wall"]) < 6:
                            dbg_examples["midwall_angle_to_wall"].append(
                                {
                                    "l_idx": int(l_idx),
                                    "leaf_len_px": float(L),
                                    "wall_axis": "horizontal",
                                    "leaf_angle_deg": float(ang),
                                    "parallel_err_deg": float(parallel_err),
                                    "min_leaf_to_wall_angle_deg": float(min_leaf_to_wall_angle_deg),
                                    "hinge_sup": hinge_sup,
                                }
                            )
                    except Exception:
                        pass
                    continue
            elif has_v_wall and not has_h_wall:
                # Vertical wall: leaf must not be near-vertical.
                parallel_err = abs(ang - 90.0)
                if parallel_err < float(min_leaf_to_wall_angle_deg):
                    dbg_counts["skipped_midwall_angle_to_wall"] += 1
                    try:
                        if len(dbg_examples["midwall_angle_to_wall"]) < 6:
                            dbg_examples["midwall_angle_to_wall"].append(
                                {
                                    "l_idx": int(l_idx),
                                    "leaf_len_px": float(L),
                                    "wall_axis": "vertical",
                                    "leaf_angle_deg": float(ang),
                                    "parallel_err_deg": float(parallel_err),
                                    "min_leaf_to_wall_angle_deg": float(min_leaf_to_wall_angle_deg),
                                    "hinge_sup": hinge_sup,
                                }
                            )
                    except Exception:
                        pass
                    continue
            # If both wall directions are present, treat as a corner and skip angle gating.

        pad = max(6.0, pad_frac * L)
        bbox = [min(p0[0], p1[0]) - pad, min(p0[1], p1[1]) - pad, max(p0[0], p1[0]) + pad, max(p0[1], p1[1]) + pad]
        cid = _stable_line_candidate_id(cand_type="swing_leaf", p0=p0, p1=p1, bbox_xyxy=bbox, quant_step_px=1.0)

        # Conservative heuristic confidence (kept low; intended for UI snapping only).
        # - Corner support gets a small boost.
        # - Strict distance gets a small boost (soft support can be noisier).
        conf_h = 0.16 + (0.08 if corner_score >= 2 else 0.02) + (0.04 if support_strength == "strict" else 0.0)
        conf_h = max(0.0, min(0.32, float(conf_h)))

        out.append(
            {
                "id": cid,
                "legacy_ids": [],
                "type": "swing_leaf",
                "bbox_xyxy": bbox,
                "geom": {"hinge_xy": [float(hinge[0]), float(hinge[1])], "tip_xy": [float(tip[0]), float(tip[1])]},
                "heuristic_confidence": float(conf_h),
                "confidence": float(conf_h),
                "pool": True,
                "features": {
                    "leaf_only": 1.0,
                    "leaf_length_px": float(L),
                    "corner_support": float(corner_score),
                    "axis_support_strength": 1.0 if support_strength == "strict" else 0.5 if support_strength == "soft" else 0.0,
                },
                "primitives": {"lines": [int(l_idx)]},
            }
        )
        dbg_counts["candidates_emitted"] += 1

    out.sort(key=lambda c: float(c.get("confidence", 0.0) or 0.0), reverse=True)
    out2 = out[: max(0, max_out)]
    if isinstance(debug_out, dict):
        try:
            debug_out.clear()
            debug_out.update(
                {
                    "enabled": True,
                    "params": {
                        "min_leaf_length_px": float(min_len),
                        "max_leaf_length_px": float(max_len),
                        "corner_probe_px": float(corner_probe),
                        "corner_endpoint_snap_px": float(corner_endpoint_snap),
                        "min_axis_support_length_px": float(min_axis_support_len),
                        "wall_support_min_stroke_width": float(wall_support_min_stroke),
                        "wall_support_min_segment_length_px": float(wall_support_min_len),
                        "axis_alignment_ratio": float(axis_ratio),
                        "pad_frac_of_length": float(pad_frac),
                        "max_candidates": int(max_out),
                        "min_tip_clearance_px": float(min_tip_clearance_px),
                        "min_leaf_to_wall_angle_deg": float(min_leaf_to_wall_angle_deg),
                    },
                    "counts": {k: int(v) for k, v in dbg_counts.items()},
                    "examples": dbg_examples,
                    "emitted_sample": [
                        {"id": str(c.get("id") or ""), "bbox_xyxy": c.get("bbox_xyxy"), "features": c.get("features")}
                        for c in out2[:5]
                        if isinstance(c, dict)
                    ],
                }
            )
        except Exception:
            pass
    return out2


def detect_double_candidates(*, swing_candidates: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Propose double-door candidates by pairing swing candidates."""
    conf = (config.get("double") or {}) if isinstance(config, dict) else {}
    pairing = conf.get("pairing") or {}
    scoring = conf.get("scoring") or {}

    max_center_dist_ratio = float(pairing.get("max_center_dist_ratio", 3.0) or 3.0)
    max_radius_ratio = float(pairing.get("max_radius_ratio", 1.35) or 1.35)
    max_bbox_iou = float(pairing.get("max_bbox_iou", 0.15) or 0.15)
    max_pairs = int(pairing.get("max_pairs", 600) or 600)
    # Optional volume-control: require both component swings to be reasonably confident
    # before pairing. When a per-type reweighter is present, it is best to apply this
    # threshold *after* swing reweighting (see detect_doors pipeline ordering).
    min_component_conf = float(pairing.get("min_component_confidence", 0.0) or 0.0)
    # Double-acting doors: allow pairing of two swing candidates that share the same hinge
    # and leaf tip, but have distinct arc directions (door swings both ways).
    double_acting = pairing.get("double_acting") if isinstance(pairing, dict) else None
    acting_enabled = True
    acting_conf = double_acting if isinstance(double_acting, dict) else {}
    if isinstance(double_acting, dict) and ("enabled" in double_acting):
        acting_enabled = bool(double_acting.get("enabled"))
    max_same_hinge_ratio = float(acting_conf.get("max_same_hinge_dist_ratio", 0.10) or 0.10)
    max_same_tip_ratio = float(acting_conf.get("max_same_tip_dist_ratio", 0.10) or 0.10)
    min_arc_dir_sep_deg = float(acting_conf.get("min_arc_dir_separation_deg", 25.0) or 25.0)

    w_pair = float(scoring.get("w_pair", 0.40) or 0.40)
    w_avg_conf = float(scoring.get("w_avg_conf", 0.60) or 0.60)

    # IMPORTANT:
    # Historically this stage was intentionally permissive: it created "double" *candidates*
    # for many nearby swing pairs and relied on a per-type reweighter to downrank false pairs.
    #
    # In practice, the double reweighter file may be missing (common in local/dev setups),
    # which makes the heuristic score the source of truth for final selection. That can
    # cause false positives: two nearby but unrelated swings get promoted to "double".
    #
    # So we keep candidate generation broad (to preserve snapping/debugging + test coverage),
    # but make confidence depend strongly on "double-likeness" geometry (hinge/tip alignment),
    # so unrelated nearby swings produce low-confidence double candidates.

    out: List[Dict[str, Any]] = []
    n = len(swing_candidates)
    if n <= 1:
        return out

    # Consider high-confidence swing candidates first to keep pairing cost reasonable.
    swings = list(swing_candidates)
    swings.sort(key=lambda c: float(c.get("confidence", 0.0) or 0.0), reverse=True)
    swings = swings[: min(len(swings), 500)]

    def _pt(v: Any) -> Optional[Tuple[float, float]]:
        try:
            if not (isinstance(v, list) and len(v) == 2):
                return None
            x, y = float(v[0]), float(v[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            return (x, y)
        except Exception:
            return None

    def _unit(vx: float, vy: float) -> Optional[Tuple[float, float]]:
        try:
            n2 = float(vx * vx + vy * vy)
            if not (n2 > 1e-9):
                return None
            n = math.sqrt(n2)
            return (vx / n, vy / n)
        except Exception:
            return None

    for i in range(len(swings)):
        a = swings[i]
        a_id = a.get("id")
        if a_id is None:
            continue
        try:
            a_conf = float(a.get("confidence", a.get("heuristic_confidence", 0.0)) or 0.0)
        except Exception:
            a_conf = 0.0
        a_geom = (a.get("geom") or {}) if isinstance(a.get("geom"), dict) else {}
        bba = _normalize_bbox_xyxy(a.get("bbox_xyxy"))
        if bba is None:
            continue
        try:
            ca = a_geom.get("center_xy") or []
            ax, ay = float(ca[0]), float(ca[1])
        except Exception:
            continue
        ha = _pt(a_geom.get("hinge_xy")) or (ax, ay)
        ta = _pt(a_geom.get("tip_xy"))
        ra = float((a.get("features") or {}).get("radius", 0.0) or 0.0)
        if not (ra > 0):
            continue

        for j in range(i + 1, len(swings)):
            b = swings[j]
            b_id = b.get("id")
            if b_id is None:
                continue
            try:
                b_conf = float(b.get("confidence", b.get("heuristic_confidence", 0.0)) or 0.0)
            except Exception:
                b_conf = 0.0
            b_geom = (b.get("geom") or {}) if isinstance(b.get("geom"), dict) else {}
            bbb = _normalize_bbox_xyxy(b.get("bbox_xyxy"))
            if bbb is None:
                continue
            try:
                cb = b_geom.get("center_xy") or []
                bx, by = float(cb[0]), float(cb[1])
            except Exception:
                continue
            hb = _pt(b_geom.get("hinge_xy")) or (bx, by)
            tb = _pt(b_geom.get("tip_xy"))
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

            bbox_iou = float(compute_iou(bba, bbb))
            if bbox_iou > max_bbox_iou:
                # Normally we skip high-overlap swing pairs as duplicates. However, a common
                # door symbol is "double-acting" (one leaf swings both directions), which
                # yields two swing candidates with the *same* hinge+tip but different arcs.
                if not acting_enabled:
                    continue
                # Require hinge+tip consistency (same leaf).
                try:
                    hinge_dist = float(math.hypot(ha[0] - hb[0], ha[1] - hb[1]))
                except Exception:
                    hinge_dist = float("inf")
                tip_dist = None
                if ta is not None and tb is not None:
                    try:
                        tip_dist = float(math.hypot(ta[0] - tb[0], ta[1] - tb[1]))
                    except Exception:
                        tip_dist = None
                if not (math.isfinite(hinge_dist) and hinge_dist <= max_same_hinge_ratio * float(rmax)):
                    continue
                if tip_dist is None or not (tip_dist <= max_same_tip_ratio * float(rmax)):
                    continue

                # Require distinct arc directions (avoid pairing true duplicates).
                try:
                    a_eps = (a_geom.get("arc_endpoints_xy") or []) if isinstance(a_geom, dict) else []
                    b_eps = (b_geom.get("arc_endpoints_xy") or []) if isinstance(b_geom, dict) else []
                    if not (isinstance(a_eps, list) and len(a_eps) == 2 and isinstance(b_eps, list) and len(b_eps) == 2):
                        continue
                    acx, acy = float(ax), float(ay)
                    # Average endpoint directions (unit vectors) to get a coarse arc "mid direction".
                    def _arc_dir(endpoints):
                        (x0, y0), (x1, y1) = endpoints
                        u0 = _unit(float(x0) - acx, float(y0) - acy)
                        u1 = _unit(float(x1) - acx, float(y1) - acy)
                        if u0 is None or u1 is None:
                            return None
                        ux, uy = (u0[0] + u1[0], u0[1] + u1[1])
                        u = _unit(ux, uy)
                        return u
                    uda = _arc_dir(a_eps)
                    udb = _arc_dir(b_eps)
                    if uda is None or udb is None:
                        continue
                    dot = max(-1.0, min(1.0, float(uda[0] * udb[0] + uda[1] * udb[1])))
                    sep_deg = float(math.degrees(math.acos(dot)))
                except Exception:
                    continue
                if not (sep_deg >= min_arc_dir_sep_deg):
                    continue

                # Build a "double" candidate representing a double-acting door.
                union = _bbox_union(bba, bbb)
                avg_conf = 0.5 * (float(a_conf) + float(b_conf))
                # Score from arc separation + tip/hinge agreement.
                sep_score = max(0.0, min(1.0, (sep_deg - min_arc_dir_sep_deg) / max(1e-6, (90.0 - min_arc_dir_sep_deg))))
                tip_score = 1.0 - (float(tip_dist) / max(1e-6, max_same_tip_ratio * float(rmax)))
                tip_score = max(0.0, min(1.0, float(tip_score)))
                hinge_score = 1.0 - (float(hinge_dist) / max(1e-6, max_same_hinge_ratio * float(rmax)))
                hinge_score = max(0.0, min(1.0, float(hinge_score)))
                pair_score = max(0.0, min(1.0, 0.55 * float(sep_score) + 0.25 * float(tip_score) + 0.20 * float(hinge_score)))
                conf_pair = (w_pair * pair_score + w_avg_conf * avg_conf)
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
                            "double_acting": 1.0,
                            "center_dist": float(center_dist),
                            "radius_ratio": float(rmax / rmin),
                            "avg_swing_conf": float(avg_conf),
                            "pair_score": float(pair_score),
                            "bbox_iou": float(bbox_iou),
                            "hinge_dist": float(hinge_dist),
                            "tip_dist": float(tip_dist),
                            "arc_dir_sep_deg": float(sep_deg),
                        },
                    }
                )
                continue

            union = _bbox_union(bba, bbb)

            # --- "double-likeness" geometry scoring ---
            #
            # Two unrelated nearby doors often pass the center/radius gates above.
            # Double doors, however, have an extra structure:
            # - each leaf's hinge-to-tip direction points toward the *other* hinge (meeting stile)
            # - the two tips are close to each other (meet in the middle)
            #
            # We compute a bounded score in [0,1] and later use it to *gate* confidence
            # (multiply), so high-confidence swings cannot produce a high-confidence double
            # unless geometry is double-like.
            hinge_dist = float(math.hypot(ha[0] - hb[0], ha[1] - hb[1]))
            sum_r = float(ra + rb)
            hinge_ratio = (hinge_dist / max(1e-6, sum_r)) if sum_r > 0 else 0.0
            # Prefer hinge_dist ~= (ra+rb), tolerate moderate mismatch.
            hinge_tol = 0.40
            hinge_sep_score = 1.0 - (abs(hinge_ratio - 1.0) / max(1e-6, hinge_tol))
            hinge_sep_score = max(0.0, min(1.0, float(hinge_sep_score)))

            # Leaf direction alignment: va should point from ha toward hb, vb from hb toward ha.
            ux = hb[0] - ha[0]
            uy = hb[1] - ha[1]
            uu = _unit(ux, uy)
            axis_score = 0.0
            cos_a = 0.0
            cos_b = 0.0
            if uu is not None and ta is not None and tb is not None:
                uux, uuy = uu
                vax, vay = (ta[0] - ha[0]), (ta[1] - ha[1])
                vbx, vby = (tb[0] - hb[0]), (tb[1] - hb[1])
                va_u = _unit(vax, vay)
                vb_u = _unit(vbx, vby)
                if va_u is not None and vb_u is not None:
                    vaxu, vayu = va_u
                    vbxu, vbyu = vb_u
                    # Clamp to [0,1] by requiring "toward the other hinge".
                    cos_a = max(0.0, min(1.0, float(vaxu * uux + vayu * uuy)))
                    cos_b = max(0.0, min(1.0, float(vbxu * (-uux) + vbyu * (-uuy))))
                    axis_score = 0.5 * (cos_a + cos_b)
            axis_score = max(0.0, min(1.0, float(axis_score)))

            # Tips should be near each other relative to hinge separation (meeting stile).
            tip_score = 0.0
            tip_dist = None
            tip_ratio = None
            if ta is not None and tb is not None:
                tip_dist = float(math.hypot(ta[0] - tb[0], ta[1] - tb[1]))
                tip_ratio = float(tip_dist / max(1e-6, hinge_dist))
                # If tips are within ~40% of hinge separation, score grows to 1.
                tip_meet_max = 0.40
                tip_score = 1.0 - (float(tip_ratio) / max(1e-6, tip_meet_max))
                tip_score = max(0.0, min(1.0, float(tip_score)))

            # Coarse proximity still matters (keeps very far pairs low even if geometry is noisy).
            prox_score = 1.0 - (center_dist / max(1e-6, (max_center_dist_ratio * rmax)))
            prox_score = max(0.0, min(1.0, float(prox_score)))

            # Composite double-likeness score.
            pair_score = 0.45 * float(axis_score) + 0.35 * float(tip_score) + 0.15 * float(hinge_sep_score) + 0.05 * float(prox_score)
            pair_score = max(0.0, min(1.0, float(pair_score)))
            avg_conf = 0.5 * (float(a.get("confidence", 0.0) or 0.0) + float(b.get("confidence", 0.0) or 0.0))
            # Optional: require both component swings to be reasonably confident before
            # creating a double-leaf candidate (volume control). This is intentionally
            # NOT applied to double-acting pairing above.
            if min_component_conf > 0.0 and (float(a_conf) < min_component_conf or float(b_conf) < min_component_conf):
                continue
            conf_pair = (w_pair * pair_score + w_avg_conf * avg_conf) * pair_score
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
                        "hinge_dist": float(hinge_dist),
                        "hinge_dist_over_sum_r": float(hinge_ratio),
                        "leaf_axis_cos_a": float(cos_a),
                        "leaf_axis_cos_b": float(cos_b),
                        "axis_score": float(axis_score),
                        "tip_dist": float(tip_dist) if tip_dist is not None else 0.0,
                        "tip_dist_over_hinge_dist": float(tip_ratio) if tip_ratio is not None else 0.0,
                        "tip_score": float(tip_score),
                        "prox_score": float(prox_score),
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

    # Output thresholds (used for final selection and candidate export).
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
    dedupe_conf = out_conf.get("dedupe") if isinstance(out_conf.get("dedupe"), dict) else {}
    dedupe_enabled = bool(dedupe_conf.get("enabled", False)) if isinstance(dedupe_conf, dict) else False
    dedupe_debug = bool(dedupe_conf.get("debug_log", False)) or _truthy_env("DOOR_DETECTOR_DEDUPE_DEBUG")
    dedupe_debug_blob: Dict[str, Any] = {
        "dedupe_enabled": bool(dedupe_enabled),
        "dedupe_debug": bool(dedupe_debug),
        "pool": {},
        "final": {},
        "remaining_overlaps": [],
        "note": "Enable output.dedupe.debug_log=true (or env DOOR_DETECTOR_DEDUPE_DEBUG=1) for detailed pair dumps.",
    }

    _dedupe_debug_log(
        dedupe_debug,
        "dedupe.config",
        file_id=str(meta.get("id") or ""),
        mode=str(meta.get("mode") or ""),
        dedupe_enabled=bool(dedupe_enabled),
        dedupe_conf={k: v for k, v in (dedupe_conf or {}).items() if k not in ("swing", "swing_arc")},
        swing_conf=(dedupe_conf.get("swing") if isinstance(dedupe_conf.get("swing"), dict) else {}),
        swing_arc_conf=(dedupe_conf.get("swing_arc") if isinstance(dedupe_conf.get("swing_arc"), dict) else {}),
    )

    # Per-type reweighters (preferred) + backward compatibility with single `reweighter_path`.
    #
    # IMPORTANT ordering detail:
    # The double reweighter frequently uses `avg_swing_conf` as a key feature.
    # So we resolve models early and apply the swing reweighter *before* generating
    # double candidates, ensuring double features are computed from calibrated swings.
    base_dirs = _door_detector_base_dirs_from_config(config)
    model_paths_by_type: Dict[str, str] = {}
    cfg_reweighters = config.get("reweighters")
    if isinstance(cfg_reweighters, dict):
        for k, v in cfg_reweighters.items():
            if not (isinstance(v, str) and v.strip()):
                continue
            t = str(k).strip().lower()
            resolved = _resolve_existing_path(v, base_dirs=base_dirs)
            if resolved:
                model_paths_by_type[t] = resolved
    legacy_model_path = config.get("reweighter_path")
    if isinstance(legacy_model_path, str) and legacy_model_path.strip():
        resolved = _resolve_existing_path(legacy_model_path, base_dirs=base_dirs)
        if resolved:
            model_paths_by_type.setdefault("swing", resolved)
    if not model_paths_by_type:
        for t in ("swing", "double", "pocket", "bifold"):
            resolved = _resolve_existing_path(f"models/reweighter_{t}_v1.json", base_dirs=base_dirs)
            if resolved:
                model_paths_by_type[t] = resolved
        legacy = _resolve_existing_path("models/reweighter_v1.json", base_dirs=base_dirs)
        if legacy:
            model_paths_by_type.setdefault("swing", legacy)
    has_any_model = bool(model_paths_by_type)

    strict_candidates_all: List[Dict[str, Any]] = []
    candidate_pool_all: List[Dict[str, Any]] = []

    # --- Swing ---
    # Run swing detection whenever enabled and we have *either* beziers or lines.
    # This is important because many PDFs approximate arcs as polylines (line chains).
    if bool((config.get("swing") or {}).get("enabled")) and (beziers or lines):
        strict_swing, pool_swing = detect_swing_candidates(lines=lines, beziers=beziers, line_index=line_index, config=config)
        # If a swing reweighter exists, apply it now so downstream pairing logic
        # (especially double pairing) uses calibrated swing confidences.
        if has_any_model and model_paths_by_type.get("swing"):
            try:
                apply_reweighter(strict_swing, model_paths_by_type["swing"])
            except Exception:
                pass
            try:
                apply_reweighter(pool_swing, model_paths_by_type["swing"])
            except Exception:
                pass
        strict_candidates_all.extend(strict_swing)
        candidate_pool_all.extend(pool_swing)
        # Optional: leaf-only swing candidates (for cases where the leaf is vector but the arc is missing/raster).
        try:
            swing_conf = (config.get("swing") or {}) if isinstance(config, dict) else {}
            leaf_only_conf = (swing_conf.get("leaf_only") or {}) if isinstance(swing_conf, dict) else {}
            if bool(leaf_only_conf.get("enabled", False)):
                candidate_pool_all.extend(
                    detect_swing_leaf_only_candidates(lines=lines, line_index=line_index, config=config)
                )
        except Exception:
            pass

    # --- Double (pair swing candidates) ---
    if bool((config.get("double") or {}).get("enabled")) and candidate_pool_all:
        double_cands = detect_double_candidates(swing_candidates=[c for c in candidate_pool_all if c.get("type") == "swing"], config=config)
        # Apply the double reweighter immediately so candidate export + final selection
        # operate on calibrated "double" confidence.
        if has_any_model and model_paths_by_type.get("double"):
            try:
                apply_reweighter(double_cands, model_paths_by_type["double"])
            except Exception:
                pass
        # For now, treat these as both pool + strict (no strict-vs-pool distinction).
        strict_candidates_all.extend(double_cands)
        candidate_pool_all.extend(double_cands)

    # --- Pocket (dashed tracks) ---
    if bool((config.get("pocket") or {}).get("enabled")) and lines:
        pocket_cands = detect_pocket_candidates(lines=lines, config=config)
        if has_any_model and model_paths_by_type.get("pocket"):
            try:
                apply_reweighter(pocket_cands, model_paths_by_type["pocket"])
            except Exception:
                pass
        strict_candidates_all.extend(pocket_cands)
        candidate_pool_all.extend(pocket_cands)

    # --- Bi-fold (zig-zag chains) ---
    if bool((config.get("bifold") or {}).get("enabled")) and lines:
        bifold_cands = detect_bifold_candidates(lines=lines, config=config)
        if has_any_model and model_paths_by_type.get("bifold"):
            try:
                apply_reweighter(bifold_cands, model_paths_by_type["bifold"])
            except Exception:
                pass
        strict_candidates_all.extend(bifold_cands)
        candidate_pool_all.extend(bifold_cands)

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

    # Deduplicate by near-duplicate geometry (IoU + containment + keypoints) if enabled.
    # This reduces candidate crowding for snapping/review without collapsing adjacent doors.
    if dedupe_enabled and candidate_pool_all:
        pool_before = int(len(candidate_pool_all))
        try:
            candidate_pool_all, _dup_map = suppress_duplicates(candidate_pool_all, dedupe_conf)
            dedupe_debug_blob["pool"] = {
                "before": pool_before,
                "after": int(len(candidate_pool_all)),
                "suppressed": int(max(0, pool_before - int(len(candidate_pool_all)))),
            }
            _dedupe_debug_log(
                dedupe_debug,
                "dedupe.pool",
                file_id=str(meta.get("id") or ""),
                before=pool_before,
                after=int(len(candidate_pool_all)),
                suppressed=int(max(0, pool_before - int(len(candidate_pool_all)))),
                dup_map_sample=dict(list(_dup_map.items())[:10]) if isinstance(_dup_map, dict) else {},
            )
        except Exception:
            # Safety: never fail detection due to dedupe bugs.
            _dedupe_debug_log(dedupe_debug, "dedupe.pool.error", file_id=str(meta.get("id") or ""))
            pass

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
    if not kept and has_any_model:
        # Safety fallback:
        # If a model exists but would drop *everything*, fall back to the conservative
        # strict selection so we don't regress to "no doors" due to an untrained /
        # miscalibrated / out-of-domain reweighter.
        #
        # This does NOT mask reweighter-path resolution regressions in tests because
        # those cases generally have no strict candidates to fall back to.
        strict_src_fb = [c for c in strict_candidates_all if str(c.get("type") or "").strip().lower() in allowed_final_types]
        strict_fb = [
            c
            for c in strict_src_fb
            if float(c.get("heuristic_confidence", c.get("confidence", 0.0)) or 0.0) >= min_candidate_conf
        ]
        strict_fb.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)
        kept = [c for c in strict_fb if float(c.get("heuristic_confidence", 0.0) or 0.0) >= legacy_min_conf]
    if not kept:
        return {"doors": [], "candidates": exported_candidates}

    # Final duplicate suppression:
    # Replace IoU-only NMS with a stricter "same-door" duplicate predicate (IoU + containment
    # + type-specific keypoints). This avoids collapsing adjacent doors while still removing
    # bbox/geometry variants of the same door.
    if dedupe_enabled:
        try:
            kept_before = int(len(kept))
            final, _dup_map = suppress_duplicates(kept, dedupe_conf)
            dedupe_debug_blob["final"] = {
                "before": kept_before,
                "after": int(len(final)),
                "suppressed": int(max(0, kept_before - int(len(final)))),
            }
            _dedupe_debug_log(
                dedupe_debug,
                "dedupe.final",
                file_id=str(meta.get("id") or ""),
                before=kept_before,
                after=int(len(final)),
                suppressed=int(max(0, kept_before - int(len(final)))),
                dup_map_sample=dict(list(_dup_map.items())[:10]) if isinstance(_dup_map, dict) else {},
            )
        except Exception:
            final = list(kept)
            _dedupe_debug_log(dedupe_debug, "dedupe.final.error", file_id=str(meta.get("id") or ""))
        final = final[: max(0, max_doors)]
    else:
        # Backward-compat fallback: IoU-only NMS (older configs/tests may rely on this).
        final = []
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

    # Post-check: report any *remaining* high-overlap pairs in final doors.
    # This helps diagnose why duplicates still appear.
    try:
        debug_iou = float(dedupe_conf.get("debug_overlap_min_iou", 0.70) or 0.70)
        debug_contain = float(dedupe_conf.get("debug_overlap_min_contain", 0.92) or 0.92)
        debug_max_pairs = int(dedupe_conf.get("debug_overlap_max_pairs", 12) or 12)
    except Exception:
        debug_iou, debug_contain, debug_max_pairs = 0.70, 0.92, 12

    overlaps: List[Dict[str, Any]] = []
    if final and (dedupe_debug or dedupe_enabled):
        for i in range(len(final)):
            a = final[i]
            bba = a.get("bbox_xyxy")
            if not (isinstance(bba, list) and len(bba) == 4):
                continue
            for j in range(i + 1, len(final)):
                b = final[j]
                bbb = b.get("bbox_xyxy")
                if not (isinstance(bbb, list) and len(bbb) == 4):
                    continue
                iou = float(compute_iou(bba, bbb))
                if iou < debug_iou:
                    # containment can catch big/small bbox variants
                    try:
                        contain = float(bbox_containment(bba, bbb))
                    except Exception:
                        contain = 0.0
                    if contain < debug_contain:
                        continue
                else:
                    try:
                        contain = float(bbox_containment(bba, bbb))
                    except Exception:
                        contain = 0.0
                overlaps.append(
                    {
                        "iou": float(iou),
                        "contain": float(contain),
                        "a": {"id": str(a.get("id") or ""), "type": str(a.get("type") or ""), "conf": float(a.get("confidence", 0.0) or 0.0)},
                        "b": {"id": str(b.get("id") or ""), "type": str(b.get("type") or ""), "conf": float(b.get("confidence", 0.0) or 0.0)},
                    }
                )
                if len(overlaps) >= max(50, debug_max_pairs * 4):
                    break
            if len(overlaps) >= max(50, debug_max_pairs * 4):
                break

    if overlaps:
        overlaps.sort(key=lambda r: (-(float(r.get("iou", 0.0) or 0.0)), -(float(r.get("contain", 0.0) or 0.0))))
        overlaps = overlaps[: max(1, debug_max_pairs)]
        try:
            dedupe_debug_blob["remaining_overlaps"] = overlaps
        except Exception:
            pass
        # If we have remaining overlaps with dedupe enabled, emit at least a summary even
        # when debug logging is off (so users know to enable it).
        _dedupe_debug_log(
            dedupe_debug or dedupe_enabled,
            "dedupe.remaining_overlaps",
            file_id=str(meta.get("id") or ""),
            overlaps=overlaps,
            note="Set output.dedupe.debug_log=true (or env DOOR_DETECTOR_DEDUPE_DEBUG=1) for more detail.",
        )

        # Detailed analysis (only when debug is on): run the predicate and dump key geom.
        if dedupe_debug:
            for rec in overlaps:
                aid = str((rec.get("a") or {}).get("id") or "")
                bid = str((rec.get("b") or {}).get("id") or "")
                a_obj = next((x for x in final if str(x.get("id") or "") == aid), None)
                b_obj = next((x for x in final if str(x.get("id") or "") == bid), None)
                if not (isinstance(a_obj, dict) and isinstance(b_obj, dict)):
                    continue
                try:
                    ab = a_obj.get("bbox_xyxy")
                    bb = b_obj.get("bbox_xyxy")
                    pred_ab = bool(is_duplicate(a_obj, b_obj, dedupe_conf))
                    pred_ba = bool(is_duplicate(b_obj, a_obj, dedupe_conf))
                except Exception:
                    ab, bb, pred_ab, pred_ba = a_obj.get("bbox_xyxy"), b_obj.get("bbox_xyxy"), False, False
                _dedupe_debug_log(
                    True,
                    "dedupe.overlap_pair",
                    file_id=str(meta.get("id") or ""),
                    iou=float(rec.get("iou", 0.0) or 0.0),
                    contain=float(rec.get("contain", 0.0) or 0.0),
                    a_id=aid,
                    a_type=str(a_obj.get("type") or ""),
                    a_conf=float(a_obj.get("confidence", 0.0) or 0.0),
                    a_bbox=ab,
                    a_geom=(a_obj.get("geom") if isinstance(a_obj.get("geom"), dict) else {}),
                    a_feat=(a_obj.get("features") if isinstance(a_obj.get("features"), dict) else {}),
                    b_id=bid,
                    b_type=str(b_obj.get("type") or ""),
                    b_conf=float(b_obj.get("confidence", 0.0) or 0.0),
                    b_bbox=bb,
                    b_geom=(b_obj.get("geom") if isinstance(b_obj.get("geom"), dict) else {}),
                    b_feat=(b_obj.get("features") if isinstance(b_obj.get("features"), dict) else {}),
                    pred_a_to_b=bool(pred_ab),
                    pred_b_to_a=bool(pred_ba),
                )

    out = {"doors": final, "candidates": exported_candidates}
    # Make debug discoverable even when stderr is not visible (e.g. Streamlit UI).
    # Keep it compact: only populate when dedupe debug is enabled OR overlaps were detected.
    try:
        has_overlaps = bool(dedupe_debug_blob.get("remaining_overlaps"))
    except Exception:
        has_overlaps = False
    if bool(dedupe_debug) or has_overlaps:
        out["dedupe_debug"] = dedupe_debug_blob
    return out


def debug_explain_unmatched_box(
    *,
    primitives: Dict[str, Any],
    bbox_full_xyxy: List[float],
    config: Dict[str, Any],
    pad_px: float = 20.0,
    max_examples: int = 12,
    verbose: bool = True,
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
    if perf_enabled():
        perf_log(
            "doors.debug_explain.inputs",
            verbose=bool(verbose),
            lines_total=int(len(lines)),
            beziers_total=int(len(beziers)),
            roi_full_xyxy=[float(v) for v in roi],
        )

    # Cheap "nearby" filtering using control-point bbox.
    near_lines: List[int] = []
    with perf_span("doors.debug_explain.filter_lines", verbose=bool(verbose), lines_total=int(len(lines))):
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
    with perf_span("doors.debug_explain.filter_beziers", verbose=bool(verbose), beziers_total=int(len(beziers))):
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
    MAX_NEAR_LINES = 2000 if verbose else 650
    MAX_NEAR_BEZIERS = 2000 if verbose else 650
    MAX_PAIRINGS_VERBOSE = 20000 if verbose else 0
    MAX_POLY_ARCS_VERBOSE = 300 if verbose else 80
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
    #
    # IMPORTANT:
    # In "summary-only" mode (`verbose=False`), avoid indexing the entire page's line set
    # (can be hundreds of thousands). Restrict to the ROI-adjacent lines, which is sufficient
    # to answer "why did swing candidate generation fail near this box?"
    line_index = SpatialIndex(cell_size=200.0)
    iter_line_indices = range(len(lines)) if verbose else near_lines
    for i in iter_line_indices:
        try:
            ln = lines[int(i)]
        except Exception:
            continue
        try:
            x0 = float(min(ln["p0"]["x"], ln["p1"]["x"]))
            y0 = float(min(ln["p0"]["y"], ln["p1"]["y"]))
            x1 = float(max(ln["p0"]["x"], ln["p1"]["x"]))
            y1 = float(max(ln["p0"]["y"], ln["p1"]["y"]))
        except Exception:
            continue
        line_index.add(int(i), [x0, y0, x1, y1])

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
    if verbose:
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

    if verbose:
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

    # Polyline-arc diagnostics within the ROI subset of lines.
    # This can be non-trivial (path search in branched components), so time it when profiling is enabled.
    with perf_span(
        "doors.debug_explain.polyline_arc_debug",
        verbose=bool(verbose),
        near_lines=int(len(near_lines)),
    ):
        poly_dbg = _debug_polyline_arcs_from_lines_subset(
            lines=lines, line_indices=near_lines, config=config, max_rejected_examples=(25 if verbose else 0)
        )
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

    # Leaf-only swing candidates (when enabled): quick signal for cases where the leaf is vector
    # but arcs are missing/rasterized. Evaluate only within ROI-adjacent lines to keep this fast.
    leaf_only_near: List[Dict[str, Any]] = []
    leaf_only_dbg: Dict[str, Any] = {}
    try:
        leaf_only_conf = (swing_conf.get("leaf_only") or {}) if isinstance(swing_conf, dict) else {}
        if bool(leaf_only_conf.get("enabled", False)):
            scan_idx = list(near_lines)[: (2000 if verbose else 900)]
            with perf_span(
                "doors.debug_explain.leaf_only",
                verbose=bool(verbose),
                scan_lines=int(len(scan_idx)),
            ):
                leaf_only_near = detect_swing_leaf_only_candidates(
                    lines=lines, line_index=line_index, config=config, line_indices=scan_idx, debug_out=leaf_only_dbg
                )
    except Exception:
        leaf_only_near = []
        leaf_only_dbg = {}

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
        if len(arc_examples) < max_examples and verbose:
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

            if verbose:
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

            if verbose:
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
        "verbose": bool(verbose),
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
                "leaf_only": {
                    "enabled": bool(((swing_conf.get("leaf_only") or {}) if isinstance(swing_conf, dict) else {}).get("enabled", False)),
                    "corner_probe_px": float((((swing_conf.get("leaf_only") or {}) if isinstance(swing_conf, dict) else {}).get("corner_probe_px", 8.0) or 8.0)),
                    "corner_endpoint_snap_px": float((((swing_conf.get("leaf_only") or {}) if isinstance(swing_conf, dict) else {}).get("corner_endpoint_snap_px", 6.0) or 6.0)),
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
            "leaf_only_near": {
                "enabled": bool(((swing_conf.get("leaf_only") or {}) if isinstance(swing_conf, dict) else {}).get("enabled", False)),
                "near_count": int(len(leaf_only_near)),
                "near_examples": [
                    {"id": str(c.get("id") or ""), "bbox_xyxy": c.get("bbox_xyxy"), "features": c.get("features")}
                    for c in (leaf_only_near[: max(0, min(int(max_examples), 5))])
                    if isinstance(c, dict)
                ],
                "debug": leaf_only_dbg if isinstance(leaf_only_dbg, dict) and leaf_only_dbg else None,
            },
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

    # Add a concise, copy-friendly summary so logs don’t require expanding huge arrays.
    try:
        counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
        swing = report.get("swing") if isinstance(report.get("swing"), dict) else {}
        poly = swing.get("polyline_arc_near") if isinstance(swing.get("polyline_arc_near"), dict) else {}
        poly_arcs = list(poly.get("arc_candidates") or []) if isinstance(poly.get("arc_candidates"), list) else []
        poly_pass = 0
        poly_fail = 0
        poly_fail_counts: dict[str, int] = {"radius": 0, "rmse": 0, "angle": 0}
        for a in poly_arcs:
            fails = a.get("fails") if isinstance(a, dict) else None
            if isinstance(fails, list) and len(fails) == 0:
                poly_pass += 1
            else:
                poly_fail += 1
                try:
                    if isinstance(fails, list):
                        for f in fails:
                            sf = str(f or "")
                            if "arc.radius" in sf:
                                poly_fail_counts["radius"] = int(poly_fail_counts.get("radius", 0)) + 1
                            elif "arc.rmse" in sf:
                                poly_fail_counts["rmse"] = int(poly_fail_counts.get("rmse", 0)) + 1
                            elif "arc.angle" in sf:
                                poly_fail_counts["angle"] = int(poly_fail_counts.get("angle", 0)) + 1
                except Exception:
                    pass

        # NOTE: `arc_pass_near_count` historically tracked only bezier-pass arcs in some builds.
        # Use it as informational, but rely on `poly_pass`/`beziers_near` for primary diagnosis.
        arc_pass_near = int(swing.get("arc_pass_near_count") or 0)
        arc_fail_counts_near = swing.get("arc_fail_counts_near") if isinstance(swing.get("arc_fail_counts_near"), dict) else {}
        leaf_stats = swing.get("leaf_pair_stats_near") if isinstance(swing.get("leaf_pair_stats_near"), dict) else {}
        pool_pass = int(leaf_stats.get("pool_pass") or 0)
        strict_pass = int(leaf_stats.get("strict_pass") or 0)

        # Extra “is there any obvious swing geometry?” signals.
        non_dashed_non_axis = 0
        try:
            for l_idx in list(near_lines)[: max(0, min(len(near_lines), 800))]:
                ln = lines[int(l_idx)]
                if bool(_is_dashed_primitive(ln)):
                    continue
                p0x = float(ln["p0"]["x"])
                p0y = float(ln["p0"]["y"])
                p1x = float(ln["p1"]["x"])
                p1y = float(ln["p1"]["y"])
                dx = p1x - p0x
                dy = p1y - p0y
                if math.hypot(dx, dy) < 2.0:
                    continue
                ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
                if abs(ang - 0.0) < 5.0 or abs(ang - 90.0) < 5.0:
                    continue
                non_dashed_non_axis += 1
        except Exception:
            non_dashed_non_axis = 0

        # Derive a single “what likely went wrong” label.
        primary = "unknown"
        hint: str | None = None
        top_arc_fail = ""
        try:
            if isinstance(arc_fail_counts_near, dict) and arc_fail_counts_near:
                # Only report a top fail if there is a non-zero winner.
                k, v = max(arc_fail_counts_near.items(), key=lambda kv: int(kv[1] or 0))
                top_arc_fail = str(k) if int(v or 0) > 0 else ""
        except Exception:
            top_arc_fail = ""

        top_poly_fail = ""
        try:
            if isinstance(poly_fail_counts, dict) and poly_fail_counts:
                k2, v2 = max(poly_fail_counts.items(), key=lambda kv: int(kv[1] or 0))
                top_poly_fail = str(k2) if int(v2 or 0) > 0 else ""
        except Exception:
            top_poly_fail = ""

        beziers_near = int(counts.get("beziers_near_roi") or 0)
        # Common failure mode: leaf line exists, but the curved arc is missing (often rasterized).
        # The main swing detector is arc-first, so this yields no `swing` candidate unless
        # leaf-only candidates are enabled.
        # NOTE:
        # `arc_pass_near_count` historically tracked only *bezier* arcs in some builds.
        # Use `poly_pass` as the polyline-arc analogue so we don't misclassify cases where
        # polyline arcs exist but there are no bezier primitives near the ROI.
        # Only call this "no_arc_primitives" when there truly is no arc geometry nearby.
        # If polyline arc candidates exist but fail thresholds, prefer the threshold-based diagnosis below.
        leaf_only_enabled = bool(((swing_conf.get("leaf_only") or {}) if isinstance(swing_conf, dict) else {}).get("enabled", False))
        leaf_only_count = int(len(leaf_only_near))

        if beziers_near <= 0 and len(poly_arcs) <= 0 and int(arc_pass_near) <= 0 and int(poly_pass) <= 0 and non_dashed_non_axis > 0:
            primary = "no_arc_primitives_near_roi"
            if leaf_only_enabled and leaf_only_count <= 0:
                hint = (
                    "No bezier arc primitives near ROI, but diagonal (leaf-like) linework exists. "
                    "`swing.leaf_only` is enabled but produced 0 candidates near this ROI (see `summary.top_leaf_only_fail`). "
                    "This usually means the leaf line's hinge endpoint lacked sufficient nearby axis-aligned wall support."
                )
            else:
                hint = (
                    "No bezier arc primitives near ROI, but diagonal (leaf-like) linework exists. "
                    "If the swing arc is missing/rasterized, arc-first swing detection cannot produce a `swing` candidate. "
                    "Consider enabling `swing.leaf_only` candidates for snapping/labeling."
                )
        elif beziers_near <= 0 and int(poly_pass) <= 0 and len(poly_arcs) <= 0 and non_dashed_non_axis <= 0:
            primary = "no_vector_arc_or_leaf_near_roi"
            hint = (
                "No bezier arcs and no non-dashed, non-axis linework near ROI. "
                "The intended swing door symbol may be rasterized (or otherwise absent from vector primitives), "
                "so arc/leaf-based candidate generation cannot produce a swing candidate."
            )
        elif arc_pass_near <= 0 and poly_pass <= 0:
            if beziers_near <= 0 and len(poly_arcs) <= 0:
                primary = "no_arc_geometry_near_roi"
                hint = "No bezier or polyline arcs near ROI; swing door arcs may be rasterized or absent from primitives."
            else:
                # Prefer polyline failure signal when available; the most common missing-door case
                # is a polyline arc that exists but fails circle-fit.
                top_any_fail = top_poly_fail or top_arc_fail or "unknown"
                primary = f"no_arc_passed_thresholds:{top_any_fail}"
                if top_any_fail == "rmse":
                    hint = (
                        "Arcs were found but circle-fit RMSE failed (often for polyline arcs). "
                        "Consider loosening swing.arc.max_circle_fit_rmse or enabling a relative RMSE tolerance "
                        "(e.g. swing.arc.max_circle_fit_rmse_ratio) / improving polyline arc extraction."
                    )
                elif top_any_fail == "radius":
                    hint = "Arcs were found but radius bounds failed; consider loosening swing.arc.{min_radius_px,max_radius_px}."
                elif top_any_fail == "angle":
                    hint = "Arcs were found but angle span failed; consider loosening swing.arc.{min_angle_deg,max_angle_deg}."
        else:
            if pool_pass <= 0:
                primary = "leaf_pairing_failed_for_all_arcs"
                hint = "Swing arcs exist near ROI, but no nearby line qualified as a leaf even under pool rules; door leaf may be merged into wall geometry or missing as a primitive."
            elif strict_pass <= 0:
                primary = "only_pool_pairs_passed"
                hint = "Loose (pool) arc+leaf pairs exist, but strict leaf rules rejected them. Candidate should exist, but final doors may be missing."
            else:
                primary = "arcs_and_leaf_pairs_exist"

        report["summary"] = {
            "primary_failure": primary,
            "hint": hint,
            "counts": {
                "lines_near_roi": int(counts.get("lines_near_roi") or 0),
                "beziers_near_roi": beziers_near,
                "non_dashed_non_axis_lines_near_roi": int(non_dashed_non_axis),
                "polyline_arc_candidates_near": int(len(poly_arcs)),
                "polyline_arc_pass_near": int(poly_pass),
                "polyline_arc_fail_near": int(poly_fail),
                "leaf_only_candidates_near": int(len(leaf_only_near)),
                "arc_pass_near_count": int(arc_pass_near),
                "leaf_pool_pass": int(pool_pass),
                "leaf_strict_pass": int(strict_pass),
            },
            "top_leaf_pool_fail": None,
            "top_leaf_strict_fail": None,
            "top_leaf_only_fail": None,
            "top_arc_fail": top_arc_fail or None,
            "top_polyline_arc_fail": top_poly_fail or None,
        }
        try:
            # Expose the most common strict/pool leaf exclusion in summary-only mode.
            pool_fc = leaf_stats.get("pool_fail_counts") if isinstance(leaf_stats, dict) else None
            strict_fc = leaf_stats.get("strict_fail_counts") if isinstance(leaf_stats, dict) else None

            def _top_fail(fc: Any) -> Optional[str]:
                if not isinstance(fc, dict) or not fc:
                    return None
                best_k = None
                best_v = 0
                for k, v in fc.items():
                    try:
                        iv = int(v or 0)
                    except Exception:
                        iv = 0
                    if iv > best_v:
                        best_v = iv
                        best_k = str(k)
                return best_k if best_k and best_v > 0 else None

            report["summary"]["top_leaf_pool_fail"] = _top_fail(pool_fc)
            report["summary"]["top_leaf_strict_fail"] = _top_fail(strict_fc)
        except Exception:
            pass
        try:
            # Leaf-only failures: expose the most common reason (when enabled).
            lo = swing.get("leaf_only_near") if isinstance(swing, dict) else None
            lo_dbg = lo.get("debug") if isinstance(lo, dict) else None
            lo_counts = lo_dbg.get("counts") if isinstance(lo_dbg, dict) else None
            if isinstance(lo_counts, dict) and lo_counts:
                # Prefer "no support" as the main actionable leaf-only failure.
                order = [
                    "skipped_no_support",
                    "skipped_midwall_tip_clearance",
                    "skipped_midwall_angle_to_wall",
                    "skipped_len",
                    "skipped_axis_aligned",
                    "skipped_dashed",
                ]
                best_k = None
                best_v = 0
                for k in order:
                    try:
                        iv = int(lo_counts.get(k, 0) or 0)
                    except Exception:
                        iv = 0
                    if iv > best_v:
                        best_v = iv
                        best_k = k
                if best_k and best_v > 0:
                    report["summary"]["top_leaf_only_fail"] = str(best_k)
        except Exception:
            pass
    except Exception:
        pass
    return report

