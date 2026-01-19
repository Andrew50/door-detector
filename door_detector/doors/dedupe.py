"""Duplicate suppression helpers for door candidates.

This module centralizes "same-door" duplicate logic so both:
- backend detection (candidate export + final door selection), and
- UI behaviors (auto-hide duplicates on confirm)
use identical rules.

The goal is to remove near-identical / variant candidates (e.g. slightly different
bboxes for the same underlying vectors) without merging adjacent doors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _normalize_bbox_xyxy(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def bbox_area(bbox_xyxy: List[float] | Tuple[float, float, float, float]) -> float:
    nb = _normalize_bbox_xyxy(bbox_xyxy)
    if nb is None:
        return 0.0
    x0, y0, x1, y1 = nb
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_intersection_area(
    a_xyxy: List[float] | Tuple[float, float, float, float],
    b_xyxy: List[float] | Tuple[float, float, float, float],
) -> float:
    a = _normalize_bbox_xyxy(a_xyxy)
    b = _normalize_bbox_xyxy(b_xyxy)
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _compute_iou(a_xyxy: Tuple[float, float, float, float], b_xyxy: Tuple[float, float, float, float]) -> float:
    """IoU for two normalized (x0,y0,x1,y1) boxes.

    Kept local to avoid importing heavier geometry dependencies.
    """
    ax0, ay0, ax1, ay1 = a_xyxy
    bx0, by0, bx1, by1 = b_xyxy
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def bbox_containment(
    a_xyxy: List[float] | Tuple[float, float, float, float],
    b_xyxy: List[float] | Tuple[float, float, float, float],
) -> float:
    """Return intersection(a,b) / min(area(a), area(b))."""
    ia = bbox_intersection_area(a_xyxy, b_xyxy)
    if ia <= 0.0:
        return 0.0
    aa = bbox_area(a_xyxy)
    ba = bbox_area(b_xyxy)
    denom = min(aa, ba)
    if denom <= 0.0:
        return 0.0
    return float(ia / denom)


def candidate_center(bbox_xyxy: List[float] | Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
    nb = _normalize_bbox_xyxy(bbox_xyxy)
    if nb is None:
        return None
    x0, y0, x1, y1 = nb
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def candidate_size_min_dim(bbox_xyxy: List[float] | Tuple[float, float, float, float]) -> float:
    nb = _normalize_bbox_xyxy(bbox_xyxy)
    if nb is None:
        return 0.0
    x0, y0, x1, y1 = nb
    return float(min(max(0.0, x1 - x0), max(0.0, y1 - y0)))


def _cand_type(c: Dict[str, Any]) -> str:
    try:
        return str(c.get("type") or "").strip().lower()
    except Exception:
        return ""


def _cand_id(c: Dict[str, Any]) -> str:
    try:
        v = c.get("id")
    except Exception:
        v = None
    return "" if v is None else str(v)


def _get_geom(c: Dict[str, Any]) -> Dict[str, Any]:
    g = c.get("geom")
    return g if isinstance(g, dict) else {}


def _get_features(c: Dict[str, Any]) -> Dict[str, Any]:
    f = c.get("features")
    return f if isinstance(f, dict) else {}


def _pt2(v: Any) -> Optional[Tuple[float, float]]:
    try:
        if not (isinstance(v, list) and len(v) == 2):
            return None
        x, y = float(v[0]), float(v[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return (x, y)
    except Exception:
        return None


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _safe_float(v: Any, *, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _radius_ratio_ok(ra: float, rb: float, *, max_ratio: float) -> bool:
    ra = float(ra)
    rb = float(rb)
    if not (ra > 0.0 and rb > 0.0):
        return False
    rmax = max(ra, rb)
    rmin = max(1e-6, min(ra, rb))
    return (rmax / rmin) <= float(max_ratio)


def _angle_ok(aa: float, ab: float, *, max_deg: float) -> bool:
    try:
        d = abs(float(aa) - float(ab))
    except Exception:
        return False
    if not math.isfinite(d):
        return False
    return d <= float(max_deg)


def _endpoints_match(
    ea: Any,
    eb: Any,
    *,
    tol_px: float,
) -> bool:
    """Order-insensitive 2-point matching with tolerance."""
    try:
        if not (isinstance(ea, list) and len(ea) == 2 and isinstance(eb, list) and len(eb) == 2):
            return False
        a0 = _pt2(ea[0])
        a1 = _pt2(ea[1])
        b0 = _pt2(eb[0])
        b1 = _pt2(eb[1])
        if a0 is None or a1 is None or b0 is None or b1 is None:
            return False
        tol = float(tol_px)
        d_same = _dist(a0, b0) + _dist(a1, b1)
        d_swap = _dist(a0, b1) + _dist(a1, b0)
        return min(d_same, d_swap) <= (2.0 * tol)
    except Exception:
        return False


def _bbox_duplicate_gate(
    a: Dict[str, Any],
    b: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[bool, float, float, float, float]:
    """Return (passes_gate, iou, contain, center_dist, size_min)."""
    ab = _normalize_bbox_xyxy(a.get("bbox_xyxy"))
    bb = _normalize_bbox_xyxy(b.get("bbox_xyxy"))
    if ab is None or bb is None:
        return False, 0.0, 0.0, float("inf"), 0.0

    iou = float(_compute_iou(ab, bb))
    contain = float(bbox_containment([ab[0], ab[1], ab[2], ab[3]], [bb[0], bb[1], bb[2], bb[3]]))

    ca = candidate_center([ab[0], ab[1], ab[2], ab[3]])
    cb = candidate_center([bb[0], bb[1], bb[2], bb[3]])
    if ca is None or cb is None:
        return False, iou, contain, float("inf"), 0.0
    center_dist = float(math.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    size_min = float(min(candidate_size_min_dim([ab[0], ab[1], ab[2], ab[3]]), candidate_size_min_dim([bb[0], bb[1], bb[2], bb[3]])))

    iou_dup = float(cfg.get("iou_dup", 0.85) or 0.85)
    contain_dup = float(cfg.get("contain_dup", 0.92) or 0.92)
    center_px = float(cfg.get("center_px", 4.0) or 4.0)
    center_frac = float(cfg.get("center_frac", 0.15) or 0.15)

    prox_ok = center_dist <= max(center_px, center_frac * max(0.0, size_min))
    passes = (iou >= iou_dup) or ((contain >= contain_dup) and prox_ok)
    return bool(passes), iou, contain, center_dist, size_min


def is_duplicate(a: Dict[str, Any], b: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """Return True if `b` is a same-door duplicate variant of `a`.

    This is intentionally stricter than "nearby": it should collapse bbox variants of
    the same door without merging adjacent doors.
    """
    ok, iou, _contain, _cd, _sz = _bbox_duplicate_gate(a, b, cfg)
    if not ok:
        return False

    ta = _cand_type(a)
    tb = _cand_type(b)

    # If IoU is extremely high, treat as duplicate without expensive per-type checks.
    iou_skip_keypoints = float(cfg.get("iou_skip_keypoints", 0.95) or 0.95)
    if iou >= iou_skip_keypoints and ta == tb and ta:
        return True

    # Cross-type: prefer `swing` over `swing_arc` when geometry matches.
    if {ta, tb} == {"swing", "swing_arc"}:
        swing = a if ta == "swing" else b
        arc = b if swing is a else a
        gs = _get_geom(swing)
        ga = _get_geom(arc)
        fs = _get_features(swing)
        fa = _get_features(arc)
        cs = _pt2(gs.get("center_xy"))
        ca = _pt2(ga.get("center_xy"))
        if cs is None or ca is None:
            return False
        rs = _safe_float(fs.get("radius"), default=0.0)
        ra = _safe_float(fa.get("radius"), default=0.0)
        angs = _safe_float(fs.get("angle_span"), default=0.0)
        anga = _safe_float(fa.get("angle_span"), default=0.0)

        swing_cfg = cfg.get("swing") if isinstance(cfg.get("swing"), dict) else {}
        arc_cfg = cfg.get("swing_arc") if isinstance(cfg.get("swing_arc"), dict) else {}
        max_ratio = float(swing_cfg.get("radius_ratio", 1.10) or 1.10)
        max_ang = float(swing_cfg.get("angle_deg", 10.0) or 10.0)
        center_tol = float(arc_cfg.get("center_px", 4.0) or 4.0)
        ep_tol = float(arc_cfg.get("endpoint_px", 6.0) or 6.0)

        if _dist(cs, ca) > center_tol:
            return False
        if not _radius_ratio_ok(rs, ra, max_ratio=max_ratio):
            return False
        if not _angle_ok(angs, anga, max_deg=max_ang):
            return False
        # Optional endpoints check when present on both.
        ea = ga.get("arc_endpoints_xy")
        es = gs.get("arc_endpoints_xy")
        if ea is not None and es is not None:
            if not _endpoints_match(es, ea, tol_px=ep_tol):
                return False
        return True

    # From here on, only same-type duplicates.
    if ta != tb or not ta:
        return False

    if ta == "swing":
        ga = _get_geom(a)
        gb = _get_geom(b)
        fa = _get_features(a)
        fb = _get_features(b)

        # Prefer circle params (center/radius/angle) over hinge/tip endpoints.
        # In real-world PDFs, small changes in which leaf line is chosen can move
        # hinge/tip a few pixels even when it is the same door; the circle fit is
        # typically more stable for duplicates.
        ca = _pt2(ga.get("center_xy")) or _pt2(ga.get("hinge_xy"))
        cb = _pt2(gb.get("center_xy")) or _pt2(gb.get("hinge_xy"))
        ra = _safe_float(fa.get("radius"), default=0.0)
        rb = _safe_float(fb.get("radius"), default=0.0)
        aa = _safe_float(fa.get("angle_span"), default=0.0)
        ab = _safe_float(fb.get("angle_span"), default=0.0)

        if ca is None or cb is None:
            return False
        swing_cfg = cfg.get("swing") if isinstance(cfg.get("swing"), dict) else {}
        center_tol = float(swing_cfg.get("center_px", cfg.get("center_px", 4.0)) or cfg.get("center_px", 4.0) or 4.0)
        ep_tol = float(swing_cfg.get("endpoint_px", 10.0) or 10.0)
        hinge_px = float(swing_cfg.get("hinge_px", 3.0) or 3.0)
        hinge_r_frac = float(swing_cfg.get("hinge_radius_frac", 0.05) or 0.05)
        tip_px = float(swing_cfg.get("tip_px", 4.0) or 4.0)
        tip_r_frac = float(swing_cfg.get("tip_radius_frac", 0.08) or 0.08)
        max_ratio = float(swing_cfg.get("radius_ratio", 1.10) or 1.10)
        max_ang = float(swing_cfg.get("angle_deg", 10.0) or 10.0)

        if _dist(ca, cb) > center_tol:
            return False
        if not _radius_ratio_ok(ra, rb, max_ratio=max_ratio):
            return False
        if not _angle_ok(aa, ab, max_deg=max_ang):
            return False

        # Endpoints check (high-signal): if both candidates share the same arc endpoints,
        # treat as duplicates even if the chosen leaf line differs (hinge/tip may shift).
        ea = ga.get("arc_endpoints_xy")
        eb = gb.get("arc_endpoints_xy")
        if ea is not None and eb is not None:
            if not _endpoints_match(ea, eb, tol_px=ep_tol):
                return False
            # Same arc -> same door symbol. Do not require hinge/tip agreement, since
            # multiple leaf-line pairings can exist for a single arc in messy PDFs.
            return True

        # Optional hinge/tip check when available (guardrail; relaxed vs earlier).
        ha = _pt2(ga.get("hinge_xy"))
        hb = _pt2(gb.get("hinge_xy"))
        ta_pt = _pt2(ga.get("tip_xy"))
        tb_pt = _pt2(gb.get("tip_xy"))
        r_ref = max(0.0, 0.5 * (ra + rb))
        hinge_tol = max(hinge_px, hinge_r_frac * r_ref)
        tip_tol = max(tip_px, tip_r_frac * r_ref)
        if ha is not None and hb is not None and _dist(ha, hb) > hinge_tol:
            return False
        if ta_pt is not None and tb_pt is not None and _dist(ta_pt, tb_pt) > tip_tol:
            return False
        return True

    if ta == "swing_arc":
        ga = _get_geom(a)
        gb = _get_geom(b)
        fa = _get_features(a)
        fb = _get_features(b)
        ca = _pt2(ga.get("center_xy"))
        cb = _pt2(gb.get("center_xy"))
        ra = _safe_float(fa.get("radius"), default=0.0)
        rb = _safe_float(fb.get("radius"), default=0.0)
        aa = _safe_float(fa.get("angle_span"), default=0.0)
        ab = _safe_float(fb.get("angle_span"), default=0.0)
        if ca is None or cb is None:
            return False
        arc_cfg = cfg.get("swing_arc") if isinstance(cfg.get("swing_arc"), dict) else {}
        center_tol = float(arc_cfg.get("center_px", 4.0) or 4.0)
        ep_tol = float(arc_cfg.get("endpoint_px", 6.0) or 6.0)
        max_ratio = float(arc_cfg.get("radius_ratio", 1.10) or 1.10)
        max_ang = float(arc_cfg.get("angle_deg", 10.0) or 10.0)
        if _dist(ca, cb) > center_tol:
            return False
        if not _radius_ratio_ok(ra, rb, max_ratio=max_ratio):
            return False
        if not _angle_ok(aa, ab, max_deg=max_ang):
            return False
        ea = ga.get("arc_endpoints_xy")
        eb = gb.get("arc_endpoints_xy")
        if ea is not None and eb is not None:
            if not _endpoints_match(ea, eb, tol_px=ep_tol):
                return False
        return True

    if ta == "swing_leaf":
        ga = _get_geom(a)
        gb = _get_geom(b)
        ha = _pt2(ga.get("hinge_xy"))
        hb = _pt2(gb.get("hinge_xy"))
        ta_pt = _pt2(ga.get("tip_xy"))
        tb_pt = _pt2(gb.get("tip_xy"))
        if ha is None or hb is None or ta_pt is None or tb_pt is None:
            return False
        swing_cfg = cfg.get("swing") if isinstance(cfg.get("swing"), dict) else {}
        hinge_px = float(swing_cfg.get("hinge_px", 3.0) or 3.0)
        tip_px = float(swing_cfg.get("tip_px", 4.0) or 4.0)
        if _dist(ha, hb) > hinge_px:
            return False
        if _dist(ta_pt, tb_pt) > tip_px:
            return False
        return True

    if ta == "double":
        ca = a.get("components") if isinstance(a.get("components"), dict) else {}
        cb = b.get("components") if isinstance(b.get("components"), dict) else {}
        sa = ca.get("swing_ids")
        sb = cb.get("swing_ids")
        if isinstance(sa, list) and isinstance(sb, list):
            try:
                if sorted([str(x) for x in sa]) == sorted([str(x) for x in sb]):
                    return True
            except Exception:
                pass
        return True

    if ta == "bifold":
        pa = a.get("primitives") if isinstance(a.get("primitives"), dict) else {}
        pb = b.get("primitives") if isinstance(b.get("primitives"), dict) else {}
        la = pa.get("lines")
        lb = pb.get("lines")
        if isinstance(la, list) and isinstance(lb, list) and la and lb:
            try:
                sa = {int(x) for x in la if x is not None}
                sb = {int(x) for x in lb if x is not None}
                if sa and sb:
                    inter = len(sa & sb)
                    union = len(sa | sb)
                    if union > 0 and (inter / union) >= 0.9:
                        return True
            except Exception:
                pass
        return True

    if ta == "pocket":
        fa = _get_features(a)
        fb = _get_features(b)
        la = _safe_float(fa.get("track_length_px"), default=0.0)
        lb = _safe_float(fb.get("track_length_px"), default=0.0)
        if la > 0.0 and lb > 0.0:
            rmax = max(la, lb)
            rmin = max(1e-6, min(la, lb))
            if (rmax / rmin) > 1.10:
                return False
        return True

    # Default for other types: if it passed bbox gate and types match, treat as duplicate.
    return True


def suppress_duplicates(
    candidates: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Return (kept, duplicate_map) for a candidate list.

    The returned list is ordered by descending score and is deterministic (ties
    broken by bbox area then id).
    """
    if not candidates:
        return [], {}

    # Sort by confidence desc, then heuristic_confidence desc, then area asc, then id asc.
    def _score_key(c: Dict[str, Any]) -> Tuple[float, float, float, str]:
        conf = _safe_float(c.get("confidence", 0.0), default=0.0)
        hconf = _safe_float(c.get("heuristic_confidence", conf), default=conf)
        bb = _normalize_bbox_xyxy(c.get("bbox_xyxy"))
        area = bbox_area([bb[0], bb[1], bb[2], bb[3]]) if bb is not None else float("inf")
        return (-conf, -hconf, float(area), _cand_id(c))

    ordered: List[Dict[str, Any]] = [c for c in candidates if isinstance(c, dict)]
    ordered.sort(key=_score_key)

    cell = float(cfg.get("grid_cell_px", 80.0) or 80.0)
    if not (cell > 0.0 and math.isfinite(cell)):
        cell = 80.0

    kept: List[Dict[str, Any]] = []
    dup_map: Dict[str, str] = {}
    grid: Dict[Tuple[int, int], List[int]] = {}

    def _cell_for(c: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        bb = _normalize_bbox_xyxy(c.get("bbox_xyxy"))
        if bb is None:
            return None
        cc = candidate_center([bb[0], bb[1], bb[2], bb[3]])
        if cc is None:
            return None
        return (int(math.floor(cc[0] / cell)), int(math.floor(cc[1] / cell)))

    for cand in ordered:
        cid = _cand_id(cand)
        if not cid:
            # Keep unlabeled candidates (rare) – we cannot map duplicates reliably.
            kept.append(cand)
            continue
        cell_xy = _cell_for(cand)
        suppressed_by: Optional[str] = None
        if cell_xy is not None:
            cx, cy = cell_xy
            # Search neighboring bins (3x3) for potential duplicates.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    key = (cx + dx, cy + dy)
                    for ki in grid.get(key, []):
                        prev = kept[ki]
                        if is_duplicate(prev, cand, cfg):
                            suppressed_by = _cand_id(prev)
                            break
                    if suppressed_by:
                        break
                if suppressed_by:
                    break

        if suppressed_by:
            dup_map[cid] = suppressed_by
            continue

        kept_idx = len(kept)
        kept.append(cand)
        if cell_xy is not None:
            grid.setdefault(cell_xy, []).append(kept_idx)

    return kept, dup_map

