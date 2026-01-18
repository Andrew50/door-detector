"""Smoke/regression tests for multi-type candidate generation.

These tests call `detect_doors()` directly with hand-crafted primitives, so they
don't depend on PDF rendering/extraction.
"""

from __future__ import annotations

from door_detector.doors.detect import detect_doors


def _base_cfg() -> dict:
    return {
        "mode_policy": {"vector": "vector_rules", "hybrid": "vector_rules", "scan": "empty_with_message"},
        "output": {
            "min_candidate_confidence": 0.0,
            "min_confidence_after_reweight": 0.0,
            "nms_iou": 0.95,  # keep overlapping types for test visibility
            "max_doors": 200,
            "max_candidates": 5000,
            "max_candidates_before_nms": 5000,
        },
    }


def _swing_cfg() -> dict:
    cfg = _base_cfg()
    cfg["swing"] = {
        "enabled": True,
        "bezier_sampling_points": 17,
        "arc": {
            "min_angle_deg": 20,
            "max_angle_deg": 125,
            "max_circle_fit_rmse": 2.5,
            "min_radius_px": 8,
            "max_radius_px": 220,
            "suppress_circle_clusters": False,
        },
        "leaf": {
            "min_length_ratio": 0.45,
            "max_length_ratio": 1.25,
            "max_hinge_dist_ratio": 0.25,
            "require_endpoint_near_center": True,
            "max_center_dist_ratio": 0.25,
            "max_radial_angle_deg": 25,
            "max_tip_to_arc_ratio": 0.35,
        },
        "scoring": {"w_fit": 0.45, "w_angle": 0.25, "w_proximity": 0.30},
    }
    return cfg


def _quarter_arc_bezier(cx: float, cy: float, r: float) -> dict:
    # Quarter-circle approximation (0° -> 90°).
    k = r * 0.5522847498307936
    return {
        "p0": {"x": cx + r, "y": cy},
        "p1": {"x": cx + r, "y": cy + k},
        "p2": {"x": cx + k, "y": cy + r},
        "p3": {"x": cx, "y": cy + r},
    }


def test_double_candidate_generated() -> None:
    cfg = _swing_cfg()
    cfg["double"] = {"enabled": True}
    meta = {"id": "t", "mode": "vector"}

    # Two swing doors near each other.
    primitives_a = {
        "beziers": [_quarter_arc_bezier(100.0, 100.0, 40.0), _quarter_arc_bezier(170.0, 100.0, 40.0)],
        "lines": [
            {"p0": {"x": 100.0, "y": 100.0}, "p1": {"x": 140.0, "y": 100.0}},
            {"p0": {"x": 170.0, "y": 100.0}, "p1": {"x": 210.0, "y": 100.0}},
        ],
    }
    primitives_b = {"beziers": list(reversed(primitives_a["beziers"])), "lines": list(reversed(primitives_a["lines"]))}

    det_a = detect_doors(primitives_a, meta, cfg)
    det_b = detect_doors(primitives_b, meta, cfg)
    ids_a = {str(c.get("id")) for c in (det_a.get("candidates") or []) if c.get("type") == "double" and c.get("id") is not None}
    ids_b = {str(c.get("id")) for c in (det_b.get("candidates") or []) if c.get("type") == "double" and c.get("id") is not None}
    assert ids_a, "expected at least one double candidate id"
    assert ids_a == ids_b, "double candidate ids should be stable under primitive reordering"


def test_pocket_candidate_generated_and_id_stable() -> None:
    cfg = _base_cfg()
    cfg["pocket"] = {"enabled": True, "geometry": {"min_track_length_px": 10.0, "max_track_length_px": 1000.0}}
    meta = {"id": "t", "mode": "vector"}

    line = {
        "p0": {"x": 10.0, "y": 20.0},
        "p1": {"x": 110.0, "y": 20.0},
        "stroke_width": 1.0,
        "dash_pattern": [3.0, 3.0],
        "is_dashed": True,
    }
    primitives_a = {"lines": [line], "beziers": []}
    primitives_b = {"lines": [dict(line, p0=line["p1"], p1=line["p0"])], "beziers": []}  # reversed endpoints

    det_a = detect_doors(primitives_a, meta, cfg)
    det_b = detect_doors(primitives_b, meta, cfg)

    ids_a = {str(c.get("id")) for c in (det_a.get("candidates") or []) if c.get("type") == "pocket" and c.get("id") is not None}
    ids_b = {str(c.get("id")) for c in (det_b.get("candidates") or []) if c.get("type") == "pocket" and c.get("id") is not None}
    assert ids_a, "expected pocket candidate ids"
    assert ids_a == ids_b, "pocket candidate ids should be stable under endpoint reversal"


def test_bifold_candidate_generated_and_id_stable_under_reorder() -> None:
    cfg = _base_cfg()
    cfg["bifold"] = {
        "enabled": True,
        "zigzag": {"min_segments": 3, "max_segments": 8, "endpoint_snap_px": 3.0, "min_turn_angle_deg": 10.0},
        "geometry": {"pad_px": 2.0},
    }
    meta = {"id": "t", "mode": "vector"}

    # Simple zig-zag chain (3 connected segments).
    l0 = {"p0": {"x": 0.0, "y": 0.0}, "p1": {"x": 10.0, "y": 0.0}}
    l1 = {"p0": {"x": 10.0, "y": 0.0}, "p1": {"x": 20.0, "y": 10.0}}
    l2 = {"p0": {"x": 20.0, "y": 10.0}, "p1": {"x": 30.0, "y": 0.0}}

    primitives_a = {"lines": [l0, l1, l2], "beziers": []}
    primitives_b = {"lines": [l2, l0, l1], "beziers": []}

    det_a = detect_doors(primitives_a, meta, cfg)
    det_b = detect_doors(primitives_b, meta, cfg)

    ids_a = {str(c.get("id")) for c in (det_a.get("candidates") or []) if c.get("type") == "bifold" and c.get("id") is not None}
    ids_b = {str(c.get("id")) for c in (det_b.get("candidates") or []) if c.get("type") == "bifold" and c.get("id") is not None}
    assert ids_a, "expected bifold candidate ids"
    assert ids_a == ids_b, "bifold candidate ids should be stable under line reordering"


def main() -> int:
    test_double_candidate_generated()
    test_pocket_candidate_generated_and_id_stable()
    test_bifold_candidate_generated_and_id_stable_under_reorder()
    print("✓ multi-type candidate tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

