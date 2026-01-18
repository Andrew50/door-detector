"""Regression test: candidate IDs are stable across primitive reordering."""

from __future__ import annotations

from door_detector.doors.detect import detect_doors


def _config() -> dict:
    return {
        "mode_policy": {"vector": "vector_rules", "hybrid": "vector_rules", "scan": "empty_with_message"},
        "output": {
            "min_candidate_confidence": 0.0,
            "min_confidence_after_reweight": 0.99,  # irrelevant (no model)
            "nms_iou": 0.35,
            "max_doors": 50,
            "max_candidates": 5000,
            "max_candidates_before_nms": 2000,
        },
        "swing": {
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
        },
    }


def test_candidate_ids_stable_under_reorder() -> None:
    # Two door glyphs (two beziers, two lines), then reorder the primitives lists.
    primitives_a = {
        "beziers": [
            # door 1 quarter arc
            {"p0": {"x": 140.0, "y": 100.0}, "p1": {"x": 140.0, "y": 122.0}, "p2": {"x": 122.0, "y": 140.0}, "p3": {"x": 100.0, "y": 140.0}},
            # door 2 quarter arc
            {"p0": {"x": 240.0, "y": 200.0}, "p1": {"x": 240.0, "y": 222.0}, "p2": {"x": 222.0, "y": 240.0}, "p3": {"x": 200.0, "y": 240.0}},
        ],
        "lines": [
            {"p0": {"x": 100.0, "y": 100.0}, "p1": {"x": 140.0, "y": 100.0}},
            {"p0": {"x": 200.0, "y": 200.0}, "p1": {"x": 240.0, "y": 200.0}},
        ],
    }
    primitives_b = {"beziers": list(reversed(primitives_a["beziers"])), "lines": list(reversed(primitives_a["lines"]))}
    meta = {"id": "t", "mode": "vector"}

    det_a = detect_doors(primitives_a, meta, _config())
    det_b = detect_doors(primitives_b, meta, _config())

    ids_a = {str(c.get("id")) for c in (det_a.get("candidates") or []) if c.get("id") is not None}
    ids_b = {str(c.get("id")) for c in (det_b.get("candidates") or []) if c.get("id") is not None}
    assert ids_a, "expected non-empty candidates for baseline primitives"
    assert ids_a == ids_b, "candidate IDs changed when primitives ordering changed"


def main() -> int:
    test_candidate_ids_stable_under_reorder()
    print("✓ candidate id stability test passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

