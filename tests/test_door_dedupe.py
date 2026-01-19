"""Unit tests for door duplicate suppression logic."""

from __future__ import annotations

from door_detector.doors.dedupe import is_duplicate, suppress_duplicates


def _dedupe_cfg() -> dict:
    return {
        "enabled": True,
        "iou_dup": 0.85,
        "contain_dup": 0.92,
        "center_px": 4.0,
        "center_frac": 0.15,
        "iou_skip_keypoints": 0.95,
        "grid_cell_px": 80.0,
        "swing": {
            "center_px": 6.0,
            "endpoint_px": 10.0,
            "hinge_px": 3.0,
            "hinge_radius_frac": 0.05,
            "tip_px": 4.0,
            "tip_radius_frac": 0.08,
            "radius_ratio": 1.10,
            "angle_deg": 10.0,
        },
        "swing_arc": {"center_px": 4.0, "endpoint_px": 10.0, "radius_ratio": 1.10, "angle_deg": 10.0},
    }


def test_suppress_duplicates_by_containment_even_when_iou_is_low() -> None:
    cfg = _dedupe_cfg()
    # Small box fully contained by a much larger one => containment near 1, IoU tiny.
    a = {"id": "a", "type": "pocket", "bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9, "features": {"track_length_px": 100.0}}
    b = {
        "id": "b",
        "type": "pocket",
        "bbox_xyxy": [-20.0, -20.0, 30.0, 30.0],
        "confidence": 0.6,
        "features": {"track_length_px": 100.0},
    }
    assert is_duplicate(a, b, cfg), "expected containment+proximity to mark as duplicate"
    kept, dup_map = suppress_duplicates([b, a], cfg)
    assert len(kept) == 1
    assert kept[0]["id"] == "a"
    assert dup_map.get("b") == "a"


def test_adjacent_swing_doors_not_collapsed_when_hinges_differ() -> None:
    cfg = _dedupe_cfg()
    # Bboxes overlap heavily (IoU ~0.9) but hinges differ a lot => NOT duplicates.
    a = {
        "id": "s1",
        "type": "swing",
        "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
        "confidence": 0.9,
        "geom": {"hinge_xy": [0.0, 0.0], "tip_xy": [40.0, 0.0], "center_xy": [50.0, 50.0]},
        "features": {"radius": 40.0, "angle_span": 90.0},
    }
    b = {
        "id": "s2",
        "type": "swing",
        "bbox_xyxy": [5.0, 0.0, 105.0, 100.0],
        "confidence": 0.88,
        "geom": {"hinge_xy": [50.0, 0.0], "tip_xy": [90.0, 0.0], "center_xy": [70.0, 50.0]},
        "features": {"radius": 40.0, "angle_span": 90.0},
    }
    assert not is_duplicate(a, b, cfg), "expected hinge mismatch to prevent dedupe"
    kept, _ = suppress_duplicates([a, b], cfg)
    assert len(kept) == 2


def test_cross_type_swing_arc_suppressed_when_matching_swing_exists() -> None:
    cfg = _dedupe_cfg()
    swing = {
        "id": "sw",
        "type": "swing",
        "bbox_xyxy": [80.0, 80.0, 140.0, 140.0],
        "confidence": 0.9,
        "geom": {
            "center_xy": [100.0, 100.0],
            "hinge_xy": [100.0, 100.0],
            "tip_xy": [140.0, 100.0],
            "arc_endpoints_xy": [[140.0, 100.0], [100.0, 140.0]],
        },
        "features": {"radius": 40.0, "angle_span": 90.0},
    }
    arc = {
        "id": "arc",
        "type": "swing_arc",
        "bbox_xyxy": [60.0, 60.0, 160.0, 160.0],
        "confidence": 0.5,
        "geom": {"center_xy": [100.0, 100.0], "arc_endpoints_xy": [[140.0, 100.0], [100.0, 140.0]]},
        "features": {"radius": 40.0, "angle_span": 90.0, "arc_only": 1.0},
    }
    assert is_duplicate(swing, arc, cfg), "expected swing to suppress matching arc-only candidate"
    kept, dup_map = suppress_duplicates([arc, swing], cfg)
    assert [k["id"] for k in kept] == ["sw"]
    assert dup_map.get("arc") == "sw"


def test_confirm_style_autohide_adds_duplicates_to_deleted_ids_only() -> None:
    cfg = _dedupe_cfg()
    cur = {"id": "a", "type": "pocket", "bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9, "features": {"track_length_px": 80.0}}
    dup = {
        "id": "b",
        "type": "pocket",
        "bbox_xyxy": [-20.0, -20.0, 30.0, 30.0],
        "confidence": 0.1,
        "features": {"track_length_px": 80.0},
    }
    deleted_ids: set[str] = set()
    for other in [cur, dup]:
        oid = str(other.get("id"))
        if oid and oid != str(cur["id"]) and is_duplicate(cur, other, cfg):
            deleted_ids.add(oid)
    deleted_ids.discard(str(cur["id"]))
    assert "b" in deleted_ids
    assert "a" not in deleted_ids


def main() -> int:
    test_suppress_duplicates_by_containment_even_when_iou_is_low()
    test_adjacent_swing_doors_not_collapsed_when_hinges_differ()
    test_cross_type_swing_arc_suppressed_when_matching_swing_exists()
    test_confirm_style_autohide_adds_duplicates_to_deleted_ids_only()
    print("✓ door dedupe tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

