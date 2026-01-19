from __future__ import annotations

from door_detector.step2_pipeline import _attach_legacy_ids_from_previous_doors_json


def test_step2_attaches_legacy_ids_from_previous_doors_json_by_pdf_iou() -> None:
    # Previous run had candidate id "d_old" at a stable PDF-space bbox.
    prev_doors = {
        "schema_version": 2,
        "candidates": [
            {
                "id": "d_old",
                "type": "swing",
                "bbox_pdf_xyxy": [10.0, 10.0, 20.0, 20.0],
                "legacy_ids": ["d_older"],
            }
        ],
    }

    # New run produced a different id but the same PDF-space bbox.
    new_candidates = [
        {
            "id": "d_new",
            "type": "swing",
            "bbox_pdf_xyxy": [10.1, 10.0, 20.0, 20.1],
            "legacy_ids": [],
        }
    ]

    rep = _attach_legacy_ids_from_previous_doors_json(new_candidates=new_candidates, prev_doors_data=prev_doors)
    assert rep.get("matched") == 1
    assert "legacy_ids" in new_candidates[0]
    # Should include old id and any older legacy ids.
    assert "d_old" in new_candidates[0]["legacy_ids"]
    assert "d_older" in new_candidates[0]["legacy_ids"]


def test_step2_does_not_cross_map_when_iou_is_too_low() -> None:
    prev_doors = {
        "schema_version": 2,
        "candidates": [{"id": "d_old", "type": "swing", "bbox_pdf_xyxy": [0.0, 0.0, 10.0, 10.0]}],
    }
    new_candidates = [
        {"id": "d_new", "type": "swing", "bbox_pdf_xyxy": [100.0, 100.0, 110.0, 110.0], "legacy_ids": []}
    ]
    rep = _attach_legacy_ids_from_previous_doors_json(new_candidates=new_candidates, prev_doors_data=prev_doors)
    assert rep.get("matched") == 0
    assert new_candidates[0].get("legacy_ids") == []

