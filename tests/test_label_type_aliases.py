from __future__ import annotations

import json
from pathlib import Path

from door_detector.reweight_fit import fit_reweighter
from door_detector.ui.labels import coerce_confirmed_by_type, coerce_rejected_by_type


def test_coerce_by_type_normalizes_legacy_keys() -> None:
    cbt = coerce_confirmed_by_type(
        {
            # spacing + plural
            "double doors": ["d1"],
            # underscore plural
            "double_doors": ["d2"],
            # hyphen + suffix
            "bi-fold door": ["b1"],
            # underscore suffix
            "bi_fold_door": ["b2"],
        }
    )
    assert "d1" in cbt["double"]
    assert "d2" in cbt["double"]
    # Bifold labels are mapped to double.
    assert "b1" in cbt["double"]
    assert "b2" in cbt["double"]

    rbt = coerce_rejected_by_type({"pocket doors": ["p1"]})
    assert "p1" in rbt["pocket"]


def test_fit_reweighter_normalizes_label_type_keys(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    sample_dir = artifacts_root / "library" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Two double candidates so we can have 1 positive + 1 negative.
    doors = {
        "candidates": [
            {
                "id": "d_pos",
                "type": "double",
                "features": {"center_dist": 0.2, "radius_ratio": 1.0, "avg_swing_conf": 0.9, "pair_score": 0.9},
            },
            {
                "id": "d_neg",
                "type": "double",
                "features": {"center_dist": 2.0, "radius_ratio": 1.2, "avg_swing_conf": 0.4, "pair_score": 0.1},
            },
        ]
    }
    (sample_dir / "doors.json").write_text(json.dumps(doors, indent=2), encoding="utf-8")

    # Intentionally use non-canonical keys ("double doors") to ensure training still works.
    labels = {
        "schema_version": 4,
        "reviewed_at": "2026-01-01T00:00:00Z",
        "confirmed_by_type": {"double doors": ["d_pos"]},
        "rejected_by_type": {"double doors": ["d_neg"]},
        "deleted_ids": [],
        "manual_additions": [],
        "unmatched_manual_boxes": [],
    }
    (sample_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    models_dir = tmp_path / "models"
    report = fit_reweighter(
        artifacts_root,
        models_dir,
        min_samples=2,
        min_pos=1,
        min_neg=1,
    )

    assert isinstance(report, dict)
    by_type = report.get("by_type")
    assert isinstance(by_type, dict)
    tr_double = by_type.get("double")
    assert isinstance(tr_double, dict)
    assert tr_double.get("status") == "trained"
    assert int(tr_double.get("num_pos") or 0) == 1
    assert int(tr_double.get("num_neg") or 0) == 1
