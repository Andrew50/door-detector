"""Regression test: resolve reweighter model paths relative to a base dir.

This protects against a common failure mode:
- `configs/door_rules.json` references `models/...`
- the process is launched from a different working directory

In that case the model exists, but a naive `Path("models/...").exists()` check fails.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from door_detector.doors.detect import detect_doors


def _quarter_arc_bezier(cx: float, cy: float, r: float) -> dict:
    # Quarter-circle approximation (0° -> 90°).
    k = r * 0.5522847498307936
    return {
        "p0": {"x": cx + r, "y": cy},
        "p1": {"x": cx + r, "y": cy + k},
        "p2": {"x": cx + k, "y": cy + r},
        "p3": {"x": cx, "y": cy + r},
    }


def test_reweighter_path_resolution_uses_existing_model() -> None:
    cx, cy = 100.0, 100.0
    r = 40.0

    # Leaf line intentionally too short for strict len_ratio but valid for pool.
    tip_x = cx + r * 0.35  # len_ratio=0.35 < strict min_length_ratio (0.45)
    primitives = {
        "lines": [{"p0": {"x": cx, "y": cy}, "p1": {"x": tip_x, "y": cy}}],
        "beziers": [_quarter_arc_bezier(cx, cy, r)],
    }
    meta = {"id": "t", "mode": "vector"}

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        model_path = base / "models" / "reweighter_swing_v1.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # A trivial model that assigns high probability to all candidates.
        model_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "model_type": "logreg",
                    "feature_order": [
                        "rmse",
                        "radius",
                        "angle_span",
                        "hinge_dist",
                        "len_ratio",
                        "center_dist",
                        "radial_angle_deg",
                        "tip_to_arc_dist",
                    ],
                    "scaler": {"type": "zscore", "mean": [0] * 8, "std": [1] * 8},
                    "weights": [0] * 8,
                    "bias": 10.0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        cfg = {
            "_door_detector_base_dir": str(base),
            "mode_policy": {"vector": "vector_rules", "hybrid": "vector_rules", "scan": "empty_with_message"},
            "reweighters": {"swing": "models/reweighter_swing_v1.json"},
            "output": {
                "min_candidate_confidence": 0.0,
                "min_confidence_after_reweight": 0.90,
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

        det = detect_doors(primitives, meta, cfg)
        assert isinstance(det, dict)
        assert det.get("candidates"), "expected a candidate pool"
        assert det.get("doors"), "expected reweighter-driven selection to keep a door"


def main() -> int:
    test_reweighter_path_resolution_uses_existing_model()
    print("✓ reweighter path resolution test passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

