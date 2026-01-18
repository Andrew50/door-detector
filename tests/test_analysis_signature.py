"""Regression test: analysis signature changes when inputs change."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from door_detector.signatures import compute_analysis_signature


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cfg = td_path / "config.json"
        model = td_path / "model.json"

        # 1) Config-only: changing config bytes should change signature.
        cfg.write_text(json.dumps({"schema_version": 1, "x": 1}, indent=2), encoding="utf-8")
        s1 = compute_analysis_signature(cfg)

        cfg.write_text(json.dumps({"schema_version": 1, "x": 2}, indent=2), encoding="utf-8")
        s2 = compute_analysis_signature(cfg)
        assert s1 != s2, "signature did not change when config content changed"

        # 2) Model included: changing model bytes should change signature even if config is stable.
        model.write_text(json.dumps({"schema_version": 1, "weights": [1, 2, 3]}, indent=2), encoding="utf-8")
        cfg.write_text(
            json.dumps({"schema_version": 1, "x": 1, "reweighter_path": str(model)}, indent=2),
            encoding="utf-8",
        )
        s3 = compute_analysis_signature(cfg)

        model.write_text(json.dumps({"schema_version": 1, "weights": [1, 2, 4]}, indent=2), encoding="utf-8")
        s4 = compute_analysis_signature(cfg)
        assert s3 != s4, "signature did not change when reweighter content changed"

        print("✓ Analysis signature regression test passed!")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

