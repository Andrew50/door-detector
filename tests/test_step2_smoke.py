"""Smoke test: Step 1 + Step 2 run end-to-end on a generated PDF.

This repo's "tests" are intentionally runnable as plain scripts (see docs/TESTING.md),
but this file is also compatible with pytest if you have it installed.
"""

from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

import fitz  # PyMuPDF

from door_detector.step1_pipeline import process_pdf
from door_detector.step2_pipeline import run_step2


def _make_minimal_swing_door_pdf(path: Path) -> None:
    """Create a PDF containing a single swing-door glyph (arc + leaf)."""
    # Use PDF points; Step 1 converts to pixels via DPI scaling.
    cx, cy = 100.0, 100.0
    r = 10.0  # -> ~55 px at 400 DPI (within config radius bounds)
    k = r * 0.5522847498307936  # 4/3*(sqrt(2)-1) for quarter-circle approximation

    arc_start = fitz.Point(cx + r, cy)      # 0°
    arc_end = fitz.Point(cx, cy + r)        # 90°
    c1 = fitz.Point(cx + r, cy + k)
    c2 = fitz.Point(cx + k, cy + r)

    doc = fitz.open()
    try:
        page = doc.new_page(width=200, height=200)
        # Arc (Bezier)
        page.draw_bezier(arc_start, c1, c2, arc_end, color=(0, 0, 0), width=1)
        # Leaf line: hinge at center, tip exactly at the arc start (min_hinge_dist=0)
        page.draw_line(fitz.Point(cx, cy), arc_start, color=(0, 0, 0), width=1)
        doc.save(str(path))
    finally:
        doc.close()


def test_step2_smoke() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pdf_path = td_path / "swing_door.pdf"
        out_dir = td_path / "artifacts"
        cfg = Path("configs/door_rules.json")

        _make_minimal_swing_door_pdf(pdf_path)

        process_pdf(pdf_path=pdf_path, output_dir=out_dir, dpi=400, page_index=0, enable_debug_overlay=False)
        run_step2(artifacts_dir=out_dir, config_path=cfg, output_dir=out_dir)

        doors_json = out_dir / "doors.json"
        overlay_png = out_dir / "doors_overlay.png"

        assert doors_json.exists()
        assert overlay_png.exists()

        data = json.loads(doors_json.read_text(encoding="utf-8"))
        assert "doors" in data
        assert "candidates" in data
        assert isinstance(data["doors"], list)
        assert isinstance(data["candidates"], list)
        # Candidates must contain enough structure for snapping + training.
        if data["candidates"]:
            c0 = data["candidates"][0]
            assert "id" in c0
            assert "bbox_xyxy" in c0
            assert "confidence" in c0
            assert "features" in c0
        # This PDF is constructed to satisfy the strict swing-door thresholds.
        assert len(data["doors"]) >= 1


def main() -> int:
    try:
        test_step2_smoke()
    except Exception as e:
        print(f"✗ Step 2 smoke test failed: {e}", file=sys.stderr)
        raise
    print("✓ Step 2 smoke test passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


