"""Smoke tests: Step 1 + Step 2 produce multi-type candidates.

These tests rely on PyMuPDF (fitz) and run end-to-end through artifacts.
"""

from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

import fitz  # PyMuPDF

from door_detector.step1_pipeline import process_pdf
from door_detector.step2_pipeline import run_step2


def _make_double_door_pdf(path: Path) -> None:
    """Create a PDF containing two nearby swing-door glyphs (should allow a double candidate)."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=300, height=200)
        # Two swing doors (quarter arcs + leaf lines)
        for cx in (100.0, 155.0):
            cy = 100.0
            # Larger radius so the centers are within the default pairing threshold
            # after DPI scaling (distance/radius ratio stays small).
            r = 25.0
            k = r * 0.5522847498307936
            arc_start = fitz.Point(cx + r, cy)
            arc_end = fitz.Point(cx, cy + r)
            c1 = fitz.Point(cx + r, cy + k)
            c2 = fitz.Point(cx + k, cy + r)
            page.draw_bezier(arc_start, c1, c2, arc_end, color=(0, 0, 0), width=1)
            page.draw_line(fitz.Point(cx, cy), arc_start, color=(0, 0, 0), width=1)
        doc.save(str(path))
    finally:
        doc.close()


def _make_pocket_door_pdf(path: Path) -> None:
    """Create a PDF containing a dashed track line (pocket candidate)."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=200, height=200)
        # A dashed line long enough to satisfy the default pocket thresholds after DPI scaling.
        page.draw_line(
            fitz.Point(40, 80),
            fitz.Point(140, 80),
            color=(0, 0, 0),
            width=1,
            dashes=[3, 3],
        )
        doc.save(str(path))
    finally:
        doc.close()


def _make_bifold_door_pdf(path: Path) -> None:
    """Create a PDF containing a simple zig-zag chain (bifold candidate)."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=220, height=200)
        pts = [fitz.Point(40, 120), fitz.Point(70, 120), fitz.Point(100, 150), fitz.Point(130, 120)]
        for i in range(len(pts) - 1):
            page.draw_line(pts[i], pts[i + 1], color=(0, 0, 0), width=1)
        doc.save(str(path))
    finally:
        doc.close()


def _run(pdf_path: Path, *, dpi: int = 400) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "artifacts"
        process_pdf(pdf_path=pdf_path, output_dir=out_dir, dpi=dpi, page_index=0, enable_debug_overlay=False)
        run_step2(artifacts_dir=out_dir, config_path=Path("configs/door_rules.json"), output_dir=out_dir)
        return json.loads((out_dir / "doors.json").read_text(encoding="utf-8"))


def test_step2_emits_double_candidate() -> None:
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "double.pdf"
        _make_double_door_pdf(pdf_path)
        data = _run(pdf_path)
        cands = data.get("candidates") or []
        # PDF-space bbox is required for the PDF.js viewer overlay.
        if cands:
            assert "bbox_pdf_xyxy" in (cands[0] or {})
        assert any(c.get("type") == "double" for c in cands), "expected a double candidate in doors.json"


def test_step2_emits_pocket_candidate_from_dashed_line() -> None:
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "pocket.pdf"
        _make_pocket_door_pdf(pdf_path)
        data = _run(pdf_path)
        cands = data.get("candidates") or []
        # PDF-space bbox is required for the PDF.js viewer overlay.
        if cands:
            assert "bbox_pdf_xyxy" in (cands[0] or {})
        assert any(c.get("type") == "pocket" for c in cands), "expected a pocket candidate in doors.json"


def test_step2_emits_bifold_candidate_from_zigzag() -> None:
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "bifold.pdf"
        _make_bifold_door_pdf(pdf_path)
        data = _run(pdf_path)
        cands = data.get("candidates") or []
        # PDF-space bbox is required for the PDF.js viewer overlay.
        if cands:
            assert "bbox_pdf_xyxy" in (cands[0] or {})
        assert any(c.get("type") == "bifold" for c in cands), "expected a bifold candidate in doors.json"


def main() -> int:
    try:
        test_step2_emits_double_candidate()
        test_step2_emits_pocket_candidate_from_dashed_line()
        test_step2_emits_bifold_candidate_from_zigzag()
    except Exception as e:
        print(f"✗ step2 multitype smoke tests failed: {e}", file=sys.stderr)
        raise
    print("✓ step2 multitype smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

