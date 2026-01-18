"""Regression test: rotated PDFs should produce in-bounds pixel primitives.

This catches mismatches between the rasterization matrix and the PDF→pixel
transform used to map vector primitives into pixel space.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

from door_detector.step1_pipeline import process_pdf
from tests.test_step1 import test_artifacts


def _make_rotated_pdf(path: Path) -> None:
    doc = fitz.open()
    try:
        page = doc.new_page(width=400, height=300)
        # Rotate the page to exercise prerotate handling.
        page.set_rotation(90)

        # Draw a few vector primitives.
        # Use coordinates that remain comfortably inside the cropbox.
        page.draw_line(fitz.Point(50, 50), fitz.Point(350, 50), color=(0, 0, 0), width=1)
        page.draw_rect(fitz.Rect(80, 80, 160, 140), color=(0, 0, 0), width=1)
        page.draw_bezier(
            fitz.Point(200, 200),
            fitz.Point(260, 180),
            fitz.Point(300, 240),
            fitz.Point(340, 200),
            color=(0, 0, 0),
            width=1,
        )

        doc.save(str(path))
    finally:
        doc.close()


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pdf_path = td_path / "rotated.pdf"
        out_dir = td_path / "out"

        _make_rotated_pdf(pdf_path)
        process_pdf(pdf_path=pdf_path, output_dir=out_dir, dpi=200, page_index=0, enable_debug_overlay=False)

        ok, errors = test_artifacts(out_dir)
        if ok:
            print("✓ Rotation transform regression test passed!")
            return 0
        print("✗ Rotation transform regression test failed:")
        for e in errors:
            print(f"  - {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

