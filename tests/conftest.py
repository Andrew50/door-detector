from __future__ import annotations

import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from door_detector.step1_pipeline import process_pdf


@pytest.fixture
def output_dir() -> Path:
    """Create a Step 1 artifacts directory for script-style artifact validation tests."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pdf_path = td_path / "minimal.pdf"
        out_dir = td_path / "artifacts"

        doc = fitz.open()
        try:
            page = doc.new_page(width=200, height=200)
            # Draw a few primitives to ensure non-empty primitives.json.
            page.draw_line(fitz.Point(20, 20), fitz.Point(180, 20), color=(0, 0, 0), width=1)
            page.draw_bezier(
                fitz.Point(20, 50),
                fitz.Point(60, 80),
                fitz.Point(140, 20),
                fitz.Point(180, 50),
                color=(0, 0, 0),
                width=1,
            )
            doc.save(str(pdf_path))
        finally:
            doc.close()

        process_pdf(pdf_path=pdf_path, output_dir=out_dir, dpi=200, page_index=0, enable_debug_overlay=False)
        yield out_dir

