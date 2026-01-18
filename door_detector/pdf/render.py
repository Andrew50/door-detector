"""PDF rendering to raster images."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Tuple

import fitz  # PyMuPDF
from PIL import Image

from door_detector.pdf.transforms import get_render_matrix

# Increase PIL pixel limit for large floor plans
Image.MAX_IMAGE_PIXELS = None


def render_page(page: fitz.Page, dpi: int = 400, output_path: Path | None = None) -> Tuple[Image.Image, float]:
    """Render a PDF page to a PNG image at the specified DPI."""
    start_time = time.time()

    # Render matrix must match `compute_transform()`.
    matrix = get_render_matrix(page, dpi)

    # Render to pixmap
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    # Convert to PIL Image
    img_data = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_data))

    render_time_ms = (time.time() - start_time) * 1000

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")

    return img, render_time_ms


def load_pdf_page(pdf_path: Path, page_index: int = 0) -> Tuple[fitz.Document, fitz.Page]:
    """Load a PDF and return the specified page."""
    doc = fitz.open(pdf_path)

    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"PDF {pdf_path} has no pages")

    if page_index >= doc.page_count:
        doc.close()
        raise ValueError(f"Page index {page_index} out of range (PDF has {doc.page_count} pages)")

    page = doc[page_index]

    rect = page.rect
    if rect.width <= 0 or rect.height <= 0:
        doc.close()
        raise ValueError(f"Page {page_index} has invalid size: {rect}")

    return doc, page

