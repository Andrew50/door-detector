"""PDF rendering to raster images."""

import io
import time
from pathlib import Path
from typing import Tuple

import fitz  # PyMuPDF
from PIL import Image


def render_page(
    page: fitz.Page, dpi: int = 400, output_path: Path | None = None
) -> Tuple[Image.Image, float]:
    """
    Render a PDF page to a PNG image at the specified DPI.

    Args:
        page: PyMuPDF page object
        dpi: Target DPI for rendering (default 400)
        output_path: Optional path to save the PNG

    Returns:
        Tuple of (PIL Image, render_time_ms)
    """
    start_time = time.time()

    # Calculate scale factor (PDF uses 72 DPI)
    scale = dpi / 72.0

    # Create transformation matrix accounting for page rotation
    matrix = fitz.Matrix(scale, scale).prerotate(page.rotation)

    # Render to pixmap
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    # Convert to PIL Image
    img_data = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_data))

    render_time_ms = (time.time() - start_time) * 1000

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")

    return img, render_time_ms


def load_pdf_page(pdf_path: Path, page_index: int = 0) -> Tuple[fitz.Document, fitz.Page]:
    """
    Load a PDF and return the specified page.

    Args:
        pdf_path: Path to PDF file
        page_index: Page index (default 0)

    Returns:
        Tuple of (document, page)

    Raises:
        ValueError: If PDF is invalid or page doesn't exist
    """
    doc = fitz.open(pdf_path)

    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"PDF {pdf_path} has no pages")

    if page_index >= doc.page_count:
        doc.close()
        raise ValueError(
            f"Page index {page_index} out of range (PDF has {doc.page_count} pages)"
        )

    page = doc[page_index]

    # Validate page has non-zero size
    rect = page.rect
    if rect.width <= 0 or rect.height <= 0:
        doc.close()
        raise ValueError(f"Page {page_index} has invalid size: {rect}")

    return doc, page

