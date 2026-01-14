"""Classify PDF pages as scan, vector, or hybrid."""

from typing import Any, Dict

import fitz  # PyMuPDF


def classify_page_mode(
    page: fitz.Page, primitives: Dict[str, Any], low_segment_threshold: int = 50, high_segment_threshold: int = 200
) -> Dict[str, Any]:
    """
    Classify a PDF page as scan, vector, or hybrid based on primitive density and image coverage.

    Args:
        page: PyMuPDF page object
        primitives: Dictionary of extracted primitives with stats
        low_segment_threshold: Low threshold for segment count (default 50)
        high_segment_threshold: High threshold for segment count (default 200)

    Returns:
        Dictionary with mode classification and statistics
    """
    # Count segments (lines + curves + rect edges)
    num_lines = primitives["stats"]["num_lines"]
    num_beziers = primitives["stats"]["num_beziers"]
    num_rects = primitives["stats"]["num_rects"]
    num_drawings = primitives["stats"]["num_drawings"]

    # Count rect edges as segments (4 per rect)
    num_segments = num_lines + num_beziers + (num_rects * 4)

    # Get embedded images and compute coverage
    images = page.get_images(full=True)
    num_images = len(images)

    # Compute image coverage
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    image_coverage = 0.0
    if num_images > 0 and page_area > 0:
        total_image_area = 0.0
        for img_info in images:
            xref = img_info[0]
            try:
                img_rects = page.get_image_rects(xref)
                for rect in img_rects:
                    img_area = rect.width * rect.height
                    total_image_area += img_area
            except Exception:
                # If we can't get image rects, skip this image
                pass

        image_coverage = min(total_image_area / page_area, 1.0)

    # Classification heuristic
    if image_coverage >= 0.60 and num_segments <= low_segment_threshold:
        mode = "scan"
    elif num_segments >= high_segment_threshold and image_coverage <= 0.20:
        mode = "vector"
    else:
        mode = "hybrid"

    return {
        "mode": mode,
        "stats": {
            "num_drawings": num_drawings,
            "num_segments": num_segments,
            "num_lines": num_lines,
            "num_beziers": num_beziers,
            "num_rects": num_rects,
            "num_images": num_images,
            "image_coverage": image_coverage,
            "page_area": page_area,
        },
    }



