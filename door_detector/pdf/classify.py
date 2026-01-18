"""Classify PDF pages as scan, vector, or hybrid."""

from __future__ import annotations

from typing import Any, Dict

import fitz  # PyMuPDF


def classify_page_mode(
    page: fitz.Page,
    primitives: Dict[str, Any],
    low_segment_threshold: int = 50,
    high_segment_threshold: int = 200,
) -> Dict[str, Any]:
    """Classify a PDF page as scan, vector, or hybrid based on primitives + image coverage."""
    num_lines = primitives["stats"]["num_lines"]
    num_beziers = primitives["stats"]["num_beziers"]
    num_rects = primitives["stats"]["num_rects"]
    num_drawings = primitives["stats"]["num_drawings"]

    num_segments = num_lines + num_beziers + (num_rects * 4)

    images = page.get_images(full=True)
    num_images = len(images)

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
                    total_image_area += rect.width * rect.height
            except Exception:
                pass

        image_coverage = min(total_image_area / page_area, 1.0)

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

