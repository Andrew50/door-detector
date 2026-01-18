"""PDF↔pixel coordinate transformations."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

import fitz  # PyMuPDF


def get_render_matrix(page: fitz.Page, dpi: int) -> fitz.Matrix:
    """Return the exact matrix used to rasterize the page at `dpi`.

    This must match `door_detector.pdf.render.render_page()` so that vector primitives
    transformed via `compute_transform()` align with `page.png`.
    """
    scale = dpi / 72.0
    # IMPORTANT:
    # In PyMuPDF, `page.get_pixmap()` already accounts for `page.rotation` as part
    # of the page geometry. Applying an additional prerotate here can double-apply
    # rotation (e.g. 90° → 180°), causing transform/pixmap mismatches.
    return fitz.Matrix(scale, scale)


def compute_transform(
    page: fitz.Page, dpi: int = 400, pix_width: int | None = None, pix_height: int | None = None
) -> Tuple[Dict[str, Any], Callable, Callable]:
    """
    Compute PDF↔pixel transformation matrices and functions.

    Args:
        page: PyMuPDF page object
        dpi: DPI used for rendering
        pix_width: Actual pixel width of rendered image (if None, will be computed)
        pix_height: Actual pixel height of rendered image (if None, will be computed)

    Returns:
        Tuple of (transform_dict, pdf_to_pix_func, pix_to_pdf_func)
    """
    scale = dpi / 72.0
    rotation_deg = int(page.rotation) % 360

    # PyMuPDF rasterizes using a render matrix. In our setup the rendered pixmap
    # dimensions match the rotated page geometry automatically, so mapping
    # `page.get_drawings()` coordinates into pixel space is a simple scale (no
    # additional prerotate here). We still translate so the transformed bbox
    # starts at (0, 0).
    cropbox = page.cropbox
    mediabox = page.mediabox
    page_rect = page.rect

    # IMPORTANT:
    # `page.get_pixmap(matrix=Matrix(scale,scale))` produces an image that reflects
    # `page.rotation` automatically, but `page.get_drawings()` coordinates are in
    # the *unrotated* page coordinate space. To map drawings into the rendered
    # pixel space, we must explicitly apply the page rotation here.
    base = fitz.Matrix(scale, scale).prerotate(rotation_deg)

    # Compute the transformed bbox and shift into a 0-based pixmap.
    bbox = cropbox * base
    shift_x = -float(bbox.x0)
    shift_y = -float(bbox.y0)

    pdf_to_pix_affine = [
        float(base.a),
        float(base.b),
        float(base.c),
        float(base.d),
        float(base.e) + shift_x,
        float(base.f) + shift_y,
    ]

    computed_pix_width = int(round(bbox.width))
    computed_pix_height = int(round(bbox.height))

    if pix_width is None or pix_height is None:
        pix_width = computed_pix_width
        pix_height = computed_pix_height
    else:
        # Guardrail: ensure our computed bbox agrees with the raster size.
        # Off-by-one can happen due to float math + rounding, so allow a small tolerance.
        tol = 2
        if abs(int(pix_width) - computed_pix_width) > tol or abs(int(pix_height) - computed_pix_height) > tol:
            raise ValueError(
                "Transform/pixmap size mismatch: "
                f"computed={computed_pix_width}x{computed_pix_height} "
                f"actual={int(pix_width)}x{int(pix_height)} "
                f"(rotation_deg={rotation_deg}, dpi={dpi})"
            )

    # Pixel to PDF transformation (inverse)
    a, b, c, d, e, f = pdf_to_pix_affine
    det = a * d - b * c
    if abs(det) < 1e-10:
        raise ValueError("Transformation matrix is singular")

    pix_to_pdf_affine = [
        d / det,
        -b / det,
        -c / det,
        a / det,
        (c * f - d * e) / det,
        (b * e - a * f) / det,
    ]

    transform_dict = {
        "dpi": dpi,
        "scale": scale,
        "page_rect": {"x0": float(page_rect.x0), "y0": float(page_rect.y0), "x1": float(page_rect.x1), "y1": float(page_rect.y1)},
        "cropbox": {"x0": float(cropbox.x0), "y0": float(cropbox.y0), "x1": float(cropbox.x1), "y1": float(cropbox.y1)},
        "mediabox": {"x0": float(mediabox.x0), "y0": float(mediabox.y0), "x1": float(mediabox.x1), "y1": float(mediabox.y1)},
        "rotation_deg": rotation_deg,
        "pdf_to_pix_affine": pdf_to_pix_affine,
        "pix_to_pdf_affine": pix_to_pdf_affine,
        "computed_pix_width": computed_pix_width,
        "computed_pix_height": computed_pix_height,
        "pix_width": int(round(pix_width)),
        "pix_height": int(round(pix_height)),
    }

    def pdf_to_pix(x: float, y: float) -> Tuple[float, float]:
        """Convert PDF coordinates to pixel coordinates."""
        a, b, c, d, e, f = pdf_to_pix_affine
        px = a * x + c * y + e
        py = b * x + d * y + f
        return px, py

    def pix_to_pdf(px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates to PDF coordinates."""
        a, b, c, d, e, f = pix_to_pdf_affine
        x = a * px + c * py + e
        y = b * px + d * py + f
        return x, y

    return transform_dict, pdf_to_pix, pix_to_pdf


def validate_transform(
    pdf_to_pix: Callable, pix_to_pdf: Callable, primitives: Dict[str, Any], num_samples: int = 20
) -> Tuple[bool, float]:
    """Validate transform by checking round-trip accuracy."""
    import random

    # Collect sample points from primitives
    sample_points = []

    for line in primitives.get("lines", [])[:num_samples]:
        sample_points.append((line["p0"]["x"], line["p0"]["y"]))
        sample_points.append((line["p1"]["x"], line["p1"]["y"]))

    for bezier in primitives.get("beziers", [])[:num_samples]:
        sample_points.append((bezier["p0"]["x"], bezier["p0"]["y"]))

    for rect in primitives.get("rects", [])[:num_samples]:
        r = rect["rect"]
        sample_points.append((r["x0"], r["y0"]))
        sample_points.append((r["x1"], r["y1"]))

    if len(sample_points) == 0:
        return True, 0.0

    random.shuffle(sample_points)
    sample_points = sample_points[:num_samples]

    max_error = 0.0
    for pdf_x, pdf_y in sample_points:
        pix_x, pix_y = pdf_to_pix(pdf_x, pdf_y)
        pdf_x2, pdf_y2 = pix_to_pdf(pix_x, pix_y)
        error = math.sqrt((pdf_x - pdf_x2) ** 2 + (pdf_y - pdf_y2) ** 2)
        max_error = max(max_error, error)

    is_valid = max_error < 1e-3
    return is_valid, max_error

