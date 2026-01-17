"""PDF↔pixel coordinate transformations."""

import math
from typing import Any, Callable, Dict, Tuple

import fitz  # PyMuPDF


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
    # PyMuPDF exposes multiple relevant rectangles / coordinate spaces:
    # - `page.cropbox` / `page.mediabox`: *unrotated* page space (where `page.get_drawings()`
    #   coordinates land for rotated pages).
    # - `page.rect`: rotated / display space.
    cropbox = page.cropbox
    mediabox = page.mediabox
    page_rect = page.rect

    # IMPORTANT: Rendering uses `fitz.Matrix(scale, scale).prerotate(page.rotation)`,
    # which rotates around the origin. We derive the coordinate transform from the exact
    # same matrix and then shift into the pixmap's (0,0)-based coordinate system.
    base = fitz.Matrix(scale, scale).prerotate(rotation_deg)

    # Transform the page rect to find the pixel-space bounding box (may include negatives
    # depending on rotation), then translate so the bbox starts at (0, 0).
    bbox = page_rect * base
    shift_x = -bbox.x0
    shift_y = -bbox.y0

    # `page.get_drawings()` coordinates for rotated PDFs are in *unrotated* space (cropbox).
    # To align vectors with the rendered pixmap, we need:
    #   (unrotated coords) --[page.rotation_matrix]--> (page.rect / rotated coords)
    #                     --[base + shift]-----------> (pixmap coords)
    #
    # We pre-compose these two affine transforms into a single 2D affine:
    # M = rotation_matrix ∘ (base+shift)  (applied in that order to points).
    rot = page.rotation_matrix  # unrotated -> rotated/page.rect coords

    r_a, r_b, r_c, r_d, r_e, r_f = float(rot.a), float(rot.b), float(rot.c), float(rot.d), float(rot.e), float(rot.f)
    b_a, b_b, b_c, b_d = float(base.a), float(base.b), float(base.c), float(base.d)
    b_e, b_f = float(base.e + shift_x), float(base.f + shift_y)

    # Compose M = R then B  (i.e., p -> p*R -> (p*R)*B).
    pdf_to_pix_affine = [
        b_a * r_a + b_c * r_b,               # a
        b_b * r_a + b_d * r_b,               # b
        b_a * r_c + b_c * r_d,               # c
        b_b * r_c + b_d * r_d,               # d
        b_a * r_e + b_c * r_f + b_e,         # e
        b_b * r_e + b_d * r_f + b_f,         # f
    ]

    computed_pix_width = int(round(bbox.width))
    computed_pix_height = int(round(bbox.height))

    if pix_width is None or pix_height is None:
        pix_width = computed_pix_width
        pix_height = computed_pix_height

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
        (b * f - d * e) / det,
        (c * e - a * f) / det,
    ]

    transform_dict = {
        "dpi": dpi,
        "scale": scale,
        "page_rect": {"x0": float(page_rect.x0), "y0": float(page_rect.y0), "x1": float(page_rect.x1), "y1": float(page_rect.y1)},
        "cropbox": {"x0": float(cropbox.x0), "y0": float(cropbox.y0), "x1": float(cropbox.x1), "y1": float(cropbox.y1)},
        "mediabox": {"x0": float(mediabox.x0), "y0": float(mediabox.y0), "x1": float(mediabox.x1), "y1": float(mediabox.y1)},
        "rotation_matrix": [r_a, r_b, r_c, r_d, r_e, r_f],
        "rotation_deg": rotation_deg,
        "pdf_to_pix_affine": pdf_to_pix_affine,
        "pix_to_pdf_affine": pix_to_pdf_affine,
        "computed_pix_width": computed_pix_width,
        "computed_pix_height": computed_pix_height,
        "pix_width": int(round(pix_width)),
        "pix_height": int(round(pix_height)),
    }

    # Create transformation functions
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
    """
    Validate transform by checking round-trip accuracy.

    Args:
        pdf_to_pix: PDF to pixel transformation function
        pix_to_pdf: Pixel to PDF transformation function
        primitives: Dictionary of extracted primitives
        num_samples: Number of random points to test

    Returns:
        Tuple of (is_valid, max_error)
    """
    import random

    # Collect sample points from primitives
    sample_points = []

    # Sample from lines
    for line in primitives.get("lines", [])[:num_samples]:
        sample_points.append((line["p0"]["x"], line["p0"]["y"]))
        sample_points.append((line["p1"]["x"], line["p1"]["y"]))

    # Sample from beziers
    for bezier in primitives.get("beziers", [])[:num_samples]:
        sample_points.append((bezier["p0"]["x"], bezier["p0"]["y"]))

    # Sample from rects
    for rect in primitives.get("rects", [])[:num_samples]:
        r = rect["rect"]
        sample_points.append((r["x0"], r["y0"]))
        sample_points.append((r["x1"], r["y1"]))

    if len(sample_points) == 0:
        # No primitives to validate against
        return True, 0.0

    # Randomly sample up to num_samples points
    random.shuffle(sample_points)
    sample_points = sample_points[:num_samples]

    max_error = 0.0
    for pdf_x, pdf_y in sample_points:
        # Round trip: PDF → pixel → PDF
        pix_x, pix_y = pdf_to_pix(pdf_x, pdf_y)
        pdf_x2, pdf_y2 = pix_to_pdf(pix_x, pix_y)

        # Compute error
        error = math.sqrt((pdf_x - pdf_x2) ** 2 + (pdf_y - pdf_y2) ** 2)
        max_error = max(max_error, error)

    # Check if error is acceptable (< 1e-3 in PDF units)
    is_valid = max_error < 1e-3

    return is_valid, max_error

