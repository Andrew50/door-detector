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
    rotation_deg = page.rotation
    page_rect = page.rect

    # Get page dimensions in PDF coordinates
    x0, y0 = page_rect.x0, page_rect.y0
    x1, y1 = page_rect.x1, page_rect.y1
    width_pdf = x1 - x0
    height_pdf = y1 - y0

    # Use provided pixel dimensions or compute from scale
    if pix_width is None or pix_height is None:
        # When PyMuPDF renders with prerotate, it computes the bounding box
        # For simplicity, if no dimensions provided, use simple calculation
        # (This should match what PyMuPDF produces for rotation=0)
        if rotation_deg == 0:
            pix_width = int(round(width_pdf * scale))
            pix_height = int(round(height_pdf * scale))
        else:
            # For rotated pages, PyMuPDF computes the bounding box
            # We'll use a temporary pixmap to get the actual dimensions
            temp_matrix = fitz.Matrix(scale, scale).prerotate(rotation_deg)
            temp_pixmap = page.get_pixmap(matrix=temp_matrix, alpha=False)
            pix_width = temp_pixmap.width
            pix_height = temp_pixmap.height
            temp_pixmap = None  # Free memory

    # Convert rotation to radians
    rotation_rad = math.radians(rotation_deg)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)

    # Build transformation matrix
    # PyMuPDF's prerotate applies rotation before scaling
    # The matrix order is: scale * rotation
    # We need to match PyMuPDF's coordinate system
    
    # For rotation, compute the bounding box of rotated page
    if rotation_deg == 0:
        # No rotation - simple scale and translate
        pdf_to_pix_affine = [
            scale,
            0.0,
            0.0,
            scale,
            -x0 * scale,
            -y0 * scale,
        ]
    else:
        # Rotate corners to find bounding box (matching PyMuPDF's prerotate behavior)
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        rotated_corners = []
        for cx, cy in corners:
            dx = cx - center_x
            dy = cy - center_y
            # Rotate around center
            rx = dx * cos_r - dy * sin_r
            ry = dx * sin_r + dy * cos_r
            rotated_corners.append((rx + center_x, ry + center_y))
        
        # Find bounding box of rotated page
        min_x = min(p[0] for p in rotated_corners)
        max_x = max(p[0] for p in rotated_corners)
        min_y = min(p[1] for p in rotated_corners)
        max_y = max(p[1] for p in rotated_corners)
        
        # Build transformation: rotate around center, then scale, then translate to pixel origin
        # The translation accounts for the bounding box offset
        pdf_to_pix_affine = [
            scale * cos_r,
            scale * sin_r,
            -scale * sin_r,
            scale * cos_r,
            -min_x * scale,
            -min_y * scale,
        ]

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
        "page_rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "rotation_deg": rotation_deg,
        "pdf_to_pix_affine": pdf_to_pix_affine,
        "pix_to_pdf_affine": pix_to_pdf_affine,
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

