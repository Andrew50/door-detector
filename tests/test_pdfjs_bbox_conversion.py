from __future__ import annotations

from door_detector.pdf.affine import fitz_bbox_to_pdfjs_bbox_xyxy, pdfjs_bbox_to_fitz_bbox_xyxy


def test_pdfjs_bbox_conversion_centered_page() -> None:
    # CAD-style centered page: x spans negative→positive, y is centered in PDF coords.
    cropbox = {"x0": -1512.12, "y0": 0.0, "x1": 1512.12, "y1": 2160.0}

    # A bbox near the top of the page in fitz coords (y-down).
    bb_fitz = [100.0, 100.0, 200.0, 200.0]
    bb_pdf = fitz_bbox_to_pdfjs_bbox_xyxy(bb_fitz, cropbox=cropbox)

    # For a centered page of height 2160, PDF y should land in [-1080, 1080].
    assert -1080.0 <= bb_pdf[1] <= 1080.0
    assert -1080.0 <= bb_pdf[3] <= 1080.0

    # Inverse conversion should recover the original bbox (within float tolerance).
    bb_fitz2 = pdfjs_bbox_to_fitz_bbox_xyxy(bb_pdf, cropbox=cropbox)
    for a, b in zip(bb_fitz, bb_fitz2):
        assert abs(a - b) < 1e-6


def test_pdfjs_bbox_conversion_normal_page() -> None:
    # Typical PDF page: x>=0, y in [0, H] in PDF coords.
    cropbox = {"x0": 0.0, "y0": 0.0, "x1": 612.0, "y1": 792.0}

    bb_fitz = [10.0, 20.0, 30.0, 40.0]
    bb_pdf = fitz_bbox_to_pdfjs_bbox_xyxy(bb_fitz, cropbox=cropbox)

    assert 0.0 <= bb_pdf[1] <= 792.0
    assert 0.0 <= bb_pdf[3] <= 792.0

    bb_fitz2 = pdfjs_bbox_to_fitz_bbox_xyxy(bb_pdf, cropbox=cropbox)
    for a, b in zip(bb_fitz, bb_fitz2):
        assert abs(a - b) < 1e-6

