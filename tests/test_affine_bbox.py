from __future__ import annotations

from door_detector.pdf.affine import apply_affine_bbox_xyxy


def test_apply_affine_bbox_xyxy_identity() -> None:
    m = [1, 0, 0, 1, 0, 0]
    bb = [10, 20, 30, 40]
    out = apply_affine_bbox_xyxy(m, bb)
    assert out == [10, 20, 30, 40]


def test_apply_affine_bbox_xyxy_translate() -> None:
    m = [1, 0, 0, 1, 5, -7]
    bb = [10, 20, 30, 40]
    out = apply_affine_bbox_xyxy(m, bb)
    assert out == [15, 13, 35, 33]


def test_apply_affine_bbox_xyxy_rotate_90_about_origin() -> None:
    # Rotation 90° CCW about origin: (x,y)->(-y,x)
    m = [0, 1, -1, 0, 0, 0]
    bb = [10, 20, 30, 40]
    # Corners: (10,20)->(-20,10), (10,40)->(-40,10), (30,20)->(-20,30), (30,40)->(-40,30)
    out = apply_affine_bbox_xyxy(m, bb)
    assert out == [-40, 10, -20, 30]

