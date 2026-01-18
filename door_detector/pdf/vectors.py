"""Extract and normalize vector primitives from PDF pages."""

from __future__ import annotations

import time
from typing import Any, Dict

import fitz  # PyMuPDF


def extract_primitives(page: fitz.Page) -> Dict[str, Any]:
    """Extract vector primitives from a PDF page and normalize them."""
    start_time = time.time()

    lines = []
    beziers = []
    rects = []

    drawings = page.get_drawings()

    for drawing in drawings:
        # Capture drawing-level styling (PyMuPDF applies these to the items).
        # Note: get_drawings() structure can vary slightly across PyMuPDF versions,
        # so keep this defensive and JSON-serializable.
        try:
            stroke_width = float(drawing.get("width") or drawing.get("linewidth") or 1.0)
        except Exception:
            stroke_width = 1.0

        dash_pattern: list[float] = []
        try:
            d = drawing.get("dashes")
            if isinstance(d, (list, tuple)):
                dash_pattern = [float(x) for x in d if x is not None]
            elif isinstance(d, str) and d.strip():
                # Some PyMuPDF versions expose dashes as a string like: "[ 0 3 ] 0"
                # where the bracketed numbers are the dash pattern and the trailing
                # number is the dash phase. Parse numbers defensively.
                nums: list[float] = []
                for tok in d.replace("[", " ").replace("]", " ").split():
                    try:
                        nums.append(float(tok))
                    except Exception:
                        continue
                # Heuristic: if it looks like "pattern + phase", drop the last number.
                if len(nums) >= 3:
                    dash_pattern = nums[:-1]
                else:
                    dash_pattern = nums
        except Exception:
            dash_pattern = []

        is_dashed = bool(len(dash_pattern) >= 2 and any((float(x) or 0.0) > 0 for x in dash_pattern))

        stroke_color = None
        try:
            c = drawing.get("color")
            # Typically RGB floats in 0..1; keep raw value (JSON-able).
            if isinstance(c, (list, tuple)) and len(c) in (3, 4):
                stroke_color = [float(x) for x in c]
        except Exception:
            stroke_color = None

        items = drawing.get("items", [])
        for item in items:
            item_type = item[0]

            if item_type == "l":  # Line segment
                p0 = {"x": item[1][0], "y": item[1][1]}
                p1 = {"x": item[2][0], "y": item[2][1]}
                lines.append(
                    {
                        "p0": p0,
                        "p1": p1,
                        "stroke_width": stroke_width,
                        "dash_pattern": dash_pattern,
                        "is_dashed": is_dashed,
                        "stroke_color": stroke_color,
                    }
                )

            elif item_type == "c":  # Cubic Bezier curve
                p0 = {"x": item[1][0], "y": item[1][1]}
                p1 = {"x": item[2][0], "y": item[2][1]}
                p2 = {"x": item[3][0], "y": item[3][1]}
                p3 = {"x": item[4][0], "y": item[4][1]}
                beziers.append(
                    {
                        "p0": p0,
                        "p1": p1,
                        "p2": p2,
                        "p3": p3,
                        "stroke_width": stroke_width,
                        "dash_pattern": dash_pattern,
                        "is_dashed": is_dashed,
                        "stroke_color": stroke_color,
                    }
                )

            elif item_type == "re":  # Rectangle
                rect_coords = item[1]
                x0 = min(rect_coords[0], rect_coords[2])
                y0 = min(rect_coords[1], rect_coords[3])
                x1 = max(rect_coords[0], rect_coords[2])
                y1 = max(rect_coords[1], rect_coords[3])

                rects.append(
                    {
                        "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                        "stroke_width": stroke_width,
                        "dash_pattern": dash_pattern,
                        "is_dashed": is_dashed,
                        "stroke_color": stroke_color,
                    }
                )

    extract_time_ms = (time.time() - start_time) * 1000

    return {
        "lines": lines,
        "beziers": beziers,
        "rects": rects,
        "stats": {
            "num_lines": len(lines),
            "num_beziers": len(beziers),
            "num_rects": len(rects),
            "num_drawings": len(drawings),
            "extract_time_ms": extract_time_ms,
        },
    }


def apply_transform_to_primitives(primitives: Dict[str, Any], transform_func) -> Dict[str, Any]:
    """Apply a coordinate transformation to all primitive points."""
    transformed = {"lines": [], "beziers": [], "rects": []}

    for line in primitives["lines"]:
        p0_x, p0_y = transform_func(line["p0"]["x"], line["p0"]["y"])
        p1_x, p1_y = transform_func(line["p1"]["x"], line["p1"]["y"])
        transformed["lines"].append({**line, "p0": {"x": p0_x, "y": p0_y}, "p1": {"x": p1_x, "y": p1_y}})

    for bezier in primitives["beziers"]:
        p0_x, p0_y = transform_func(bezier["p0"]["x"], bezier["p0"]["y"])
        p1_x, p1_y = transform_func(bezier["p1"]["x"], bezier["p1"]["y"])
        p2_x, p2_y = transform_func(bezier["p2"]["x"], bezier["p2"]["y"])
        p3_x, p3_y = transform_func(bezier["p3"]["x"], bezier["p3"]["y"])
        transformed["beziers"].append(
            {
                **bezier,
                "p0": {"x": p0_x, "y": p0_y},
                "p1": {"x": p1_x, "y": p1_y},
                "p2": {"x": p2_x, "y": p2_y},
                "p3": {"x": p3_x, "y": p3_y},
            }
        )

    for rect in primitives["rects"]:
        r = rect["rect"]
        tx0, ty0 = transform_func(r["x0"], r["y0"])
        tx1, ty1 = transform_func(r["x1"], r["y1"])

        x0 = min(tx0, tx1)
        x1 = max(tx0, tx1)
        y0 = min(ty0, ty1)
        y1 = max(ty0, ty1)

        transformed["rects"].append({**rect, "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}})

    transformed["stats"] = primitives["stats"]
    return transformed

