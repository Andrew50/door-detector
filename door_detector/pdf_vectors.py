"""Extract and normalize vector primitives from PDF pages."""

import time
from typing import Any, Dict, List

import fitz  # PyMuPDF


def extract_primitives(page: fitz.Page) -> Dict[str, Any]:
    """
    Extract vector primitives from a PDF page and normalize them.

    Args:
        page: PyMuPDF page object

    Returns:
        Dictionary with normalized primitives:
        - lines: List of line segments
        - beziers: List of cubic Bezier curves
        - rects: List of rectangles
        - paths_raw: Raw path data for debugging
    """
    start_time = time.time()

    lines = []
    beziers = []
    rects = []
    paths_raw = []

    # Get all drawings from the page
    drawings = page.get_drawings()

    for drawing in drawings:
        # Store raw path for debugging
        raw_path = {
            "type": drawing.get("type", "unknown"),
            "rect": list(drawing.get("rect", [])) if drawing.get("rect") else None,
            "items": len(drawing.get("items", [])),
        }
        paths_raw.append(raw_path)

        # Extract style information
        stroke_width = drawing.get("width", 1.0)
        stroke_color = drawing.get("color", [0, 0, 0])
        fill_color = drawing.get("fill", None)
        dashes = drawing.get("dashes", None)

        # Process items in the drawing
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
                        "color": stroke_color,
                        "dashes": dashes,
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
                        "color": stroke_color,
                        "dashes": dashes,
                    }
                )

            elif item_type == "re":  # Rectangle
                rect_coords = item[1]
                # Re-normalize just in case fitz returns non-standard coords
                x0 = min(rect_coords[0], rect_coords[2])
                y0 = min(rect_coords[1], rect_coords[3])
                x1 = max(rect_coords[0], rect_coords[2])
                y1 = max(rect_coords[1], rect_coords[3])
                
                rects.append(
                    {
                        "rect": {
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                        },
                        "stroke_width": stroke_width,
                        "color": stroke_color,
                        "fill": fill_color,
                        "dashes": dashes,
                    }
                )

    extract_time_ms = (time.time() - start_time) * 1000

    return {
        "lines": lines,
        "beziers": beziers,
        "rects": rects,
        "paths_raw": paths_raw,
        "stats": {
            "num_lines": len(lines),
            "num_beziers": len(beziers),
            "num_rects": len(rects),
            "num_drawings": len(drawings),
            "extract_time_ms": extract_time_ms,
        },
    }


def apply_transform_to_primitives(
    primitives: Dict[str, Any], transform_func
) -> Dict[str, Any]:
    """
    Apply a coordinate transformation to all primitive points.

    Args:
        primitives: Dictionary of primitives from extract_primitives
        transform_func: Function that takes (x, y) and returns transformed (x, y)

    Returns:
        New dictionary with transformed primitives
    """
    transformed = {
        "lines": [],
        "beziers": [],
        "rects": [],
        "paths_raw": primitives["paths_raw"],  # Keep raw paths unchanged
    }

    # Transform lines
    for line in primitives["lines"]:
        p0_x, p0_y = transform_func(line["p0"]["x"], line["p0"]["y"])
        p1_x, p1_y = transform_func(line["p1"]["x"], line["p1"]["y"])
        transformed["lines"].append(
            {
                **line,
                "p0": {"x": p0_x, "y": p0_y},
                "p1": {"x": p1_x, "y": p1_y},
            }
        )

    # Transform Bezier curves
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

    # Transform rectangles
    for rect in primitives["rects"]:
        r = rect["rect"]
        tx0, ty0 = transform_func(r["x0"], r["y0"])
        tx1, ty1 = transform_func(r["x1"], r["y1"])
        
        # Re-normalize to ensure x0 <= x1 and y0 <= y1 after transformation
        x0 = min(tx0, tx1)
        x1 = max(tx0, tx1)
        y0 = min(ty0, ty1)
        y1 = max(ty0, ty1)
        
        transformed["rects"].append(
            {
                **rect,
                "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            }
        )

    # Copy stats
    transformed["stats"] = primitives["stats"]

    return transformed



