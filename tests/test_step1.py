"""Test script for Step 1 pipeline."""

import json
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def validate_artifacts(output_dir: Path) -> Tuple[bool, List[str]]:
    """
    Test that all required artifacts exist and are valid.

    Returns:
        Tuple of (all_passed, list_of_errors)
    """
    errors = []
    page_size = None
    primitives_data = None
    required_files = [
        "page.png",
        "primitives.json",
        "transform.json",
        "meta.json",
    ]
    optional_files = [
        # Produced by step1 unless `--no-debug-overlay` is passed.
        "debug_overlay.png",
    ]

    # Check all required files exist
    for filename in required_files:
        filepath = output_dir / filename
        if not filepath.exists():
            errors.append(f"Missing required file: {filename}")
        else:
            # Validate file contents
            if filename.endswith(".png"):
                try:
                    img = Image.open(filepath)
                    img.verify()
                    if filename == "page.png":
                        page_size = (img.width, img.height)
                        print(f"  ✓ {filename}: {img.width}x{img.height} pixels")
                    else:
                        print(f"  ✓ {filename}: Valid image")
                except Exception as e:
                    errors.append(f"Invalid image {filename}: {e}")

            elif filename.endswith(".json"):
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    print(f"  ✓ {filename}: Valid JSON")

                    # Validate specific schemas
                    if filename == "primitives.json":
                        primitives_data = data
                        if not isinstance(data, dict):
                            errors.append("primitives.json: Expected dict")
                        else:
                            required_keys = ["lines", "beziers", "rects", "stats"]
                            for key in required_keys:
                                if key not in data:
                                    errors.append(f"primitives.json: Missing key '{key}'")
                            print(f"    - Lines: {len(data.get('lines', []))}")
                            print(f"    - Beziers: {len(data.get('beziers', []))}")
                            print(f"    - Rects: {len(data.get('rects', []))}")

                    elif filename == "transform.json":
                        required_keys = [
                            "dpi",
                            "scale",
                            "page_rect",
                            "rotation_deg",
                            "pdf_to_pix_affine",
                            "pix_to_pdf_affine",
                            "pix_width",
                            "pix_height",
                        ]
                        for key in required_keys:
                            if key not in data:
                                errors.append(f"transform.json: Missing key '{key}'")
                        # Validate affine matrices
                        for aff_key in ["pdf_to_pix_affine", "pix_to_pdf_affine"]:
                            if aff_key in data:
                                aff = data[aff_key]
                                if not isinstance(aff, list) or len(aff) != 6:
                                    errors.append(
                                        f"transform.json: {aff_key} must be a list of 6 numbers"
                                    )
                        print(f"    - DPI: {data.get('dpi')}")
                        print(f"    - Pixel size: {data.get('pix_width')}x{data.get('pix_height')}")

                    elif filename == "meta.json":
                        required_keys = ["id", "source_pdf", "dpi", "mode", "stats"]
                        for key in required_keys:
                            if key not in data:
                                errors.append(f"meta.json: Missing key '{key}'")
                        mode = data.get("mode")
                        if mode not in ["scan", "vector", "hybrid"]:
                            errors.append(f"meta.json: Invalid mode '{mode}'")
                        print(f"    - Mode: {mode}")
                        print(f"    - Total time: {data.get('stats', {}).get('total_ms', 0):.1f}ms")

                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON {filename}: {e}")
                except Exception as e:
                    errors.append(f"Error reading {filename}: {e}")

    # Check optional files (do not fail if missing)
    for filename in optional_files:
        filepath = output_dir / filename
        if not filepath.exists():
            print(f"  INFO: {filename}: (optional) not present")
            continue
        try:
            img = Image.open(filepath)
            img.verify()
            print(f"  ✓ {filename}: Valid image")
        except Exception as e:
            errors.append(f"Invalid optional image {filename}: {e}")

    # Additional validation: primitives should lie within the rendered page image.
    # If they do not, downstream detections (bbox_xyxy) will be off-screen.
    if page_size and isinstance(primitives_data, dict):
        w, h = page_size
        tol = 5.0  # pixels of tolerance for float rounding

        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        def upd(x: float, y: float) -> None:
            nonlocal min_x, min_y, max_x, max_y
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        for line in primitives_data.get("lines", []):
            upd(line["p0"]["x"], line["p0"]["y"])
            upd(line["p1"]["x"], line["p1"]["y"])
        for bez in primitives_data.get("beziers", []):
            upd(bez["p0"]["x"], bez["p0"]["y"])
            upd(bez["p1"]["x"], bez["p1"]["y"])
            upd(bez["p2"]["x"], bez["p2"]["y"])
            upd(bez["p3"]["x"], bez["p3"]["y"])
        for rect in primitives_data.get("rects", []):
            r = rect["rect"]
            upd(r["x0"], r["y0"])
            upd(r["x1"], r["y1"])

        if min_x != float("inf"):
            print(f"  ✓ primitives bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
            if min_x < -tol or min_y < -tol or max_x > (w + tol) or max_y > (h + tol):
                errors.append(
                    "primitives.json: Primitive coordinates fall outside page.png bounds "
                    f"(page={w}x{h}, bounds=[{min_x:.1f},{min_y:.1f},{max_x:.1f},{max_y:.1f}]). "
                    "This usually indicates a PDF↔pixel transform mismatch (often on rotated pages)."
                )

    return len(errors) == 0, errors


def test_step1_artifacts(output_dir: Path) -> None:
    """Pytest wrapper around the script-style validator."""
    ok, errors = validate_artifacts(output_dir)
    assert ok, "Step1 artifacts validation failed:\n- " + "\n- ".join(errors)


def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python tests/test_step1.py <output_dir>")
        print("  Example: python tests/test_step1.py artifacts/test_output")
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)

    print(f"Testing artifacts in: {output_dir}")
    print()

    all_passed, errors = validate_artifacts(output_dir)

    print()
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Tests failed with the following errors:")
        for error in errors:
            print(f"  - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

