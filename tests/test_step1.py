"""Test script for Step 1 pipeline."""

import json
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def test_artifacts(output_dir: Path) -> Tuple[bool, List[str]]:
    """
    Test that all required artifacts exist and are valid.

    Returns:
        Tuple of (all_passed, list_of_errors)
    """
    errors = []
    required_files = [
        "page.png",
        "primitives.json",
        "transform.json",
        "meta.json",
        "debug_overlay.png",
    ]

    # Check all files exist
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

    return len(errors) == 0, errors


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

    all_passed, errors = test_artifacts(output_dir)

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

