"""Step 1 pipeline: PDF → analysis-ready representation."""

import argparse
import sys
from pathlib import Path

from door_detector.artifacts import write_artifacts
from door_detector.pdf_render import load_pdf_page, render_page
from door_detector.pdf_vectors import apply_transform_to_primitives, extract_primitives
from door_detector.scan_classifier import classify_page_mode
from door_detector.step1_signature import compute_step1_signature
from door_detector.transforms import compute_transform, validate_transform


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 400,
    page_index: int = 0,
    enable_debug_overlay: bool = True,
) -> None:
    """
    Process a PDF page and generate analysis-ready artifacts.

    Args:
        pdf_path: Path to input PDF
        output_dir: Output directory for artifacts
        dpi: DPI for rendering (default 400)
        page_index: Page index to process (default 0)
        enable_debug_overlay: Whether to generate debug overlay (default True)
    """
    import time

    total_start = time.time()

    # Load PDF page
    doc, page = load_pdf_page(pdf_path, page_index)
    try:
        # Extract page ID from filename and page index to ensure uniqueness
        page_id = f"{pdf_path.stem}_p{page_index}"

        # Render to PNG
        image, render_time_ms = render_page(page, dpi=dpi)

        # Extract vector primitives
        extract_start = time.time()
        primitives = extract_primitives(page)
        extract_time_ms = (time.time() - extract_start) * 1000

        # Compute transform (using actual rendered image dimensions)
        transform_dict, pdf_to_pix, pix_to_pdf = compute_transform(
            page, dpi=dpi, pix_width=image.width, pix_height=image.height
        )

        # Apply transform to primitives (for pixel-space analysis)
        primitives_pix = apply_transform_to_primitives(primitives, pdf_to_pix)

        # Validate transform
        is_valid, max_error = validate_transform(pdf_to_pix, pix_to_pdf, primitives)
        if not is_valid:
            print(
                f"Warning: Transform validation failed (max error: {max_error:.6f} PDF units)",
                file=sys.stderr,
            )

        # Classify page mode
        page_mode = classify_page_mode(page, primitives)

        # Collect timings
        timings = {
            "render_ms": render_time_ms,
            "extract_ms": extract_time_ms,
            "total_ms": (time.time() - total_start) * 1000,
        }

        step1_info = None
        try:
            step1_info = compute_step1_signature(pdf_path=pdf_path, dpi=dpi, page_index=page_index)
        except Exception:
            # Signature is used for smart skipping in the UI; don't fail Step 1 if it can't be computed.
            step1_info = None

        # Write artifacts
        write_artifacts(
            output_dir=output_dir,
            page_id=page_id,
            source_pdf=str(pdf_path),
            image=image,
            primitives=primitives_pix,  # Use pixel-space primitives
            transform=transform_dict,
            page_mode=page_mode,
            timings=timings,
            step1_info=step1_info,
        )

        # Generate debug overlay if requested
        if enable_debug_overlay:
            _create_debug_overlay(output_dir, image, primitives_pix)

        print(f"Successfully processed {pdf_path}")
        print(f"  Mode: {page_mode['mode']}")
        print(f"  Output: {output_dir}")
        print(f"  Total time: {timings['total_ms']:.1f}ms")

    finally:
        doc.close()


def _create_debug_overlay(output_dir: Path, image, primitives: dict) -> None:
    """Create a debug overlay showing primitives overlaid on the raster image."""
    from PIL import ImageDraw

    # Create a copy of the image for drawing
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    import random

    # Sample up to 200 primitives for visualization
    all_prims = []
    all_prims.extend(primitives.get("lines", [])[:100])
    all_prims.extend(primitives.get("beziers", [])[:50])
    all_prims.extend(primitives.get("rects", [])[:50])

    random.shuffle(all_prims)
    sample_prims = all_prims[:200]

    # Draw sampled primitives
    for prim in sample_prims:
        if "p0" in prim and "p1" in prim:  # Line
            p0 = (int(prim["p0"]["x"]), int(prim["p0"]["y"]))
            p1 = (int(prim["p1"]["x"]), int(prim["p1"]["y"]))
            draw.line([p0, p1], fill=(255, 0, 0), width=2)
        elif "p0" in prim and "p3" in prim:  # Bezier (approximate as line)
            p0 = (int(prim["p0"]["x"]), int(prim["p0"]["y"]))
            p3 = (int(prim["p3"]["x"]), int(prim["p3"]["y"]))
            draw.line([p0, p3], fill=(0, 255, 0), width=2)
        elif "rect" in prim:  # Rectangle
            r = prim["rect"]
            x0, y0 = int(r["x0"]), int(r["y0"])
            x1, y1 = int(r["x1"]), int(r["y1"])
            
            # Ensure x0 <= x1 and y0 <= y1 for PIL
            draw.rectangle(
                [(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))],
                outline=(0, 0, 255),
                width=2,
            )

    # Save overlay
    overlay_path = output_dir / "debug_overlay.png"
    overlay.save(overlay_path, "PNG")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Step 1: Convert PDF floor plan to analysis-ready representation"
    )
    parser.add_argument("pdf_path", type=Path, help="Path to input PDF file")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for artifacts (e.g., artifacts/floor_plan)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="DPI for rendering (default: 400, use 600 for very small symbols)",
    )
    parser.add_argument(
        "--page-index",
        type=int,
        default=0,
        help="Page index to process (default: 0)",
    )
    parser.add_argument(
        "--no-debug-overlay",
        action="store_true",
        help="Skip generating debug overlay image",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF file not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    try:
        process_pdf(
            pdf_path=args.pdf_path,
            output_dir=args.out,
            dpi=args.dpi,
            page_index=args.page_index,
            enable_debug_overlay=not args.no_debug_overlay,
        )
    except Exception as e:
        print(f"Error processing PDF: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

