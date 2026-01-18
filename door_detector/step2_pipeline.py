"""Step 2 pipeline: Artifacts → Door detections."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from door_detector.analysis_signature import compute_analysis_signature
from door_detector.door_detection import detect_doors
from door_detector.door_overlay import create_door_overlay


def run_step2(
    artifacts_dir: Path,
    config_path: Path,
    output_dir: Path | None = None
) -> None:
    """Run door detection on a Step 1 artifacts directory."""
    
    if output_dir is None:
        output_dir = artifacts_dir

    # 1. Load config and compute signature
    config = json.loads(config_path.read_bytes())
    analysis_signature = compute_analysis_signature(config_path)

    # 2. Load artifacts
    primitives_path = artifacts_dir / "primitives.json"
    meta_path = artifacts_dir / "meta.json"
    image_path = artifacts_dir / "page.png"

    if not all(p.exists() for p in [primitives_path, meta_path, image_path]):
        raise FileNotFoundError(f"Missing required artifacts in {artifacts_dir}")

    with open(primitives_path) as f:
        primitives = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    
    image = Image.open(image_path)

    # 3. Detect doors
    import time
    start_time = time.time()
    det = detect_doors(primitives, meta, config)
    doors = list(det.get("doors", []) if isinstance(det, dict) else (det or []))
    candidates = list(det.get("candidates", []) if isinstance(det, dict) else [])
    detect_ms = (time.time() - start_time) * 1000

    # 4. Save doors.json
    doors_data = {
        "schema_version": 1,
        "page_id": meta["id"],
        "source_artifacts_dir": str(artifacts_dir),
        "config_path": str(config_path),
        "analysis_signature": analysis_signature,
        "mode": meta["mode"],
        "detect_ms": detect_ms,
        "doors": doors,
        "candidates": candidates,
    }
    
    with open(output_dir / "doors.json", "w") as f:
        json.dump(doors_data, f, indent=2)

    # 5. Create overlay
    create_door_overlay(image, doors, output_dir / "doors_overlay.png")

    print(f"Successfully processed {artifacts_dir}")
    print(f"  Detections: {len(doors)}")
    print(f"  Time: {detect_ms:.1f}ms")
    print(f"  Output: {output_dir / 'doors.json'}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Step 2: Detect doors from normalized artifacts"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Path to Step 1 artifacts directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/door_rules.json"),
        help="Path to door detection rules config",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output directory (defaults to artifacts dir)",
    )

    args = parser.parse_args()

    if not args.artifacts.exists():
        print(f"Error: Artifacts directory not found: {args.artifacts}", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        run_step2(
            artifacts_dir=args.artifacts,
            config_path=args.config,
            output_dir=args.out
        )
    except Exception as e:
        print(f"Error in Step 2: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


