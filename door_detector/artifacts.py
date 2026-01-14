"""Write and read artifact bundles."""

import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image


def write_artifacts(
    output_dir: Path,
    page_id: str,
    source_pdf: str,
    image: Image.Image,
    primitives: Dict[str, Any],
    transform: Dict[str, Any],
    page_mode: Dict[str, Any],
    timings: Dict[str, float],
) -> None:
    """
    Write all artifacts to disk in a deterministic format.

    Args:
        output_dir: Output directory (e.g., artifacts/{id})
        page_id: Identifier for this page
        source_pdf: Path to source PDF
        image: Rendered PIL Image
        primitives: Extracted primitives dictionary
        transform: Transform dictionary
        page_mode: Page mode classification
        timings: Timing statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write PNG
    image_path = output_dir / "page.png"
    image.save(image_path, "PNG")

    # Write primitives.json
    primitives_path = output_dir / "primitives.json"
    with open(primitives_path, "w") as f:
        json.dump(primitives, f, indent=2, sort_keys=True, ensure_ascii=False)

    # Write transform.json
    transform_path = output_dir / "transform.json"
    # Convert to JSON-serializable format (ensure floats are formatted consistently)
    transform_serializable = _make_json_serializable(transform)
    with open(transform_path, "w") as f:
        json.dump(transform_serializable, f, indent=2, sort_keys=True, ensure_ascii=False)

    # Write meta.json
    meta = {
        "id": page_id,
        "source_pdf": source_pdf,
        "dpi": transform["dpi"],
        "pix_width": transform["pix_width"],
        "pix_height": transform["pix_height"],
        "page_rect": transform["page_rect"],
        "mode": page_mode["mode"],
        "stats": {
            **page_mode["stats"],
            **primitives["stats"],
            **timings,
        },
    }
    meta_path = output_dir / "meta.json"
    meta_serializable = _make_json_serializable(meta)
    with open(meta_path, "w") as f:
        json.dump(meta_serializable, f, indent=2, sort_keys=True, ensure_ascii=False)


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, float):
        # Format floats consistently (round to reasonable precision)
        return round(obj, 6)
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        # Convert unknown types to string
        return str(obj)



