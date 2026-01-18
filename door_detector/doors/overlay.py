"""Visualize door detections on top of the raster image."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

# Increase PIL pixel limit
Image.MAX_IMAGE_PIXELS = None


def create_door_overlay(image: Image.Image, doors: List[Dict[str, Any]], output_path: Path) -> None:
    """Draw bounding boxes and labels for detected doors on the image."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    colors = {
        "swing": (0, 255, 0),  # Green
        "double": (0, 180, 255),  # Cyan-blue
        "pocket": (255, 215, 0),  # Gold
        "bifold": (186, 85, 211),  # Purple
    }
    default_color = (255, 0, 0)  # Red

    for door in doors:
        bbox = door["bbox_xyxy"]
        dtype = door["type"]
        conf = door["confidence"]

        x0, y0, x1, y1 = bbox
        norm_bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

        color = colors.get(dtype, default_color)

        draw.rectangle(norm_bbox, outline=color, width=3)

        label = f"{dtype} {conf:.2f}"
        text_bbox = draw.textbbox((norm_bbox[0], norm_bbox[1] - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((norm_bbox[0], norm_bbox[1] - 20), label, fill=(0, 0, 0), font=font)

    overlay.save(output_path, "PNG")

