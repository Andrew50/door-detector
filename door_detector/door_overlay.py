"""Visualize door detections on top of the raster image."""

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

# Increase PIL pixel limit
Image.MAX_IMAGE_PIXELS = None


def create_door_overlay(
    image: Image.Image,
    doors: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Draw bounding boxes and labels for detected doors on the image."""
    # Create a copy for drawing
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    
    # Try to load a font, fallback to default
    try:
        # On many Linux systems, DejaVuSans is available
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Define colors for different types
    colors = {
        "swing": (0, 255, 0),    # Green
        "double": (0, 255, 255), # Cyan
        "pocket": (255, 165, 0), # Orange
        "bifold": (255, 0, 255), # Magenta
    }
    default_color = (255, 0, 0) # Red

    for door in doors:
        bbox = door["bbox_xyxy"]
        dtype = door["type"]
        conf = door["confidence"]
        
        # Ensure bbox is normalized for PIL
        x0, y0, x1, y1 = bbox
        norm_bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        
        color = colors.get(dtype, default_color)
        
        # Draw rectangle
        draw.rectangle(norm_bbox, outline=color, width=3)
        
        # Draw label
        label = f"{dtype} {conf:.2f}"
        # Draw background for text readability
        text_bbox = draw.textbbox((norm_bbox[0], norm_bbox[1] - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((norm_bbox[0], norm_bbox[1] - 20), label, fill=(0, 0, 0), font=font)

    overlay.save(output_path, "PNG")

