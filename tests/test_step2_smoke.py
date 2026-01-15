"""Smoke test for Step 2 door detection."""

import json
import subprocess
import sys
from pathlib import Path


def test_step2():
    """Run Step 1 and then Step 2 on a test PDF and validate outputs."""
    # Find a test PDF
    pdf_dir = Path("tests/door_drawings")
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print("Error: No test PDFs found in tests/door_drawings")
        return 1
    
    test_pdf = pdfs[0]
    out_dir = Path("artifacts/smoke_test")
    
    # 1. Run Step 1
    print(f"Running Step 1 on {test_pdf}...")
    subprocess.run([
        sys.executable, "-m", "door_detector.step1_pipeline",
        str(test_pdf),
        "--out", str(out_dir),
        "--dpi", "400"
    ], check=True)
    
    # 2. Run Step 2
    print(f"Running Step 2 on {out_dir}...")
    subprocess.run([
        sys.executable, "-m", "door_detector.step2_pipeline",
        "--artifacts", str(out_dir),
        "--config", "configs/door_rules.json"
    ], check=True)
    
    # 3. Validate Step 2 artifacts
    doors_json = out_dir / "doors.json"
    overlay_png = out_dir / "doors_overlay.png"
    
    if not doors_json.exists():
        print(f"Error: Missing {doors_json}")
        return 1
    if not overlay_png.exists():
        print(f"Error: Missing {overlay_png}")
        return 1
        
    with open(doors_json) as f:
        data = json.load(f)
        if "doors" not in data:
            print("Error: 'doors' key missing in doors.json")
            return 1
        print(f"✓ Found {len(data['doors'])} door detections")
        
    print("✓ Step 2 smoke test passed!")
    return 0


if __name__ == "__main__":
    sys.exit(test_step2())

