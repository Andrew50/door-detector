# Door Detector: Door Detection in Floor Plans

A system for detecting and highlighting doors in architectural floor plan PDFs.

## Step 1: PDF → Analysis-Ready Representation

This step converts a single-page floor plan PDF into a normalized, analysis-ready format.

### Installation

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

   If `pip` is not found, try:
   ```bash
   python3 -m pip install -e .
   ```

### Usage

```bash
door-detector-step1 inputs/floor_plan.pdf --out artifacts/floor_plan --dpi 400
```

### Output Artifacts

For each processed PDF, the following artifacts are generated in the output directory:

- `page.png` - Rasterized floor plan at specified DPI
- `primitives.json` - Extracted vector primitives (lines, curves, rectangles)
- `transform.json` - PDF↔pixel coordinate transformation matrices
- `meta.json` - Metadata including page mode (scan/vector/hybrid), stats, and timings
- `debug_overlay.png` - Optional visualization showing primitives overlaid on the raster

## Full Pipeline: PDF → Doors

Door Detector now supports a multi-step pipeline with feedback-driven improvement.

### 1. Step 1: PDF → Normalized Artifacts
```bash
door-detector-step1 inputs/floor_plan.pdf --out artifacts/floor_plan
```

### 2. Step 2: Detection
```bash
door-detector-step2 --artifacts artifacts/floor_plan --config configs/door_rules.json
```
This generates `doors.json` and `doors_overlay.png`.

### 3. Review & Feedback (UI)
```bash
./venv/bin/streamlit run door_detector/review_app.py
```
Open the web app, select your artifact directory, and mark detections as **Accepted** or **Rejected**. Click **Save Labels** to generate `labels.json`.

### 4. Reweighting (Learning from Feedback)
Once you have reviewed a few plans:
```bash
door-detector-reweight --artifacts artifacts --out models/reweighter_v1.json
```
To use the learned weights in future detections, update your `configs/door_rules.json` to include:
```json
{
  "reweighter_path": "models/reweighter_v1.json",
  ...
}
```

## Testing
Run the full smoke test:
```bash
./venv/bin/python tests/test_step2_smoke.py
```

