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

Door Detector supports a multi-step pipeline with feedback-driven improvement, all accessible via a unified Streamlit UI.

### 1. Launch the Unified UI
```bash
./venv/bin/streamlit run door_detector/review_app.py
```

The UI is organized into three tabs:

- **🚀 Run Pipeline**: Upload PDFs, select page index, and run the full Step 1 (normalization) and Step 2 (detection) process.
- **🔍 Review Detections**: Navigate through detections, cycle through doors, and provide feedback:
    - **Accept/Reject**: Mark model predictions as true or false positives.
    - **Add Missed Doors**: Use the drawing tool to mark false negatives (missed doors).
    - **Save Labels**: Feedback is saved to `labels.json` in the artifact directory.
- **🧠 Train Reweighter**: Train a logistic regression model on your saved labels to improve future detection confidence.

### 2. Command Line Usage (Optional)

If you prefer using the CLI:

#### Step 1: PDF → Normalized Artifacts
```bash
door-detector-step1 inputs/floor_plan.pdf --out artifacts/floor_plan
```

#### Step 2: Detection
```bash
door-detector-step2 --artifacts artifacts/floor_plan --config configs/door_rules.json
```

#### Step 3: Reweighting (Learning from Feedback)
```bash
door-detector-reweight --artifacts artifacts --out models/reweighter_v1.json
```

To use the learned weights, update your `configs/door_rules.json` to include:
```json
{
  "reweighter_path": "models/reweighter_v1.json",
  ...
}
```

## Feedback Data Model (`labels.json`)

Reviewer feedback is persisted alongside the artifacts:

```json
{
  "schema_version": 1,
  "page_id": "floor_plan_p0",
  "reviewed_at": "2026-01-14T12:40:00Z",
  "accepted_ids": ["d_000123"],
  "rejected_ids": ["d_000124"],
  "added_boxes": [
    {
      "bbox_xyxy": [512.0, 220.0, 605.0, 310.0],
      "note": "Added via UI"
    }
  ],
  "notes": "..."
}
```

## Testing
Run the full smoke test:
```bash
./venv/bin/python tests/test_step2_smoke.py
```

