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

### Artifact Structure

See the plan document for detailed schema specifications.

## Testing Step 1

### Quick Start

1. **Set up virtual environment (if not already done):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```
   
   If `pip` is not found, use:
   ```bash
   python3 -m pip install -e .
   ```

2. **Get test data:**
   - Download floor plans from the [provided Google Drive folder](https://drive.google.com/drive/folders/1QSsrLCr13xX6h-LYBolEslI369vtahwc?usp=sharing)
   - Place PDF files in an `inputs/` directory (or any directory you prefer)

3. **Run on a single PDF:**
   ```bash
   door-detector-step1 inputs/your_floor_plan.pdf --out artifacts/test_output --dpi 400
   ```

4. **Verify output:**
   Check that the following files were created in `artifacts/test_output/`:
   - `page.png` - Should show the rendered floor plan
   - `primitives.json` - Should contain extracted vector data
   - `transform.json` - Should contain transformation matrices
   - `meta.json` - Should contain metadata and classification
   - `debug_overlay.png` - Should show primitives overlaid on the raster

### Testing Checklist

- [ ] Installation completes without errors
- [ ] CLI command runs successfully
- [ ] All 5 artifact files are generated
- [ ] `page.png` is a valid image file
- [ ] `primitives.json` contains non-empty arrays for lines/beziers/rects
- [ ] `transform.json` contains valid affine matrices
- [ ] `meta.json` shows a valid mode (scan/vector/hybrid)
- [ ] `debug_overlay.png` shows primitives aligned with the raster
- [ ] Transform validation passes (check console output for warnings)

### Test Script

Run the automated test script:
```bash
python tests/test_step1.py inputs/your_floor_plan.pdf
```

This will verify all artifacts are generated correctly and check data integrity.

