# Testing Guide for Step 1

## Prerequisites

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
   
   **Note:** Make sure you have Python 3.10 or higher:
   ```bash
   python3 --version  # Should show 3.10 or higher
   ```

2. **Get test data:**
   - Download floor plans from: https://drive.google.com/drive/folders/1QSsrLCr13xX6h-LYBolEslI369vtahwc?usp=sharing
   - Create an `inputs/` directory and place PDF files there

## Basic Testing

### 1. Process a single PDF

```bash
door-detector-step1 inputs/floor_plan_01.pdf --out artifacts/test_01 --dpi 400
```

Expected output:
```
Successfully processed inputs/floor_plan_01.pdf
  Mode: vector
  Output: artifacts/test_01
  Total time: 123.4ms
```

### 2. Verify artifacts

Check that all files were created:
```bash
ls -la artifacts/test_01/
```

You should see:
- `page.png` - Rasterized floor plan
- `primitives.json` - Vector primitives
- `transform.json` - Coordinate transforms
- `meta.json` - Metadata
- `debug_overlay.png` - Debug visualization

### 3. Run automated tests

```bash
python tests/test_step1.py artifacts/test_01
```

This will validate:
- All required files exist
- JSON files are valid and contain required keys
- Images are valid PNG files
- Data schemas match expectations

## Visual Inspection

### Check the debug overlay

Open `artifacts/test_01/debug_overlay.png` in an image viewer. You should see:
- The original floor plan raster
- Red lines showing extracted line segments
- Green lines showing Bezier curves (approximated)
- Blue rectangles showing extracted rectangles

**What to look for:**
- Primitives should align with the raster image
- If primitives are misaligned, the transform may be incorrect
- If no primitives are visible, the PDF might be a scan (check `meta.json` mode)

### Check metadata

```bash
cat artifacts/test_01/meta.json | python -m json.tool
```

Look for:
- `mode`: Should be "scan", "vector", or "hybrid"
- `stats.num_segments`: Number of extracted segments
- `stats.image_coverage`: Percentage of page covered by images
- `stats.total_ms`: Processing time

## Testing Different Scenarios

### Test with different DPI

```bash
# Higher DPI for detailed analysis
door-detector-step1 inputs/floor_plan_01.pdf --out artifacts/test_01_600dpi --dpi 600

# Compare file sizes and processing times
ls -lh artifacts/test_01/page.png artifacts/test_01_600dpi/page.png
```

### Test scanned PDFs

If you have scanned floor plans (images embedded in PDF):
- The mode should be classified as "scan"
- `primitives.json` may have fewer segments
- `stats.image_coverage` should be high (>0.6)

### Test vector PDFs

For true vector PDFs:
- Mode should be "vector"
- `primitives.json` should have many segments
- `stats.image_coverage` should be low (<0.2)

## Troubleshooting

### "PDF file not found"
- Check the path is correct
- Use absolute paths if relative paths don't work

### "Transform validation failed"
- Check console output for the error value
- Small errors (< 0.01) are usually acceptable
- Large errors indicate a problem with rotation handling

### Empty primitives.json
- PDF might be a pure scan (no vector data)
- Check `meta.json` mode - if "scan", this is expected
- Try a different PDF with vector content

### Primitives misaligned in debug_overlay.png
- Check `transform.json` for correct rotation_deg
- Verify `pix_width` and `pix_height` match `page.png` dimensions
- The transform may need adjustment for complex rotations

## Performance Testing

Process multiple PDFs and check timing:

```bash
for pdf in inputs/*.pdf; do
    echo "Processing $pdf..."
    door-detector-step1 "$pdf" --out "artifacts/$(basename $pdf .pdf)" --dpi 400
done
```

Check `meta.json` in each output for `total_ms` to compare performance.

## Expected Results

For a typical vector floor plan:
- Processing time: 50-500ms (depending on complexity)
- Primitives: 100-5000+ segments
- Mode: "vector" or "hybrid"
- Transform validation: Should pass (< 1e-3 error)

For scanned floor plans:
- Processing time: 50-200ms
- Primitives: 0-100 segments
- Mode: "scan"
- Transform validation: May have fewer points to validate

