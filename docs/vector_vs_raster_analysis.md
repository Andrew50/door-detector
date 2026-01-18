# Vector-first vs Raster Detection: What to Focus On (Assessment Memo)

## Executive summary

- **Recommendation**: **Focus your *door detection* on vector-first methods** for this assessment, because the provided test set in `tests/door_drawings/` appears to be **true vector/CAD exports** (very high vector primitive density, very low embedded-image coverage).
- **Do not remove raster entirely**: keep rasterization (`page.png`) because it is the simplest way to **visualize results** (overlay highlights) and provides a clean **fallback path** if you encounter scanned PDFs later.
- **Pragmatic scope**: ship a strong vector detector + a review UI; implement a minimal scan-handling story (detect “scan mode” and either run a lightweight raster fallback or clearly report “scan not supported yet”).

## Evidence from the provided test set

### Method used

The repo already includes a scan/vector/hybrid classifier: `door_detector.pdf.classify.classify_page_mode`.

To quickly classify **without rendering** (fast, avoids transform/rotation validation noise), you can run:

```bash
cd /home/aj/dev/door_detector
source venv/bin/activate
python - <<'PY'
from pathlib import Path
import fitz
from door_detector.pdf.vectors import extract_primitives
from door_detector.pdf.classify import classify_page_mode

for pdf in sorted(Path("tests/door_drawings").glob("*.pdf")):
    doc = fitz.open(pdf)
    page = doc[0]
    primitives = extract_primitives(page)
    mode = classify_page_mode(page, primitives)
    s = mode["stats"]
    print(f"{mode['mode']}\timg_cov={s['image_coverage']:.3f}\tsegs={s['num_segments']}\timgs={s['num_images']}\t{pdf.name}")
    doc.close()
PY
```

### What we observed

- **All 20 PDFs** in `tests/door_drawings/` classified as **`vector`**.
- Typical stats:
  - **`image_coverage`** ≈ **0.001–0.010** for most pages (occasionally up to ~0.10)
  - **`num_segments`** commonly **tens of thousands** to **100k+**

### Interpretation

This is consistent with **CAD/vector PDFs**:
- Many vector segments (walls, fixtures, annotations as linework).
- Only small embedded images (logos, stamps, small raster inserts), hence low image coverage.

It is *not* consistent with photographed/scanned plans, which typically show:
- One large image covering most of the page (high `image_coverage`, often > 0.60).
- Few vector segments (maybe a border box, or nothing).

## Should you remove raster components?

### No — keep raster for visualization and robustness

Even if you do **vector-first detection**, you will still benefit from rasterization because:
- **Overlay output**: the clearest “deliverable” for reviewers is a **highlighted image/PDF**.
- **UI simplicity**: most lightweight UIs (Streamlit/Gradio) are easiest if you display a PNG and draw boxes/marks on it.
- **Future-proofing**: you can gracefully handle scanned PDFs later without redesigning the pipeline.

### But yes — de-scope raster *detection* work for now

Given your test set appears vector, the highest ROI is:
- Implement a **vector-first detector** (door = arc-like curve + nearby leaf line, etc.).
- Provide a clean **confidence score** and output format.
- Build a **review/feedback UI** that allows toggling detections and adding missed doors.

Raster *detection* (training an object detector on pixels) can be treated as:
- Optional “scan mode” enhancement
- Bonus work if you have spare time

## How to structure the solution around this decision

### Core pipeline (recommended)

- **Step 1 (existing)**: produce artifacts including:
  - `page.png` (for visualization)
  - `primitives.json` (for vector detection)
  - `meta.json` (`mode` decision: vector/scan/hybrid)
- **Step 2 (your door detector)**:
  - If `meta.mode` is `vector`/`hybrid`: run **vector-first detector**
  - If `meta.mode` is `scan`: either
    - run a minimal raster fallback, or
    - return an empty list + a clear message (honest and defensible in a 3–5 min demo)
- **Step 3 (review UI + feedback)**:
  - Show `page.png` + detections
  - Allow “false positive” and “missed door” marking
  - Save corrections as a `labels.json` next to the artifacts

## Concrete “rule of thumb” thresholds (as implemented)

From `door_detector.pdf.classify.classify_page_mode`:
- **Scan** if `image_coverage >= 0.60` and `num_segments <= low_segment_threshold` (default 50)
- **Vector** if `num_segments >= high_segment_threshold` (default 200) and `image_coverage <= 0.20`
- **Hybrid** otherwise

## Suggested narrative for your README / demo video

- “We first classify each PDF page as vector/scan/hybrid.”
- “On the provided dataset (20 plans), all pages are vector-like, so the detector uses vector primitives (fast and explainable).”
- “We still render a raster image for easy overlay visualization and for a future scan fallback path.”
- “The UI allows marking false positives/missed doors; those labels can train a lightweight re-ranker (vector features) or fine-tune a pixel detector later.”


