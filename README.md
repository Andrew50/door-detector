# Door Detector: Door Detection in Floor Plans

Detect and review door candidates (swing/double/pocket/bifold) in architectural floor plan PDFs.

This repo is set up to be easy to run locally (Streamlit UI) and easy to inspect (artifacts + overlays).

## Quickstart (run the review UI)

Requirements: **Python 3.10+**.

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
streamlit run door_detector/review_app.py
```

In the UI:

- Upload a PDF (saved under `artifacts/library/<file_id>/source.pdf`)
- Click **Analyze** (runs Step 1 + Step 2)
- Review detections (confirm / reject / delete) and add missed doors (**Edit Doors → Shift+drag**)
- Labels are saved to `artifacts/library/<file_id>/labels.json` (schema v4)

## Data

- Floor plan PDFs: [Google Drive folder](https://drive.google.com/drive/folders/1QSsrLCr13xX6h-LYBolEslI369vtahwc?usp=sharing)

## Setup

Requirements: **Python 3.10+**.

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
```

Optional (dev / tests):

```bash
python3 -m pip install -e ".[dev]"
```

More details (incl. troubleshooting / optional PDF.js rebuild notes): see `docs/SETUP.md`.

## Run (optional): CLI pipeline

```bash
# Step 1: PDF → normalized artifacts (raster + extracted vector primitives)
door-detector-step1 inputs/floor_plan.pdf --out artifacts/floor_plan --dpi 400 --page-index 0

# Step 2: detect doors
door-detector-step2 --artifacts artifacts/floor_plan --config configs/door_rules.json

# (Optional) learn from review labels (writes per-type models under --out-dir)
door-detector-reweight --artifacts artifacts --out-dir models
```

If you train models and save them under non default names, update `configs/door_rules.json` → `reweighters` to point at them (see `docs/retraining.md`).

## Outputs (what to look at)

For a single processed page (CLI or UI), you’ll typically see:

- `page.png`: rasterized page
- `primitives.json`: extracted vector primitives (pixel space)
- `transform.json`: PDF↔pixel transforms
- `meta.json`: timings + mode (scan/vector/hybrid)
- `debug_overlay.png`: optional Step 1 debug overlay (can be disabled)
- `doors.json`: Step 2 output (final `doors` + broader `candidates`)
- `doors_overlay.png`: Step 2 visualization overlay
- `labels.json`: reviewer feedback (created/updated by the UI)

For detection and learning details, see `docs/door_selection_process.md`.

## Testing

```bash
python3 tests/test_step2_smoke.py
```

If you prefer `pytest`, install the dev extra and run:

```bash
pytest -q
```

More tests and local workflows: see `docs/TESTING.md`.

