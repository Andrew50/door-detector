# Door Detector: Door Detection in Floor Plans

Detect and highlight doors (swing doors + extensions) in **single-page** architectural floor plan PDFs.

This repo is set up to be easy to run locally (Streamlit UI) and easy to inspect (artifacts + overlays).

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

## Run (recommended): Streamlit review app

```bash
./venv/bin/streamlit run door_detector/review_app.py
```

In the UI:

- Upload a PDF into the library
- Run analysis (Step 1 + Step 2)
- Review detections (confirm / delete) and add missed doors (Shift+drag)
- Optionally retrain the reweighter from accumulated labels

## Run (optional): CLI pipeline

```bash
# Step 1: PDF → normalized artifacts (raster + extracted vector primitives)
door-detector-step1 inputs/floor_plan.pdf --out artifacts/floor_plan --dpi 400

# Step 2: detect doors
door-detector-step2 --artifacts artifacts/floor_plan --config configs/door_rules.json

# (Optional) learn from review labels
door-detector-reweight --artifacts artifacts --out models/reweighter_v1.json
```

If you train a model, point `configs/door_rules.json` at it (see `docs/retraining.md`).

## Outputs (what to look at)

Each artifacts directory contains (at minimum):

- `page.png`: rasterized page
- `primitives.json`: extracted vector primitives
- `transform.json`: PDF↔pixel transforms
- `meta.json`: timings + mode (scan/vector/hybrid)
- `doors.json`: detected doors + candidate pool
- `labels.json`: reviewer feedback (created/updated by the UI)

For detection and learning details, see `docs/door_selection_process.md`.

## Testing

```bash
./venv/bin/python tests/test_step2_smoke.py
```

If you prefer `pytest`, install the dev extra and run:

```bash
pytest -q
```

More tests and local workflows: see `docs/TESTING.md`.

## Demo video / deployment

- Demo video (Loom): **TODO — add link**
- Deployment: **run locally** via Streamlit (no hosted deployment included in this repo)

