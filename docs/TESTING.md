# Testing Guide

## Prerequisites

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
python3 -m pip install -e ".[dev]"
```

Floor-plan PDFs are not included in the repository. For manual CLI/UI checks, place documents you are allowed to process under `inputs/` (gitignored).

## Automated checks

End-to-end smoke test on a tiny generated PDF (Step 1 + Step 2):

```bash
python3 tests/test_step2_smoke.py
```

Full unit suite:

```bash
pytest -q
```

## Manual CLI checks (optional)

```bash
door-detector-step1 inputs/floor_plan_01.pdf --out artifacts/test_01 --dpi 400
ls -la artifacts/test_01/
python tests/test_step1.py artifacts/test_01
door-detector-step2 --artifacts artifacts/test_01 --config configs/door_rules.json
```

Expected Step 1 artifacts: `page.png`, `primitives.json`, `transform.json`, `meta.json` (and optional `debug_overlay.png`).

Expected Step 2 artifacts: `doors.json` (`doors` + `candidates`), `doors_overlay.png`.

Inspect `meta.json` for `mode` (`scan` / `vector` / `hybrid`). Vector rules are intended for `vector` and `hybrid` pages; pure scans return empty results by policy.

Open `debug_overlay.png` / `doors_overlay.png` to sanity-check alignment of primitives and detections against the raster.

## UI

```bash
streamlit run door_detector/review_app.py
```
