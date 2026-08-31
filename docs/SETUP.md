# Setup Guide

## Requirements

- Python 3.10+
- pip

Node/npm are **not** required to run the app. Rebuild the PDF.js viewer only if you edit its TypeScript/React sources.

## Install

```bash
cd /path/to/door-detector
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
python3 -m pip install -e .
```

Dev/tests:

```bash
python3 -m pip install -e ".[dev]"
```

Verify CLIs:

```bash
door-detector-step1 --help
door-detector-step2 --help
door-detector-reweight --help
```

Run the review UI:

```bash
streamlit run door_detector/review_app.py
```

## Streamlit pin

Streamlit is pinned (`streamlit~=1.48.0`) because `streamlit-drawable-canvas==0.9.3` depends on Streamlit internals that changed in newer releases. If you see `AttributeError: ... image_to_url`, recreate the venv and reinstall with `python3 -m pip install -e .`.

## PDF.js viewer rebuild (optional)

Bundled assets live at `door_detector/ui/pdfjs_component/frontend/dist/`. If those are missing, the UI falls back to the raster (`page.png`) viewer.

After editing the viewer frontend:

```bash
cd door_detector/ui/pdfjs_component/frontend
npm install
npm run build
```

## Troubleshooting

**`pip` / `python3` not found** — use `python3 -m pip`, or install Python 3.10+ from [python.org](https://www.python.org/downloads/).

**`venv` missing on Debian/Ubuntu** — `sudo apt-get install python3-venv`.

**Browser console noise** (`Permissions-Policy`, iframe sandbox) — usually from Streamlit’s component iframe model or a reverse proxy header. Safe to ignore for local `streamlit run`. Behind Nginx you can strip injected policy headers if you care about the warnings.

Next: [TESTING.md](TESTING.md).
