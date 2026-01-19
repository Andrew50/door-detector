# Setup Guide

## System Requirements

- Python 3.10 or higher
- pip (Python package installer)

## Step-by-Step Setup

### 1. Check Python Installation

First, verify Python is installed:

```bash
python3 --version
```

You should see something like `Python 3.10.x` or higher. If not, install Python 3.10+ from [python.org](https://www.python.org/downloads/).

### 2. Create Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
cd /home/aj/dev/door_detector
python3 -m venv venv
```

This creates a `venv` directory with an isolated Python environment.

### 3. Activate Virtual Environment

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt, indicating the virtual environment is active.

### 4. Upgrade pip (Recommended)

```bash
python3 -m pip install --upgrade pip
```

Or if `pip` is available:
```bash
pip install --upgrade pip
```

### 5. Install the Package

```bash
pip install -e .

```

If `pip` command is not found, use:
```bash
python3 -m pip install -e .
```

This installs the package in "editable" mode, so code changes are immediately available.

### 6. Verify Installation

Check that the CLI command is available:

```bash
door-detector-step1 --help
```

You should see usage information. If you get a "command not found" error:
- Make sure the virtual environment is activated
- Try: `python3 -m door_detector.step1_pipeline --help`

## Troubleshooting

### "pip: command not found"

**Solution 1:** Use `python3 -m pip` instead:
```bash
python3 -m pip install -e .
```

**Solution 2:** Install pip:
```bash
python3 -m ensurepip --upgrade
```

**Solution 3:** On some systems, pip3 is available:
```bash
pip3 install -e .
```

### "python3: command not found"

- On some systems, use `python` instead of `python3`
- Check what's available: `which python` or `which python3`
- You may need to install Python 3.10+ from your system package manager

### "venv: command not found" or "No module named venv"

Install the venv module:
```bash
# On Ubuntu/Debian
sudo apt-get install python3-venv

# On macOS (if using Homebrew)
brew install python3

# Or use virtualenv as alternative
pip install virtualenv
virtualenv venv
```

### Virtual Environment Not Activating

- Make sure you're in the project directory
- Check that `venv` directory exists: `ls -la venv/`
- Try the full path: `source /home/aj/dev/door_detector/venv/bin/activate`

### Package Installation Fails

**Check Python version:**
```bash
python3 --version  # Must be 3.10 or higher
```

### Browser console warnings (Permissions-Policy / iframe sandbox)

You may see browser console warnings like:

- `Unrecognized feature: 'ambient-light-sensor'` (and similar)
- `An iframe which has both allow-scripts and allow-same-origin for its sandbox attribute can escape its sandboxing.`

**What they mean**

- The **"Unrecognized feature"** messages can come from either:
  - **A reverse proxy / hosting layer** injecting an outdated `Permissions-Policy` (or legacy `Feature-Policy`) HTTP response header, or
  - **Streamlit’s own component iframe implementation**, which sets an `iframe allow="..."` list that includes some deprecated/unknown tokens in modern Chromium.
- The **iframe sandbox** message is a Chromium warning about Streamlit **component iframes**. Door Detector’s viewer uses `streamlit.components.v1.html` for pan/zoom and overlay interactivity, which relies on the Streamlit component iframe model.

**How to actually eliminate the “Unrecognized feature” warnings**

If you’re running Streamlit behind a reverse proxy (Nginx/Traefik/Caddy/Cloudflare), remove or replace the header there.

Example Nginx snippet (in your `location /` block):

```nginx
# Strip legacy / noisy policies coming from upstream or defaults
proxy_hide_header Permissions-Policy;
proxy_hide_header Feature-Policy;

# Optional: add a minimal, modern policy (only if you want one)
# (Keep this list small to avoid deprecated directives causing warnings.)
add_header Permissions-Policy "camera=(), microphone=(), geolocation=()" always;
```

If you run Streamlit directly with `streamlit run ...` and **no** proxy, there isn’t a supported way (in Streamlit today) to change the component iframe `allow` list or suppress Chromium’s warnings from inside the app. In that case, these warnings are generally safe to ignore.

## PDF.js viewer build (optional)

The review UI uses a bundled **PDF.js** Streamlit component for a crisp, zoomable viewer.

- **If you are just running the app**: the UI will use the PDF.js viewer **if** the built assets exist at `door_detector/ui/pdfjs_component/frontend/dist/`.
- **If the built assets are missing**: the app falls back to the legacy raster viewer (using `page.png` from Step 1). You can still review detections, but the PDF.js-specific overlay path won’t be used.
- **If you edit the viewer frontend** (TypeScript/React): rebuild it with Node.

Build steps:

```bash
cd door_detector/ui/pdfjs_component/frontend
npm install
npm run build
```

This regenerates `door_detector/ui/pdfjs_component/frontend/dist/`.

**Streamlit / Drawable Canvas Compatibility (Pinned):**
This project pins Streamlit to a compatible version because `streamlit-drawable-canvas==0.9.3` relies on Streamlit internals that changed in newer Streamlit releases.

If you previously installed a newer Streamlit and see errors like `AttributeError: module 'streamlit.elements.image' has no attribute 'image_to_url'`, the simplest fix is to recreate your virtualenv and reinstall:

```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
```

**Install build tools (if needed):**
```bash
# On Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# On macOS
xcode-select --install
```

**Try installing dependencies separately:**
```bash
pip install pymupdf pillow numpy setuptools wheel
pip install -e .
```

## Quick Reference

```bash
# Create venv
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install package
pip install -e .

# Deactivate venv (when done)
deactivate
```

## Next Steps

After successful installation, see [TESTING.md](TESTING.md) for how to test the implementation.
