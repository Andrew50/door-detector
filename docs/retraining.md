## Feedback learning (Option 1): Reweight / rerank vector candidates

This document describes the recommended “learning from feedback” approach for this project: **keep a vector-first door detector**, but improve it over time by **reweighting features and recalibrating thresholds** based on reviewer feedback.

This is intentionally lightweight:

- **No deep learning required**
- **Fast iteration** (seconds, CPU-only)
- **Explainable** (you can show which geometric cues drove a decision)
- Works especially well when pages are **vector/hybrid** (`meta.json` → `mode` is `vector` or `hybrid`)

### What “retraining” means here

Despite the filename, this option is not “train a new object detector from scratch.” It is:

- **Candidate generation stays rule-based** (geometry from `primitives.json`)
- A small model (or even a weighted score) learns to map candidate features → **door probability**
- You use that probability to **reduce false positives** and tune recall/precision

## Inputs and outputs

### Inputs (already produced by Step 1)

Each processed PDF page has an artifacts directory containing:

- `page.png` (for visualization and UI overlay)
- `primitives.json` (vector primitives in **pixel** coordinates)
- `meta.json` (includes `mode`: scan/vector/hybrid)
- `transform.json` (optional for mapping results back to PDF coords)

### Outputs (you add)

Per page:

- `doors.json`: model predictions (candidates + confidence + geometry)
- `doors_overlay.png`: visual overlay of predictions
- `labels.json`: human feedback (accept/reject/add)

Global (learned parameters):

- `models/reweighter_<type>_v1.json`: per-door-type feature normalization + weights + bias
  - (the deployment keep/drop thresholds are configured in `configs/door_rules.json`, not stored in the model)

## Core idea: separate “propose” from “score”

### 1) Propose door candidates (geometry rules)

On vector / hybrid pages, propose candidates using primitives such as:

- **Bezier arcs** that resemble door swing arcs (often quarter-circle-ish)
- **Leaf lines** near the arc that represent the door panel
- (Optional) nearby “wall lines” that suggest the opening boundary

The output of this stage should be a list of *candidates*, not final decisions.

### 2) Score candidates (learned reweighter)

For each candidate, compute features (below), then score it using either:

- a **weighted linear score + sigmoid** (logistic regression), or
- a small decision tree (if you later allow a dependency), or
- a tuned heuristic with learned thresholds

The model only decides:

- **should we keep this candidate?**
- and what **confidence** to assign

## Data model

### `doors.json` (predictions)

Approximate format (matches `door-detector-step2`, schema v2):

```json
{
  "schema_version": 2,
  "page_id": "floor_plan_p0",
  "source_artifacts_dir": "artifacts/floor_plan",
  "config_path": "configs/door_rules.json",
  "analysis_signature": "sha1:…",
  "mode": "vector",
  "detect_ms": 123.4,
  "doors": [
    {
      "id": "d_…",
      "type": "swing",
      "bbox_xyxy": [120.5, 340.2, 220.1, 430.9],
      "bbox_pdf_xyxy": [12.3, 45.6, 78.9, 101.1],
      "confidence": 0.92,
      "heuristic_confidence": 0.78,
      "legacy_ids": [],
      "features": {
        "rmse": 0.4,
        "radius": 55.0,
        "angle_span": 90.0
      }
    }
  ],
  "candidates": [
    {
      "id": "d_…",
      "type": "swing",
      "bbox_xyxy": [120.5, 340.2, 220.1, 430.9],
      "bbox_pdf_xyxy": [12.3, 45.6, 78.9, 101.1],
      "confidence": 0.66,
      "heuristic_confidence": 0.66,
      "legacy_ids": ["d_legacy…"],
      "features": {
        "rmse": 0.9,
        "radius": 60.0,
        "angle_span": 82.0
      }
    }
  ]
}
```

Notes:

- Keeping `features` in the output makes the system **auditable** and helps debugging.
- `bbox_pdf_xyxy` is added by Step 2 when `transform.json` is present and is used by the PDF.js viewer overlay.

### `labels.json` (feedback)

Current format (schema v4; written by the Streamlit UI):

```json
{
  "schema_version": 4,
  "reviewed_at": "2026-01-14T12:40:00Z",
  "confirmed_by_type": {
    "swing": ["d_000123"],
    "double": [],
    "pocket": [],
    "bifold": []
  },
  "rejected_by_type": {
    "swing": [],
    "double": [],
    "pocket": [],
    "bifold": []
  },
  "deleted_ids": ["d_000124"],
  "manual_candidates": [],
  "manual_additions": [
    {
      "drawn_bbox_xyxy": [512.0, 220.0, 605.0, 310.0],
      "snapped_candidate_id": "d_000130",
      "iou": 0.63,
      "snapped_bbox_xyxy": [500.5, 215.2, 612.1, 320.9]
    }
  ],
  "unmatched_manual_boxes": [
    {
      "bbox_xyxy": [800.0, 100.0, 880.0, 180.0],
      "note": "No candidate match"
    }
  ]
}
```

Interpretation:

- **confirmed_by_type** → positive examples for that door type
- **rejected_by_type** → “not this type” feedback (kept separate from global deletions)
- **deleted_ids** → global negative examples (“not a door at all”)
- **manual_additions** record Shift+drag selections and may snap to an existing candidate id
- **unmatched_manual_boxes** are UI-only visibility; they are ignored for training unless you turn them into candidates

## Feature design (what to learn weights over)

All features should be computed from `primitives.json` in pixel space.

High-value feature categories:

- **Arc quality**
  - `arc_angle_deg`: swing arcs often span ~70–110°
  - `arc_radius_px`: depends on drawing scale, so normalize (see below)
  - `arc_fit_error`: how close the bezier is to a circular arc
- **Leaf relationship**
  - distance from arc endpoint to nearest leaf endpoint
  - angle between leaf and arc tangent at endpoint
  - leaf length relative to arc radius
- **Style consistency**
  - ratio of stroke width (leaf vs arc vs nearby walls)
  - dashed vs solid mismatch (often indicates non-doors)
- **Context**
  - local line density / clutter around candidate (helps reduce furniture FPs)
  - “opening gap” heuristic (a door leaf often sits in a gap along a wall line)

### Normalization (important)

Because PDFs vary in scale and DPI, normalize size features.

Two practical normalization strategies:

- **Normalize by “typical stroke width”** on the page (median of line stroke widths)
- **Normalize by “typical wall spacing / text height”** if you can estimate it

At minimum, store:

- `scale_px = median_stroke_width_px` and compute `arc_radius_norm = arc_radius_px / scale_px`

## Model: logistic regression (recommended baseline)

Use a tiny linear model to map features → probability:

\[
P(\text{door} \mid x) = \sigma(w^\top \hat{x} + b)
\]

Where:

- \(\hat{x}\) is the normalized feature vector (z-score or robust scaling)
- \(\sigma\) is the sigmoid function

Why logistic regression fits this assessment:

- stable on small datasets
- easy to implement with just `numpy`
- produces calibrated-ish probabilities
- explainable (feature weights)

### What the “training” step produces

Store learned parameters in JSON:

```json
{
  "schema_version": 2,
  "model_type": "logreg",
  "feature_order": [
    "rmse",
    "radius",
    "angle_span",
    "hinge_dist",
    "len_ratio",
    "center_dist",
    "radial_angle_deg",
    "tip_to_arc_dist"
  ],
  "scaler": {
    "type": "zscore",
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "std":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  },
  "weights": [0.1, 0.2, 0.3, -0.1, 0.4, -0.2, -0.3, -0.1],
  "bias": -0.3
}
```

Thresholding is intentionally configured separately, via:

- `configs/door_rules.json` → `output.min_confidence_after_reweight`

## Implementation checklist (what code you’d write)

### Detection runtime (inference)

- Load artifacts
- If `meta.mode` is `vector` or `hybrid`:
  - generate candidates
  - compute features
  - score with the reweighter (if present)
  - apply post-reweight threshold → final doors
- Write `doors.json` and `doors_overlay.png`

### Review UI

Minimum UI behaviors:

- Display `page.png` with predicted boxes
- List predictions with:
  - confidence
  - toggle **accept/reject**
- Support “Add missed door” with rectangle drawing (or two-click bounding box)
- Save a `labels.json` alongside the page artifacts

### “Fit reweighter” script (offline)

Given a folder of reviewed artifacts:

- Load all `doors.json` + `labels.json`
- Build a dataset:
  - positives = accepted predictions
  - negatives = rejected predictions
- Fit model weights (warm-started + regularized toward the prior for stability on small data)
- Apply minimum-data gating (don’t write a new model unless both classes exist and there is enough labeled data)
- Save `models/reweighter_<type>_v1.json` (one model per door type)

CLI:

```bash
door-detector-reweight --artifacts artifacts --out-dir models
```

Deployment tuning (precision/recall) is controlled via `configs/door_rules.json`:

- `output.min_candidate_confidence` (candidate volume)
- `output.min_confidence_after_reweight` (final keep/drop threshold)

## Why this option is often best in a short take-home

- **Fast to ship**: you can produce a compelling loop (detect → review → improve) without deep ML infra.
- **Strong narrative**: explainable geometry + data-driven scoring.
- **Matches dataset**: if most pages are vector PDFs, vector-first + reweighting will outperform a rushed pixel detector.

## When you should prefer full retraining (Option 2)

If the input PDFs are mostly scans (high `image_coverage`), vector primitives won’t exist.

In that case:

- keep the same `labels.json` format
- convert labels into YOLO/COCO
- fine-tune a small detector on `page.png` (tiling as needed)

That’s a bigger engineering effort and usually needs more labeled data and/or a GPU to iterate quickly.



