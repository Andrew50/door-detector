## Feedback learning (Option 1): Reweight / rerank vector candidates

This document describes the recommended “learning from feedback” approach for this project: **keep a vector-first door detector**, but improve it over time by **reweighting features and recalibrating thresholds** based on reviewer feedback.

This is intentionally lightweight:

- **No deep learning required**
- **Fast iteration** (seconds, CPU-only)
- **Explainable** (you can show which geometric cues drove a decision)
- Works especially well when PDFs are **true vector/CAD** (as observed in `docs/vector_vs_raster_analysis.md`)

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

- `models/reweighter_v1.json`: feature normalization + weights + threshold(s)

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

Recommended format (example):

```json
{
  "schema_version": 1,
  "page_id": "floor_plan_01",
  "generated_at": "2026-01-14T12:34:56Z",
  "detector": {
    "name": "vector_candidate_v1",
    "reweighter": "models/reweighter_v1.json"
  },
  "doors": [
    {
      "id": "d_000123",
      "bbox_xyxy": [120.5, 340.2, 220.1, 430.9],
      "confidence": 0.92,
      "source": "vector",
      "candidate": {
        "arc_refs": ["bezier:4812"],
        "leaf_refs": ["line:10291"]
      },
      "features": {
        "arc_radius_px": 42.1,
        "arc_angle_deg": 88.3,
        "arc_fit_error": 0.012,
        "leaf_length_px": 39.9,
        "hinge_gap_px": 1.6,
        "stroke_width_ratio": 1.02,
        "local_line_density": 0.31
      }
    }
  ]
}
```

Notes:

- Keeping `features` in the output makes the system **auditable** and helps debugging.
- `candidate.arc_refs` / `leaf_refs` are optional but useful when investigating failures.

### `labels.json` (feedback)

Keep it intentionally simple:

```json
{
  "schema_version": 1,
  "page_id": "floor_plan_01",
  "reviewed_at": "2026-01-14T12:40:00Z",
  "accepted_ids": ["d_000123", "d_000130"],
  "rejected_ids": ["d_000124"],
  "added_boxes": [
    {
      "bbox_xyxy": [512.0, 220.0, 605.0, 310.0],
      "note": "missed door near stair core"
    }
  ],
  "notes": "Many false positives around curved furniture."
}
```

Interpretation:

- **Accepted** predictions become positive examples.
- **Rejected** predictions become negative examples.
- **Added boxes** represent false negatives. They can be used later for:
  - improving candidate generation rules, and/or
  - training a pixel detector (bonus/Option 2)

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
  "schema_version": 1,
  "model_type": "logreg_l2",
  "feature_order": [
    "arc_angle_deg",
    "arc_radius_norm",
    "arc_fit_error",
    "leaf_length_norm",
    "hinge_gap_norm",
    "stroke_width_ratio",
    "local_line_density"
  ],
  "scaler": {
    "type": "zscore",
    "mean": [90.1, 8.2, 0.02, 7.9, 0.3, 1.01, 0.28],
    "std":  [12.7, 2.1, 0.01, 1.9, 0.2, 0.08, 0.10]
  },
  "weights": [1.2, 0.9, -2.5, 0.7, -1.0, 0.4, -0.8],
  "bias": -0.3,
  "decision_threshold": 0.65
}
```

## Implementation checklist (what code you’d write)

### Detection runtime (inference)

- Load artifacts
- If `meta.mode` is `vector` or `hybrid`:
  - generate candidates
  - compute features
  - score with the reweighter (if present)
  - apply threshold → final doors
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
- Fit model weights
- Choose threshold to hit a target:
  - e.g., “maximize F1” or “precision ≥ 0.9”
- Save `models/reweighter_v1.json`

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


