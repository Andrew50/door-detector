# Door selection + learning loop (propose → score → review → retrain)

This document explains how Door Detector goes from **vector primitives** to:

- a **broad candidate pool** (used for snapping + training),
- a **final doors list** (what the system “predicts”),
- and how the **Edit Doors** UI turns user actions into labels that feed the reweighter.

It also calls out where the “hardcoded constraints” live vs where the “learned model” applies.

---

## Artifacts on disk

After Step 2 runs, each page artifacts directory contains `doors.json`, which includes:

- `doors`: the **final** strict, post-threshold/post-NMS predictions shown as door overlays
- `candidates`: a **broader** candidate pool (looser geometry gating), used for snapping and training

Step 2 writes both fields here:

- `door_detector/step2_pipeline.py` (`run_step2`) writes `doors` and `candidates` into `doors.json`.

---

## Detection pipeline (Step 2): propose → score → NMS

All door detection happens in:

- `door_detector/doors/detect.py` (`detect_doors`)

### 1) Propose candidates (geometry constraints)

The system uses vector primitives from `primitives.json`:

- Beziers (door swing arcs)
- Lines (door leaf lines)

It proposes candidates by:

1. fitting a circle to sampled Bezier points,
2. applying **arc thresholds** (min/max radius, max RMSE, min/max angle span),
3. searching nearby lines with a spatial index,
4. applying **leaf thresholds** (len ratio, hinge proximity, “endpoint near center”, radial alignment, etc.).

There are two “tracks” of thresholds:

- **Strict thresholds**: mostly driven by `configs/door_rules.json` under `swing.arc` and `swing.leaf`
- **Pool thresholds**: a *looser* set of thresholds, currently hardcoded in `detect_doors`, to keep a broader set of plausible candidates for snapping/training

### 2) Score candidates (heuristic confidence)

Each candidate gets a heuristic confidence composed from sub-scores:

- circle fit quality (RMSE)
- how close arc angle span is to a typical swing
- hinge proximity

These weights come from config (`swing.scoring` in `configs/door_rules.json`).

### 3) Split into two outputs

`detect_doors()` builds:

- `candidates` (**candidate pool**):
  - looser gating
  - contains near-misses that may not pass strict constraints
  - sorted and capped (to keep `doors.json` manageable)

- `doors` (**final predictions**):
  - strict geometry gates
  - must also pass `output.min_confidence`
  - then NMS (IoU-based) is applied to produce the final list

### 4) Apply the learned reweighter (optional)

If `configs/door_rules.json` includes `reweighter_path` and the model file exists,
`detect_doors()` applies the reweighter to **both**:

- the final strict list (`doors`)
- the broader pool (`candidates`)

The reweighter replaces each candidate’s `confidence` with a logistic-regression probability computed from its feature vector.

This is implemented in:

- `door_detector/doors/detect.py` (`apply_reweighter`)

---

## What is “hardcoded constraints” vs “learned model”?

### Hardcoded / rule-based parts

These determine **which candidates exist at all**.

- **Config-driven thresholds**: `configs/door_rules.json`
  - `swing.arc.*` (radius, angle, RMSE, cluster suppression settings)
  - `swing.leaf.*` (len ratio, hinge distance, radial/tip constraints)
  - `output.min_confidence`, `output.nms_iou`, `output.max_doors`

- **Hardcoded pool looseness**: `door_detector/doors/detect.py`
  - “pool_*” thresholds expand the set of candidates exported in `doors.json["candidates"]`
  - these do **not** change what becomes a final “door” unless the rest of the pipeline uses the pool to make final decisions

### Learned reweighter

This only changes candidate `confidence` (ranking), based on a feature vector.
It does not change geometry or create new candidates.

---

## UI selection: Edit Doors (Shift+drag snapping)

The Streamlit UI is implemented in:

- `door_detector/review_app.py` (thin Streamlit entrypoint wrapper)
- `door_detector/ui/app.py` and `door_detector/ui/*` (actual UI implementation)

### What the user does

- **Confirm door**: marks a candidate id as a positive (door)
- **Delete / Not a door**: marks a candidate id as a negative
- **Edit Doors → Shift+drag**: draws a selector rectangle; the app snaps it to a candidate and confirms it

### How snapping works (two stages)

There are two snap computations:

1) **Viewer-side snap (fast UX)**  
Inside the HTML/JS pan+zoom viewer (`_panzoom_image_viewer`), `snapCandidateForDraw()` tries to pick the best candidate for the drawn box:

- Prefer a **server-supplied candidate pool** (sent via a hidden Streamlit sink).
- If unavailable, fall back to snapping against the currently rendered SVG overlay door boxes.

This is only used to propose an id quickly for UX (e.g., drawing a green snapped overlay immediately).

2) **Server-side snap (authoritative)**  
When Streamlit receives the draw event, `_process_draw_event_if_any()`:

- converts preview coords → full-res coords
- loads candidates from `doors_data["candidates"]` (fallback to `doors_data["doors"]`)
- validates any client-proposed `snapped_candidate_id` (must overlap)
- otherwise runs its own IoU/intersection-based snap search

The server result is what gets written into `labels.json` as `snapped_candidate_id`.

---

## Label storage (`labels.json`, schema v2)

Each artifacts directory can store a `labels.json` with schema version 2:

- `confirmed_ids`: positive labels (doors)
- `deleted_ids`: negative labels (not doors)
- `manual_additions`: records of Shift+drag selections (includes drawn box + snapped candidate id + IoU)
- `unmatched_manual_boxes`: selector boxes that did not match any candidate (UI-only; not used for training)

The UI is intentionally **v2-only**: legacy fields are rejected.

On re-analysis, the UI re-applies:

- deletions (hide removed doors)
- confirmations (render confirmed doors as green)

---

## Training: reweighter (“retrainer”)

Training is implemented in:

- `door_detector/reweight_fit.py` (`fit_reweighter`)

Inputs:

- `artifacts/**/labels.json` (schema v2 only)
- corresponding `artifacts/**/doors.json`

Training samples:

- Positive: any candidate id in `confirmed_ids`
- Negative: any candidate id in `deleted_ids`
- Ignored: detected-but-unlabeled candidates, and unmatched manual boxes

Feature vector:

- The trainer uses a fixed feature list (currently): `rmse`, `radius`, `angle_span`, `hinge_dist`, `len_ratio`
- It reads features from `doors.json["candidates"]` (fallback to `doors.json["doors"]`)

Model:

- standardize features (mean/std)
- logistic regression with gradient descent
- writes `models/reweighter_v1.json` (weights + scaler + bias)

---

## Known limitations / tuning notes

- **Candidate ids stability**: ids are currently derived from primitive indices (Bezier index + line index). If the primitive ordering changes, ids can drift across reruns.
- **Pool thresholds are hardcoded**: the “looser pool” constraints live in `door_detector/doors/detect.py` today. If you need them tunable per deployment, move them into config.
- **Reweighter scope**: it can only rescore candidates that exist (and only uses a small feature set unless expanded).

