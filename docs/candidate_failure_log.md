## Candidate failure log

This file records concrete “door looked real but wasn’t in `doors.json["candidates"]`” cases, what geometric criterion failed, and what change was made to improve recall (without regressing previously-detected candidates).

---

### 2026-01-19 — `f_1768798404156` — arc missing/rasterized; leaf-only enabled but hinge wall-support gate too brittle (thin short wall segments)

- **Symptom (UI)**: Edit Doors → Shift+drag produced:
  - `snapCandidateForDrawPdf no match (no overlap)`
  - `unmatched_debug_report.summary.primary_failure: "no_arc_primitives_near_roi"`
- **What geometry exists** (from `unmatched_debug_report.summary.counts`):
  - `beziers_near_roi == 0` and `polyline_arc_candidates_near == 0` (no vector arc primitives near ROI)
  - `non_dashed_non_axis_lines_near_roi > 0` (diagonal “leaf-like” linework exists)
  - `leaf_only_candidates_near == 0` (leaf-only generator produced nothing in that ROI)
- **Why it failed to become a candidate**:
  - Arc-first swing detection requires an arc primitive (bezier or polyline) to seed a candidate; with no arc primitives, no `swing`/`swing_arc` candidate can be created.
  - `swing.leaf_only` was enabled, but its hinge “wall support” evidence was too brittle when the wall/jamb near the hinge is represented as **many thin short axis-aligned segments**; no endpoint saw a single support segment long enough to satisfy the support gate, so no `swing_leaf` pool candidates were emitted.
- **Detection change**:
  - Updated `detect_swing_leaf_only_candidates._axis_support` to use **aggregate axis support length** (sum of nearby axis-aligned segment lengths within the hinge neighborhood), rather than requiring a single long support segment. This improves recall for leaf-only doors while keeping these candidates **pool-only** and low-confidence (snapping/labeling only).
  - Augmented `unmatched_debug_report` to include `swing.leaf_only_near.debug` and `summary.top_leaf_only_fail` so future cases can immediately report which leaf-only gate blocked candidate creation.

### 2026-01-19 — (Edit Doors failing door) — swing arc drawn as **dashed polyline segments** → no swing candidate

- **Symptom (UI)**: Edit Doors → Shift+drag around an obvious swing door produced `overlapCandidates: 0` and `snapCandidateForDrawPdf no match (no overlap)`. The intended door did not appear in **Selection matches** because it did not exist in `doors.json["candidates"]`.
- **What geometry exists**:
  - The swing arc is present, but it is represented as **dashed short line segments** (a polyline approximation of the arc), not Beziers (`beziers_near_roi == 0`).
- **Why it failed to become a candidate**:
  - `_extract_polyline_arcs_from_lines` previously **skipped dashed line primitives entirely**, so dashed arc chains never entered the polyline-arc circle-fit stage.
  - With no bezier arcs and no accepted polyline arcs, arc-first swing detection could not generate `swing` / `swing_arc` candidates for that door.
- **Detection change**:
  - Added `swing.polyline_arc.allow_dashed` (configurable) and enabled it in `configs/door_rules.json` so dashed short segments can participate in polyline-arc extraction (still subject to radius/RMSE/angle-span filters and circle-cluster suppression).

### 2026-01-19 — `f_1768786852390` — swing arc present but polyline-arc topology rejection

- **Symptom (UI)**: Edit Doors → Shift+drag around a swing door near the `S22` tag snapped to nearby `bifold` candidates with very low IoU (\(\approx 0.034\)), and the intended swing door was not in **Selection matches**.
- **What geometry exists**:
  - The swing arc is present, but it is represented as a **polyline** (chain of short `lines` primitives), not Beziers (`beziers_near_roi == 0`, `lines_near_roi > 0`).
  - Locally, the arc chain is ~6 short segments (global line idxs around `48018..48023`).
- **Why it failed to become a candidate**:
  - The polyline-arc extractor previously required the short-segment component to have **exactly two degree-1 endpoints** (a simple open chain).
  - In this door, the arc endpoint is “attached” to nearby short geometry (leaf/jamb stubs), producing a component with **only one degree-1 endpoint** (topology not equal to 2 endpoints), so the arc was rejected and no `swing`/`swing_arc` candidate was created.
- **Detection change**:
  - Updated `_extract_polyline_arcs_from_lines` (and the unmatched debug mirror `_debug_polyline_arcs_from_lines_subset`) to **recover an arc-like path** in small branched components by searching for a best-fitting simple path starting at the available endpoint and ending at a nearby junction-ish node, subject to the same circle-fit + angle-span thresholds.
  - Added **guardrails** (component size / endpoint count / path search caps) to avoid path-explosion and keep runtime reasonable.

### 2026-01-18 — Swing door with vector leaf but missing arc primitives → no `swing` candidate

- **File**: `artifacts/library/f_1768786852390`
- **Selection ROI (full-res px)**: `[8293.58, 3837.83, 8635.52, 4079.86]`
- **Observed behavior**:
  - Shift+drag produced `overlapCandidates: 1` and snapped to `d_e7ccbae8182c` with IoU ≈ 0.0216.
  - The snapped candidate was **not** the intended door; it was a low-confidence `bifold` candidate overlapping only a tiny region of the drawn box.

- **Door geometry in primitives (why it wasn’t a candidate)**:
  - `beziers_near_roi == 0` (no vector bezier arc near the selection)
  - A plausible **solid diagonal “leaf” line** exists (non-dashed, non-axis linework), but the **swing arc** is not present as vector primitives (and polyline-arc extraction in this ROI only finds dashed/track-like components, not a door arc).
  - Since swing detection is currently **arc-first**, the missing arc prevents generating a `swing` candidate even though a leaf line exists.

- **Missed criterion / gate**:
  - Swing candidate generation requires **arc geometry** (bezier or polyline-arc) to anchor center/radius; leaf pairing is performed only *after* an arc passes filters.
  - In this case, the **leaf is vector** but the **arc is missing**, so the detector never enters the leaf-pairing stage for the intended door.

- **Change made (to make this labelable without weakening global detection)**:
  - **Candidate generation**: added a conservative `swing_leaf` **leaf-only** candidate generator (enabled via `configs/door_rules.json → swing.leaf_only`) that proposes candidates from diagonal leaf-like lines whose hinge endpoint is near a wall corner (axis-aligned line support). These candidates are **pool-only** and low-confidence (intended for snapping/labeling, not auto-selection).
  - **Snap robustness**: tightened snap rules so we don’t “snap” on tiny corner overlaps (raised `MIN_SNAP_IOU`, added coverage + intersection-fraction gating).
  - **Manual candidate fallback**: treat *extremely weak* snaps (IoU < 0.04 with debug_reason `weak_snap_low_iou`) as unmatched so the UI creates a **manual box candidate** for labeling.
  - **Log augmentation**: added `unmatched_debug_report.summary` (and a server-side `[door_detector] unmatched_debug_summary` line) to surface the key “missing geometry” signals without needing to copy huge verbose JSON blobs.

---

### 2026-01-19 — `f_1768786633852` — swing arc drawn as polylines attached to leaf/jamb (loop + oversized component)

- **Symptom (UI)**: Edit Doors → Shift+drag around the swing door (left of the thick wall) snapped to a nearby `bifold` candidate with **very low IoU** (\(\approx 0.003\)); the intended swing door was missing from **Selection matches** because it did not exist in `doors.json["candidates"]`.
- **What geometry exists**:
  - The swing arc is represented as a **polyline** (short `lines` primitives), not Beziers (`beziers_near_roi == 0`).
  - The arc shares endpoints with the **axis-aligned** leaf/jamb lines, forming a small cycle (“loop-with-branches”) in the short-segment graph.
- **Why it failed to become a candidate**:
  - The polyline-arc extractor used **all** short segments, including axis-aligned leaf/jamb/wall stubs. That can (a) merge into a very large connected component and/or (b) produce **0 degree-1 endpoints** (cycle), causing the extractor to skip the component and never emit an arc → no `swing`/`swing_arc` candidate.
- **Detection change**:
  - In `_extract_polyline_arcs_from_lines` (and `_debug_polyline_arcs_from_lines_subset`):
    - **Exclude perfectly axis-aligned segments** from the polyline-arc graph (leaf pairing still uses all lines), preventing runaway connectivity and separating the arc chain from leaf/jamb stubs.
    - **Allow loop-with-branches recovery**: when a component has `end_nodes == 0` but contains a junction node (degree != 2), search for a best-fitting simple path that satisfies the same circle-fit + angle-span thresholds.
  - Result: the intended door now appears as `swing` candidates with bboxes starting around `x0≈7212` (instead of missing entirely).

---

### 2026-01-19 — `f_1768786852390` — leaf line exists but `swing_leaf` leaf-only gate too strict (hinge not at wall endpoint)

- **Symptom (UI)**: Edit Doors → Shift+drag around an obvious swing door produced `overlapCandidates: 0` and `snapCandidateForDrawPdf no match (no overlap)`. The intended door did not exist in `doors.json["candidates"]` (only the manual-box fallback was available).
- **Selection ROI (full-res px)**: `[9390.44, 6469.33, 9836.69, 6894.71]` (from `unmatched_debug_report.bbox_full_xyxy`)
- **What geometry exists**:
  - No bezier arcs near ROI (`beziers_near_roi == 0`).
  - A few polyline-arc fits exist but fail `swing.arc.min_angle_deg` (fragmented dashed arc segments; `polyline_arc_candidates_near > 0` but `polyline_arc_pass_near == 0`, `top_arc_fail == "angle"`).
  - Solid diagonal (leaf-like) linework exists near ROI (`non_dashed_non_axis_lines_near_roi > 0`).
- **Why it failed to become a candidate**:
  - Arc-first swing detection produced no `swing`/`swing_arc` candidates because no arc passed thresholds.
  - `swing.leaf_only` was enabled, but its corner-support gate required the hinge endpoint to be near a *wall line endpoint*. In this plan, the hinge point can lie on the *interior* of long axis-aligned wall segments, so no `swing_leaf` candidates were created.
- **Detection change**:
  - Loosened `swing_leaf` corner support to use **point→segment distance** (hinge near a wall segment), not just endpoint proximity. This increases recall for “leaf is vector, arc missing/fragmented” cases without affecting final auto-selected doors (these remain pool-only, low-confidence candidates).
  - Further loosened the “corner” requirement: allow **single-axis wall support** (hinge on a long wall segment, not necessarily at a wall corner), with guardrails:
    - tip endpoint must be farther from nearby wall segments than the hinge endpoint
    - leaf direction must not be near-parallel to the supporting wall direction
  - Augmented `unmatched_debug_report.summary.counts` with `leaf_only_candidates_near` so we can immediately tell whether leaf-only candidate generation is (or isn’t) helping in a given ROI.

---

### 2026-01-19 — `f_1768793538233` — swing/double doors drawn as **polyline arcs** → rejected by absolute circle-fit RMSE (missing door candidates)

- **Symptom (UI)**: Edit Doors → Shift+drag around obvious doors (e.g. labels `114A`, `102`, `106A`) produced only small `swing_arc` snaps with very low IoU (\(\approx 0.03\)–\(0.05\)). The intended door was **not present** as a proper `swing`/`double` candidate in **Selection matches**, and the final `doors_overlay.png` showed **no detected door** in those regions.
- **What geometry exists**:
  - The door swing arc is present as **vector line segments** (a polyline chain), not Beziers (`beziers_near_roi` was dominated by nearby annotation/bubble curves).
  - In ROI1 (`114A`), polyline-arc extraction found a plausible arc with radius \(\approx 135\) px and angle span \(\approx 102^\circ\), but the fitted circle RMSE was \(\approx 4.2\) px.
  - In ROI3 (`106A`, double door), a single extracted polyline chain contained **two concatenated arcs**, so fitting the whole chain yielded a very large angle span and very high RMSE, causing rejection.
- **Why it failed to become a candidate**:
  - Polyline arcs were filtered using a single **absolute** `swing.arc.max_circle_fit_rmse` threshold (2.5 px). For larger-radius door arcs, a few pixels of absolute error is still visually acceptable, but was rejected.
  - When a component contains multiple concatenated arcs (double doors / double-acting symbols), fitting the entire chain at once produces bad RMSE/angle and no sub-arc recovery was attempted.
- **Detection change**:
  - Added `swing.arc.max_circle_fit_rmse_ratio` (relative tolerance \( \mathrm{RMSE}/r \)) and enabled it in `configs/door_rules.json` to accept large-radius polylines with reasonable *relative* fit error.
  - Updated polyline-arc extraction (`_extract_polyline_arcs_from_lines` and `_debug_polyline_arcs_from_lines_subset`) to attempt **contiguous subpath recovery** when the full chain fails, so a valid 90-ish degree arc can be recovered from a longer concatenated chain.
  - Augmented `unmatched_debug_report.summary` with `top_polyline_arc_fail` so we can quickly see when the missing door is due to polyline arc RMSE/angle rejection.

### 2026-01-19 — `f_1768793538233` — candidates exist but snap rejected as “candidate too small” when ROI includes room tag bubble

- **Symptom (UI)**: Edit Doors → Shift+drag around a swing door (including the room tag bubble inside the arc) produced:
  - `snapCandidateForDrawPdf no match (coverage candidate too small)` (client-side)
  - `unmatched_debug_report extra.debug_reason: "weak_snap_low_iou"` (server-side)
  - Result: `snap: null` and the UI fell back to a manual-box candidate even though door candidates overlapped.
- **What geometry exists**:
  - Valid `swing` / `swing_arc` candidates overlap the ROI on the server (e.g. `swing_arc` candidates with \( \mathrm{IoU} \approx 0.013 \) in that ROI).
- **Why it failed to snap**:
  - The snap logic includes an anti-false-positive guardrail that rejects coverage/intersection snaps when the candidate covers <6% of the drawn area (`MIN_INTER_FRAC_OF_DRAWN = 0.06`).
  - When the reviewer includes a **large room tag bubble** inside the swing arc in the drawn box, the ROI becomes much larger than the door candidate bbox, driving \( \mathrm{inter\_frac} \) down to \(\approx 0.01\) and causing an unnecessary rejection.
- **Change made**:
  - Kept the strict 6% guardrail for generic symbol-like candidates, but introduced a relaxed threshold for **door-like** candidate types with reasonable confidence:
    - `MIN_INTER_FRAC_OF_DRAWN_DOORLIKE = 0.01` (applies to `swing`, `swing_arc`, `double`, `pocket`, `bifold`, `swing_leaf` when `confidence >= 0.55`)
  - Implemented in both:
    - `door_detector/ui/pdfjs_component/frontend/src/pdfjs_viewer.tsx` (`snapCandidateForDrawPdf`)
    - `door_detector/ui/app.py` (`_snap_to_candidate`)

### 2026-01-19 — `f_1768786633852` — swing arc missing; leaf-only candidates blocked by “wall support line too long” gate

- **Symptom (UI)**: Shift+drag produced `snap: null` (client snap rejected as “candidate too small”), and the unmatched report showed `polyline_arc_candidates_near: 1` with `top_polyline_arc_fail: "angle"` (the only arc-like polyline was a tiny \(\approx 16^\circ\) fragment, not a door swing).
- **What geometry exists**:
  - Clear diagonal leaf linework exists near ROI (`non_dashed_non_axis_lines_near_roi > 0`), but there is **no 90° swing arc** in vector primitives (door is drawn “leaf-only”).
  - Axis-aligned wall/jamb support near the hinge is present, but rendered as **short thick segments** (e.g. 20–35 px chunks), not long continuous lines.
- **Why it failed to become a candidate**:
  - `swing.leaf_only` was enabled, but its hinge support gate required **individual** axis-aligned support segments to exceed `min_axis_support_length_px` (default 40 px), so these short wall fragments were ignored and no `swing_leaf` candidates were created (`leaf_only_candidates_near: 0`).
- **Detection change**:
  - Updated `detect_swing_leaf_only_candidates._axis_support` to allow **thick** short axis-aligned segments to count as wall support, using two new leaf-only knobs:
    - `swing.leaf_only.wall_support_min_stroke_width` (default `1.0`)
    - `swing.leaf_only.wall_support_min_segment_length_px` (default `12.0`)
  - Updated the unmatched summary classification so this case reports as `no_arc_passed_thresholds:angle` (rather than `no_arc_primitives_near_roi`) when polyline arcs exist but fail thresholds.
