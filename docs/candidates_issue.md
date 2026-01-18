## Candidate list coverage issues (multi-PDF robustness)

This document captures an issue encountered while reviewing door detections in the Door Detector UI: **the candidate list does not include enough candidates across different PDF drafting styles**, which prevents snapping/manual selection from working outside of the “one working PDF” the heuristics were effectively tuned for.

### Background: what “candidates” are and why they matter

Door Detector’s review UI supports **Shift+drag** to draw a selection box and “snap” that box to a nearby detected door candidate. This is powered by `doors.json`:

- `doors.json.doors`: final filtered detections (what you review/confirm by default)
- `doors.json.candidates`: a broader pool used for snapping and training

The **snapper only snaps when a candidate bbox overlaps the drawn selection box**. If there are no overlapping candidates near the user’s selected door, snapping appears broken, but the root cause is usually **candidate generation failure** (not a viewer bug).

### Symptoms observed in the UI

When Shift+dragging around an obvious door, console logs showed:

- `poolCandidates` non-zero (a candidate pool exists)
- `overlapCandidates: 0` near the drawn box
- Therefore snapping returns “no match (no overlap)”

In these cases, the UI is behaving as designed: **no overlap ⇒ no snap**.

### Root causes (why candidates were missing on many PDFs)

#### 1) Swing detection originally assumed Bezier arcs

Many floor plan PDFs represent door swing arcs as **cubic Beziers**, but many others approximate arcs as **polylines** (chains of short line segments).

There were two failures modes:

- **No Beziers near the door** ⇒ the swing detector could not see the arc at all.
- Worse: swing detection was *gated* on `beziers` being non-empty, which meant **swing detection was skipped entirely** on “polyline-only arc” PDFs.

Fix applied:
- Swing detection now runs if swing is enabled and **either** `beziers` **or** `lines` exist.

#### 2) Candidate heuristics were scale-sensitive (absolute pixels)

Several rules are expressed in **absolute pixel units** at the Step 1 render DPI (typically 400):

- `swing.arc.min_radius_px`, `max_radius_px`
- `swing.arc.max_circle_fit_rmse`
- Leaf pairing ratios depend on measured pixel distances

Different PDFs can have different effective symbol scales (line weights / drawing scale / page size), so a “good” door arc in one PDF might violate `max_radius_px` or related thresholds in another.

This contributes to behavior that looks like “tuned for one PDF”.

#### 3) Circle-cluster suppression removes door-tag bubbles (by design)

Many plans include circled labels like “6”, “15”, etc. These are often drawn as **4× 90° arcs**, which together form a full 360° circle.

The swing detector includes a deliberate filter:

- `swing.arc.suppress_circle_clusters`
- cluster bins: `circle_cluster_center_bin_px`, `circle_cluster_radius_bin_px`
- suppression thresholds: `circle_cluster_min_arcs`, `circle_cluster_min_total_angle_deg`

This prevents false door detections from labels/stamps/symbols.

Implication:
- If the user’s selection box covers primarily a **door label bubble**, there may be **no door candidates overlapping** that box (because the arcs are suppressed).
- This can look like “a good selection ROI didn’t snap”, but the ROI is overlapping a suppressed symbol, not a door candidate.

To make this diagnosable, we added debug output to explicitly report when nearby arcs are “suppressed by circle-cluster filter”.

#### 4) Leaf-pairing assumptions don’t match some drafting styles

Even when a valid arc exists near the door, candidate formation requires pairing the arc with a “leaf” line segment that meets several constraints:

- leaf length ratio vs radius
- hinge proximity ratio
- optional “endpoint near center” constraint
- radial angle limit
- tip-to-arc distance ratio

Some PDFs draw leaves as multiple segments, different topology, or “broken” lines; the pairing logic can then fail (`pool_pass: 0` / `strict_pass: 0`) even though the arc itself is clearly door-like.

This reduces candidate recall and prevents snapping.

### Debugging tools added during investigation

#### Unmatched-box debug report (browser console)

When Shift+drag results in no snap, the backend now emits a structured debug report and the viewer prints it to the browser console:

- nearby primitive counts (`lines_near_roi`, `beziers_near_roi`)
- arc qualification failures: radius / rmse / angle span
- leaf pairing failure counts (pool vs strict)
- circle-cluster suppression info (when enabled)

This is designed to answer: **“Were there even any usable vectors here? If yes, which filters blocked them?”**

#### Persist unmatched boxes during an edit session

Unmatched boxes are stored in the edit draft and re-rendered after reruns (instead of disappearing), so the user can iteratively adjust or compare.

### Candidate-generation improvements implemented

1) **Polyline-arc support**
- Added detection of arc-like polylines from short line segments (endpoint-snapped chains), circle-fit, and angle-span validation.
- These arcs feed into the same swing candidate pipeline as Bezier arcs.

2) **Fix swing gating**
- Swing detection is no longer skipped just because `beziers` is empty.

3) **Candidate pool de-duplication**
- Candidate pool is deduplicated by stable candidate id to prevent duplicate candidates crowding out diversity.

### Important UX constraint (snap behavior)

The snapper’s core rule remains:

- **Only snap if a candidate bbox overlaps the drawn selection box**.

We intentionally did *not* change snapping to “nearest candidate” when overlap is 0, because that can create surprising snaps far outside the selection and breaks the mental model for review labeling.

### How to reproduce / validate

1) Pick a failing PDF and Shift+drag over a door that should be selectable.
2) If it does not snap, copy the `unmatched_debug_report parsed` object from the browser console.
3) Inspect:
   - `beziers_near_roi == 0` and `lines_near_roi > 0`: likely polyline arc case
   - `arc_fail_counts_near.radius` high: scale thresholds too tight
   - `arc_circle_cluster_suppression.near_examples[].suppressed == true`: selection overlaps a label bubble, not a door candidate
   - `leaf_pair_stats_near.pool_pass == 0`: arc is present but leaf-pairing rules don’t match drafting style

### Remaining work / recommended next steps

- **Make arc thresholds scale-aware**:
  - Replace absolute `min_radius_px`/`max_radius_px` with heuristics based on page scale, stroke width distributions, or learned calibration.

- **Improve leaf pairing robustness**:
  - Allow leaf to be a short polyline chain (not only a single line segment)
  - Add additional leaf candidates derived from nearby line clusters
  - Consider weakening `require_endpoint_near_center` or radial/tip checks for the pool (keep strictness for final doors)

- **Broaden the candidate pool without exploding false positives**:
  - Keep conservative `doors` selection, but increase candidate recall for snapping/training.
  - Use reweighting to score down false positives rather than dropping them early.

- **Add an “arc-only” review aid only if needed**:
  - If leaf pairing is a persistent blocker, consider exporting “arc-only” candidates for snapping/training, but ensure they are excluded from final doors.

