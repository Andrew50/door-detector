# Debugging Edit Doors selection/snapping issues

This doc explains how to debug cases where **Edit Doors → Shift+drag** does **not** surface the door you intended in the **Selection matches** suggestions (or snaps to the wrong thing).

It is written for the “PDF.js viewer + Streamlit server” flow currently used in Door Detector.

---

## Mental model (what has to go right)

### Key principle
**You can only label what exists as a candidate.**

- If the detector did not create a candidate (in `doors.json["candidates"]`), the UI cannot magically snap to the intended door.
- If a candidate exists but doesn’t overlap your drawn region, snapping (correctly) finds nothing.

### Two separate computations

#### 1) Viewer-side snap (fast UX)
Implemented in `door_detector/ui/pdfjs_component/frontend/src/pdfjs_viewer.tsx` (`snapCandidateForDrawPdf`).

- Uses the **frontend** `candidatePool` passed by Python (downsampled).
- Outputs logs like:
  - `[door_detector] snapCandidateForDrawPdf …`
  - `[door_detector] snapCandidateForDrawPdf chosen …` / `no match …`

#### 2) Server-side snap (authoritative)
Implemented in `door_detector/ui/app.py` (`_process_draw_event_if_any`).

- Uses **full** candidate list from `doors.json["candidates"]` (fallback to `doors.json["doors"]`).
- Produces a ranked `fstate["_last_draw_suggestions"]` which powers the right-panel **Selection matches** UI.
- Emits `unmatched_debug_report` (printed by the viewer) when it cannot match to a candidate.

---

## What to collect when it “doesn’t snap”

### Always capture these
- **The snap log block** from the browser console:
  - `[door_detector] snapCandidateForDrawPdf …`
  - `[door_detector] snapCandidateForDrawPdf chosen …` or `no match …`
  - `[door_detector] draw_rect endDrag (pdfjs) …` (include `drawn_pdf_xyxy`)
- If printed, the full:
  - `[door_detector] unmatched_debug_report raw {…}`
- The **file id** (e.g. `f_1768774913300`) and whether you just **Re-analyzed**.

### Optional (enable only when needed)
To enable PDF.js component lifecycle logs:

```js
window.__door_detectorPdfjsDebug = true
```

This is useful to detect iframe remounts, which can break messaging / selection updates.

### If you see: “Received component message for unregistered ComponentInstance!”
This warning is emitted by Streamlit’s host page when the PDF.js component posts an event
at a moment where Streamlit has temporarily torn down (or not yet registered) that component
instance (often during a rerun).

- **Usually harmless**: a rerun immediately after a draw/click can trigger this transiently.
- **Actionable if events seem “ignored”**: enable `window.__door_detectorPdfjsDebug = true` and capture
  the nearby mount/unmount logs to confirm whether the iframe is being replaced during your interaction.

---

## First triage: snap vs candidate generation

### A) “overlapCandidates: 0” in the snap log
This means: **none of the candidates the viewer knows about overlap the drawn box**.

Common causes:
- You drew a box that misses the candidate bbox (padding can surprise you).
- The relevant candidate exists on the server but wasn’t included in the **downsampled** frontend pool.
- There truly is no candidate for that door.

Next step: use the `unmatched_debug_report` (if present) and/or run the overlap check script below.

### B) “overlapCandidates > 0 but chosen is wrong”
This means: the intended door was either:
- not in the overlap set, or
- in the overlap set but ranked lower than a false-positive symbol.

Next step: use **Selection matches** + type filter to cycle.
If the intended door is not in suggestions, it likely does not exist as a candidate (or doesn’t overlap).

---

## Determining “what geometry exists here” (server-side truth)

### 1) Check whether any candidates overlap your ROI (pixel space)

1. Find the `bbox_full_xyxy` from `unmatched_debug_report raw` (it is already in full-res pixel space).
2. Run:

```bash
/home/aj/dev/door_detector/.venv/bin/python - <<'PY'
import json
from pathlib import Path

file_dir = Path("artifacts/library/<FILE_ID_HERE>")
doors = json.loads((file_dir/"doors.json").read_text())
cands = list(doors.get("candidates") or [])

bbox = <PASTE_bbox_full_xyxy_LIST_HERE>

def norm(b):
    x0,y0,x1,y1 = map(float,b)
    return [min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)]

def inter_area(a,b):
    ax0,ay0,ax1,ay1 = norm(a)
    bx0,by0,bx1,by1 = norm(b)
    ix0=max(ax0,bx0); iy0=max(ay0,by0)
    ix1=min(ax1,bx1); iy1=min(ay1,by1)
    iw=max(0, ix1-ix0); ih=max(0, iy1-iy0)
    return iw*ih

hits=[]
for c in cands:
    bb=c.get("bbox_xyxy")
    if not (isinstance(bb,list) and len(bb)==4):
        continue
    ia=inter_area(bbox, bb)
    if ia>0:
        hits.append((ia, c))

hits.sort(key=lambda x: x[0], reverse=True)
print("total candidates:", len(cands))
print("overlapping candidates:", len(hits))
for ia,c in hits[:10]:
    print(" - overlap", round(ia,2), "id", c.get("id"), "type", c.get("type"), "conf", c.get("confidence"))
PY
```

Interpretation:
- **0 overlaps**: no candidate bbox intersects your ROI. Either the door has no candidate, or the candidate bbox is elsewhere.
- **>0 overlaps but all wrong type**: candidate generation exists but is mis-typed or mislocalized; cycling/type filtering should help.

### 2) Use the unmatched debug report to find the failing stage

When snapping finds no match, the server emits `unmatched_debug_report` with a structured explanation.

Key fields:
- `counts.beziers_near_roi`
- `swing.arc_pass_near_count` and `swing.arc_fail_counts_near`
- `swing.leaf_pair_stats_near.pool_pass` / `strict_pass`
- `swing.polyline_arc_near.rejected_components` (especially `component.topology`)
- `pocket.near_hits` (for pocket doors)

#### Common patterns

##### Pattern 1: `beziers_near_roi == 0` and `swing.arc_pass_near_count == 0`
There is **no arc geometry** detected in the primitives near your selection.

Likely causes:
- The arc is rasterized (scan-like) or not represented as vectors.
- The arc is represented as **line segments**, but polyline-arc extraction is failing.

Knobs:
- `swing.polyline_arc.*` in `configs/door_rules.json`
  - `endpoint_snap_px`: increase to reconnect broken arcs
  - `min_segments`: decrease for chunkier arcs
  - `max_segment_length_px`: increase for longer arc segments
  - `allow_branches`: enable to recover arcs “touched” by other geometry

##### Pattern 2: `swing.arc_pass_near_count > 0` but `leaf_pair_stats_near.strict_pass == 0`
Arcs exist, but the detector cannot find a valid **leaf line** pairing, so you often only get `swing_arc` (arc-only) candidates.

Knobs (strict leaf gating lives in config):
- `swing.leaf.max_hinge_dist_ratio`
- `swing.leaf.require_endpoint_near_center` (try `false` for noisier plans)
- `swing.leaf.max_center_dist_ratio`
- `swing.leaf.max_radial_angle_deg`
- `swing.leaf.max_tip_to_arc_ratio`

Important note:
Even if strict pairing fails, you still want candidates for snapping. If you consistently see arcs but no usable leaf, consider:
- allowing **arc-only** candidates in manual labeling (type `swing_arc`), or
- implementing a “promote arc+leaf from ROI” manual candidate creation flow.

##### Pattern 3: `component.topology` rejections (polyline arcs)
This indicates the arc polyline is part of a **branched component** (e.g. wall line touches the arc chain), producing >2 endpoints.

Knobs:
- ensure `swing.polyline_arc.allow_branches: true`
- increase `swing.polyline_arc.max_paths_per_component` if complex components need more exploration

---

## Recommended tuning strategy (safe and repeatable)

### Don’t “just lower everything”
Lowering thresholds globally will:
- increase false positives,
- flood the candidate pool,
- and make snapping noisier.

### Do this instead
1. **Broaden candidate existence**, not final doors:
   - enlarge arc radius bounds (`swing.arc.max_radius_px`) so large doors become candidates
   - loosen polyline arc extraction so non-Bezier arcs are found
2. Keep final door precision controlled via:
   - `output.min_confidence_after_reweight`
   - NMS (`output.nms_iou`)
3. Use the UI improvements:
   - **Selection matches** cycling
   - type filter (after user touches it)
   - double→component swing expansion

---

## Prompting template (copy/paste when asking for help)

Include:
- **file id**:
- **what you did** (Shift+drag? click? type filter?):
- the full block of console logs:
  - `snapCandidateForDrawPdf`
  - `draw_rect endDrag (pdfjs)` (must include `drawn_pdf_xyxy`)
  - `unmatched_debug_report raw` (if present)
- a screenshot of the door region

Example:

```text
File: f_...
Action: Edit Doors → Shift+drag around door arc+leaf; expected swing door
Type filter: All types (never touched) / or set to swing

Console:
[door_detector] snapCandidateForDrawPdf ...
[door_detector] draw_rect endDrag (pdfjs) ...
[door_detector] unmatched_debug_report raw ...

Screenshot: (attached)
```

