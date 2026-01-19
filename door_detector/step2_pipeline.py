"""Step 2 pipeline: Artifacts → Door detections."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from door_detector.signatures import compute_analysis_signature
from door_detector.doors.detect import detect_doors
from door_detector.doors.geometry import compute_iou
from door_detector.doors.overlay import create_door_overlay
from door_detector.pdf.affine import apply_affine_bbox_xyxy, fitz_bbox_to_pdfjs_bbox_xyxy, normalize_bbox_xyxy


def _coerce_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        if x is None:
            continue
        s = str(x)
        if s:
            out.append(s)
    return out


def _norm_bbox(rec: Dict[str, Any], key: str) -> Optional[List[float]]:
    bb = rec.get(key)
    if not isinstance(bb, list) or len(bb) != 4:
        return None
    try:
        return [float(v) for v in normalize_bbox_xyxy(bb)]
    except Exception:
        return None


def _bbox_area(bb: Optional[List[float]]) -> float:
    if not (isinstance(bb, list) and len(bb) == 4):
        return 0.0
    try:
        return max(0.0, float(bb[2] - bb[0])) * max(0.0, float(bb[3] - bb[1]))
    except Exception:
        return 0.0


def _extract_id_records(doors_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return a list of {id,type,bbox_pdf_xyxy?,bbox_xyxy?,legacy_ids?} records."""
    out: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("candidates", "doors"):
        seq = doors_data.get(key) or []
        if not isinstance(seq, list):
            continue
        for c in seq:
            if not isinstance(c, dict):
                continue
            cid = c.get("id")
            if cid is None:
                continue
            cid_s = str(cid)
            if not cid_s or cid_s in seen:
                continue
            seen.add(cid_s)
            out.append(
                {
                    "id": cid_s,
                    "type": str(c.get("type") or "").strip(),
                    "bbox_pdf_xyxy": _norm_bbox(c, "bbox_pdf_xyxy"),
                    "bbox_xyxy": _norm_bbox(c, "bbox_xyxy"),
                    "legacy_ids": _coerce_str_list(c.get("legacy_ids")),
                }
            )
    return out


def _attach_legacy_ids_from_previous_doors_json(
    *,
    new_candidates: List[Dict[str, Any]],
    prev_doors_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Best-effort remap across reanalysis by attaching old ids as `legacy_ids`.

    Why this exists:
    - UI labels persist by candidate id strings.
    - Step 1 changes (transform/primitives) can shift pixel geometry, producing new ids.
    - We preserve continuity by matching old→new candidates by PDF-space bbox overlap
      and adding the old ids into `legacy_ids`, which the UI already uses to remap
      `labels.json` on load.
    """
    if not isinstance(prev_doors_data, dict):
        return {"matched": 0, "unmatched": 0, "reason": "prev_doors_not_dict"}

    prev_recs = _extract_id_records(prev_doors_data)
    if not prev_recs or not isinstance(new_candidates, list) or not new_candidates:
        return {"matched": 0, "unmatched": int(len(prev_recs)), "reason": "empty_prev_or_new"}

    # Build normalized records for the new candidate list (by current id).
    new_recs: list[Dict[str, Any]] = []
    for c in new_candidates:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if cid is None:
            continue
        new_recs.append(
            {
                "id": str(cid),
                "type": str(c.get("type") or "").strip(),
                "bbox_pdf_xyxy": _norm_bbox(c, "bbox_pdf_xyxy"),
                "bbox_xyxy": _norm_bbox(c, "bbox_xyxy"),
            }
        )

    if not new_recs:
        return {"matched": 0, "unmatched": int(len(prev_recs)), "reason": "no_new_ids"}

    # Candidate pools: prefer PDF-space bboxes (stable across DPI/crop) if available.
    new_by_type: dict[str, list[Dict[str, Any]]] = {}
    for r in new_recs:
        new_by_type.setdefault(r.get("type") or "", []).append(r)

    # Greedy one-to-one matching to avoid collapsing multiple old ids onto one new id.
    used_new: set[str] = set()
    prev_recs_sorted = sorted(
        prev_recs,
        key=lambda r: (
            # prefer mapping larger boxes first
            _bbox_area(r.get("bbox_pdf_xyxy")) or _bbox_area(r.get("bbox_xyxy")),
            str(r.get("type") or ""),
            str(r.get("id") or ""),
        ),
        reverse=True,
    )

    def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        # Prefer PDF-space IoU when both are present; otherwise pixel-space.
        ap = a.get("bbox_pdf_xyxy")
        bp = b.get("bbox_pdf_xyxy")
        if isinstance(ap, list) and isinstance(bp, list):
            return float(compute_iou(ap, bp))
        ax = a.get("bbox_xyxy")
        bx = b.get("bbox_xyxy")
        if isinstance(ax, list) and isinstance(bx, list):
            return float(compute_iou(ax, bx))
        return 0.0

    # Thresholds tuned for stability; PDF-space should be very stable across Step 1 changes.
    MIN_IOU_SAME_TYPE = 0.55
    MIN_IOU_CROSS_TYPE = 0.70

    old_to_new: dict[str, str] = {}

    def _find_best(old: Dict[str, Any], pool: list[Dict[str, Any]]) -> Tuple[float, Optional[Dict[str, Any]]]:
        best_i = 0.0
        best_r: Optional[Dict[str, Any]] = None
        for nr in pool:
            nid = str(nr.get("id") or "")
            if not nid or nid in used_new:
                continue
            i = _iou(old, nr)
            if i > best_i:
                best_i = i
                best_r = nr
        return best_i, best_r

    # Pass 1: same-type matching.
    for old in prev_recs_sorted:
        ot = str(old.get("type") or "")
        pool = new_by_type.get(ot) or []
        if not pool:
            continue
        best_i, best_r = _find_best(old, pool)
        if best_r is not None and best_i >= MIN_IOU_SAME_TYPE:
            nid = str(best_r.get("id") or "")
            oid = str(old.get("id") or "")
            if oid and nid:
                old_to_new[oid] = nid
                used_new.add(nid)

    # Pass 2: cross-type matching for any remaining old ids (more conservative).
    all_pool = new_recs
    for old in prev_recs_sorted:
        oid = str(old.get("id") or "")
        if not oid or oid in old_to_new:
            continue
        best_i, best_r = _find_best(old, all_pool)
        if best_r is not None and best_i >= MIN_IOU_CROSS_TYPE:
            nid = str(best_r.get("id") or "")
            if nid:
                old_to_new[oid] = nid
                used_new.add(nid)

    if not old_to_new:
        return {"matched": 0, "unmatched": int(len(prev_recs)), "reason": "no_matches"}

    # Attach legacy ids to the in-memory new candidate dicts.
    new_by_id: dict[str, Dict[str, Any]] = {}
    for c in new_candidates:
        if isinstance(c, dict) and c.get("id") is not None:
            new_by_id[str(c.get("id"))] = c

    prev_by_id = {str(r.get("id")): r for r in prev_recs if r.get("id") is not None}
    matched = 0
    for old_id, new_id in old_to_new.items():
        tgt = new_by_id.get(new_id)
        if not isinstance(tgt, dict):
            continue
        prev_rec = prev_by_id.get(str(old_id)) or {}
        add = [str(old_id)] + _coerce_str_list(prev_rec.get("legacy_ids"))
        cur = _coerce_str_list(tgt.get("legacy_ids"))
        merged = list(dict.fromkeys(cur + add))  # stable unique
        tgt["legacy_ids"] = merged
        matched += 1

    return {"matched": int(matched), "unmatched": int(max(0, len(prev_recs) - matched))}


def run_step2(
    artifacts_dir: Path,
    config_path: Path,
    output_dir: Path | None = None
) -> None:
    """Run door detection on a Step 1 artifacts directory."""
    
    if output_dir is None:
        output_dir = artifacts_dir

    # 1. Load config and compute signature
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_bytes())
    # Best-effort base dir for resolving `models/...` references when the process
    # is launched from a different working directory than the repo root.
    try:
        base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        config["_door_detector_base_dir"] = str(base_dir)
    except Exception:
        pass
    analysis_signature = compute_analysis_signature(config_path)

    # 2. Load artifacts
    primitives_path = artifacts_dir / "primitives.json"
    meta_path = artifacts_dir / "meta.json"
    image_path = artifacts_dir / "page.png"

    if not all(p.exists() for p in [primitives_path, meta_path, image_path]):
        raise FileNotFoundError(f"Missing required artifacts in {artifacts_dir}")

    with open(primitives_path) as f:
        primitives = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    
    image = Image.open(image_path)
    transform = None
    try:
        transform_path = artifacts_dir / "transform.json"
        if transform_path.exists():
            with open(transform_path) as f:
                transform = json.load(f)
    except Exception:
        transform = None

    # 3. Detect doors
    import time
    start_time = time.time()
    det = detect_doors(primitives, meta, config)
    doors = list(det.get("doors", []) if isinstance(det, dict) else (det or []))
    candidates = list(det.get("candidates", []) if isinstance(det, dict) else [])
    dedupe_debug = det.get("dedupe_debug") if isinstance(det, dict) else None
    detect_ms = (time.time() - start_time) * 1000

    # 3b. Add PDF-space bboxes for UI (PDF.js) and future-proofed downstream use.
    # NOTE: `pix_to_pdf_affine` maps pixel → PyMuPDF (fitz) coordinates (Y-down).
    # PDF.js expects PDF-spec coordinates (Y-up), so we flip the Y axis using the cropbox.
    pix_to_pdf_affine = None
    cropbox = None
    try:
        if isinstance(transform, dict):
            m = transform.get("pix_to_pdf_affine")
            if isinstance(m, list) and len(m) == 6:
                pix_to_pdf_affine = [float(v) for v in m]
            cb = transform.get("cropbox")
            if isinstance(cb, dict):
                cropbox = cb
    except Exception:
        pix_to_pdf_affine = None

    if pix_to_pdf_affine is not None and isinstance(cropbox, dict):
        for seq in (doors, candidates):
            for d in seq:
                try:
                    bb = d.get("bbox_xyxy")
                    if not isinstance(bb, list) or len(bb) != 4:
                        continue
                    bb = normalize_bbox_xyxy(bb)
                    bbox_fitz = apply_affine_bbox_xyxy(pix_to_pdf_affine, bb)
                    d["bbox_pdf_xyxy"] = fitz_bbox_to_pdfjs_bbox_xyxy(bbox_fitz, cropbox=cropbox)
                except Exception:
                    continue

    # 3c. If `doors.json` already exists in the output dir, attach old ids as `legacy_ids`
    # so `labels.json` can be remapped seamlessly after reanalysis (important for Step 1 changes).
    prev_doors_data: Dict[str, Any] = {}
    try:
        prev_path = Path(output_dir) / "doors.json"
        if prev_path.exists():
            prev_doors_data = json.loads(prev_path.read_bytes())
    except Exception:
        prev_doors_data = {}

    remap_summary: Dict[str, Any] = {}
    try:
        if prev_doors_data:
            remap_summary = _attach_legacy_ids_from_previous_doors_json(
                new_candidates=candidates,
                prev_doors_data=prev_doors_data,
            )
    except Exception:
        remap_summary = {}

    # 4. Save doors.json
    doors_data = {
        "schema_version": 2,
        "page_id": meta["id"],
        "source_artifacts_dir": str(artifacts_dir),
        "config_path": str(config_path),
        "analysis_signature": analysis_signature,
        "mode": meta["mode"],
        "detect_ms": detect_ms,
        "doors": doors,
        "candidates": candidates,
    }
    if isinstance(dedupe_debug, dict) and dedupe_debug:
        doors_data["dedupe_debug"] = dedupe_debug
    if remap_summary:
        doors_data["id_remap_summary"] = remap_summary
    
    with open(output_dir / "doors.json", "w") as f:
        json.dump(doors_data, f, indent=2)

    # 5. Create overlay
    create_door_overlay(image, doors, output_dir / "doors_overlay.png")

    print(f"Successfully processed {artifacts_dir}")
    print(f"  Detections: {len(doors)}")
    print(f"  Time: {detect_ms:.1f}ms")
    print(f"  Output: {output_dir / 'doors.json'}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Step 2: Detect doors from normalized artifacts"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Path to Step 1 artifacts directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/door_rules.json"),
        help="Path to door detection rules config",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output directory (defaults to artifacts dir)",
    )

    args = parser.parse_args()

    if not args.artifacts.exists():
        print(f"Error: Artifacts directory not found: {args.artifacts}", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        run_step2(
            artifacts_dir=args.artifacts,
            config_path=args.config,
            output_dir=args.out
        )
    except Exception as e:
        print(f"Error in Step 2: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


