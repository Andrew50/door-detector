"""Signatures for detecting stale artifacts and configs.

This module centralizes signature logic used by:
- Step 1 artifacts (fast, stat-based signature for smart skipping in the UI)
- Step 2 / UI analysis config (content hash including referenced model bytes)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

# ---- Step 2 / analysis config signature ----


def compute_analysis_signature(config_path: Path) -> str:
    """Compute a SHA-256 signature for `config_path` and referenced model bytes.

    This is used to detect when an existing `doors.json` is out of date.
    """
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)

    sig_content = bytearray(config_bytes)

    def _base_dirs() -> list[Path]:
        out: list[Path] = []
        # Heuristic: configs typically live at `<repo>/configs/*.json` in this repo.
        try:
            if config_path.parent.name == "configs":
                out.append(config_path.parent.parent)
            else:
                out.append(config_path.parent)
        except Exception:
            pass
        # Fallback: infer repo root from package location (`door_detector/signatures.py` -> parents[1]).
        try:
            out.append(Path(__file__).resolve().parents[1])
        except Exception:
            pass
        # Deduplicate.
        seen: set[str] = set()
        uniq: list[Path] = []
        for p in out:
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            uniq.append(p)
        return uniq

    def _resolve_existing(path_str: str) -> Path | None:
        if not isinstance(path_str, str) or not path_str:
            return None
        raw = path_str.strip()
        if not raw:
            return None
        try:
            p = Path(raw)
        except Exception:
            return None
        try:
            if p.is_absolute():
                return p if p.exists() else None
        except Exception:
            return None
        # 1) As-is (relative to CWD).
        try:
            if p.exists():
                return p
        except Exception:
            pass
        # 2) Relative to inferred base dirs.
        for bd in _base_dirs():
            try:
                cand = bd / p
                if cand.exists():
                    return cand
            except Exception:
                continue
        return None

    added_any_reweighter = False

    # Preferred: per-type reweighters.
    reweighters = config.get("reweighters")
    if isinstance(reweighters, dict):
        for k in sorted(reweighters.keys(), key=lambda x: str(x)):
            v = reweighters.get(k)
            if not isinstance(v, str) or not v:
                continue
            re_path = _resolve_existing(v)
            if re_path is not None:
                sig_content.extend(b"|reweighter:")
                sig_content.extend(str(k).encode("utf-8"))
                sig_content.extend(b"|")
                sig_content.extend(re_path.read_bytes())
                added_any_reweighter = True

    # Backward compatibility: single reweighter path.
    legacy_path = config.get("reweighter_path")
    if isinstance(legacy_path, str) and legacy_path:
        re_path = _resolve_existing(legacy_path)
        if re_path is not None:
            sig_content.extend(b"|reweighter:legacy|")
            sig_content.extend(re_path.read_bytes())
            added_any_reweighter = True

    # If config doesn't reference any models, include default files (matches runtime auto-discovery).
    if not added_any_reweighter:
        for t in ("swing", "double", "pocket", "bifold"):
            re_path = _resolve_existing(f"models/reweighter_{t}_v1.json")
            if re_path is None:
                continue
            sig_content.extend(b"|reweighter:auto|")
            sig_content.extend(str(t).encode("utf-8"))
            sig_content.extend(b"|")
            sig_content.extend(re_path.read_bytes())
            added_any_reweighter = True
        legacy_auto = _resolve_existing("models/reweighter_v1.json")
        if legacy_auto is not None:
            sig_content.extend(b"|reweighter:auto|legacy|")
            sig_content.extend(legacy_auto.read_bytes())
            added_any_reweighter = True

    return hashlib.sha256(bytes(sig_content)).hexdigest()


# ---- Step 1 artifacts signature ----

# Bump this when Step 1 artifact generation semantics change (e.g. transforms).
STEP1_SIGNATURE_VERSION = 1

# Bump this when Step 1 logic changes but you don't want to bump the global schema.
# (Keeps the signature invalidating old artifacts after code changes.)
STEP1_LOGIC_VERSION = "transform_prerotate_v2"


def compute_step1_signature(*, pdf_path: Path, dpi: int, page_index: int) -> Dict[str, Any]:
    """Return a dict containing signature metadata and the signature string.

    This is intentionally fast: it uses `stat()` rather than hashing full file
    bytes, which makes it cheap to run on every UI interaction.
    """
    st = pdf_path.stat()

    payload: Dict[str, Any] = {
        "signature_version": STEP1_SIGNATURE_VERSION,
        "logic_version": STEP1_LOGIC_VERSION,
        "pdf": {
            "path": str(pdf_path),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        },
        "params": {"dpi": int(dpi), "page_index": int(page_index)},
    }

    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sig = hashlib.sha256(data).hexdigest()

    return {**payload, "signature": sig}

