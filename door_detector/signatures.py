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
    reweighter_path = config.get("reweighter_path")
    if isinstance(reweighter_path, str) and reweighter_path:
        re_path = Path(reweighter_path)
        if re_path.exists():
            sig_content.extend(b"|")
            sig_content.extend(re_path.read_bytes())

    return hashlib.sha256(bytes(sig_content)).hexdigest()


# ---- Step 1 artifacts signature ----

# Bump this when Step 1 artifact generation semantics change (e.g. transforms).
STEP1_SIGNATURE_VERSION = 1

# Bump this when Step 1 logic changes but you don't want to bump the global schema.
# (Keeps the signature invalidating old artifacts after code changes.)
STEP1_LOGIC_VERSION = "transform_prerotate_v1"


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

