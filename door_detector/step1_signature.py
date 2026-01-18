"""Compute a fast signature for Step 1 artifacts.

Used to decide whether Step 1 needs to be re-run for a given `source.pdf` and
Step 1 parameters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

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

    payload = {
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

