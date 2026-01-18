"""Compute a stable signature for an analysis configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


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

