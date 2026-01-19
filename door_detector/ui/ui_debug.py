"""Lightweight UI debug helpers (breadcrumb trail + warn-once).

Keep this module dependency-light to avoid import cycles between Streamlit UI modules.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence


def ui_event_log(event: str, payload: Dict[str, Any]) -> None:
    """Print a single compact UI debug line.

    This intentionally uses `print()` (not `logging`) so the output is visible in the
    Streamlit server terminal without requiring logger configuration.

    Prefix is stable for grep/copy:
    - `[door_detector] ui_event ...`
    """
    try:
        obj = {"event": str(event or "")}
        if isinstance(payload, dict):
            obj.update(payload)
        print("[door_detector] ui_event", json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str))
    except Exception:
        return


def push_breadcrumb(
    fstate: Dict[str, Any],
    entry: Dict[str, Any],
    *,
    limit: int = 60,
) -> None:
    """Append a small debug record to `fstate` (best-effort, bounded)."""
    try:
        hist = fstate.get("_ui_breadcrumbs")
        if not isinstance(hist, list):
            hist = []
        e = dict(entry or {})
        e.setdefault("ts", time.time())
        hist.append(e)
        if isinstance(limit, int) and limit > 0 and len(hist) > limit:
            hist = hist[-limit:]
        fstate["_ui_breadcrumbs"] = hist
    except Exception:
        return


def tail_breadcrumbs(fstate: Dict[str, Any], *, n: int = 12) -> List[Dict[str, Any]]:
    """Return the last N breadcrumb entries (best-effort)."""
    try:
        hist = fstate.get("_ui_breadcrumbs")
        if not isinstance(hist, list) or not hist:
            return []
        nn = int(n) if isinstance(n, int) else 12
        if nn <= 0:
            return []
        out = hist[-nn:]
        return [x for x in out if isinstance(x, dict)]
    except Exception:
        return []


def warn_once(fstate: Dict[str, Any], key: str) -> bool:
    """Return True if caller should emit a warning for `key` (deduped per session/file)."""
    try:
        k = str(key or "")
        if not k:
            return True
        seen = fstate.get("_ui_warned_keys")
        if not isinstance(seen, set):
            seen = set()
        if k in seen:
            return False
        seen.add(k)
        fstate["_ui_warned_keys"] = seen
        return True
    except Exception:
        return True


def sample_ids(ids: Sequence[Any], *, limit: int = 12) -> List[str]:
    """Stable stringified sample of ids for logs."""
    out: List[str] = []
    try:
        for v in ids:
            if v is None:
                continue
            s = str(v)
            if not s:
                continue
            out.append(s)
            if len(out) >= int(limit):
                break
    except Exception:
        pass
    return out

