"""Performance helpers (logging currently disabled).

Historically this module printed `[door_detector][perf] ...` JSON lines when enabled via env vars.
For current debugging work we intentionally keep *all perf logging disabled* to reduce noise.
The public API remains so existing call sites don't need to change.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s not in ("", "0", "false", "f", "no", "n", "off")


def enabled() -> bool:
    # Logging disabled (keep API for compatibility).
    return False


def _safe(v: Any) -> Any:
    # Keep logs compact and JSON-serializable.
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v[:50]]
    if isinstance(v, dict):
        out: Dict[str, Any] = {}
        for k, vv in list(v.items())[:50]:
            out[str(k)] = _safe(vv)
        return out
    return str(v)


def log(name: str, **fields: Any) -> None:
    # Perf logging disabled.
    return


@contextmanager
def span(name: str, **fields: Any) -> Iterator[None]:
    if not enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # No-op while logging disabled.
        _ = (name, dt_ms, fields)

