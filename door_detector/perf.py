"""Lightweight opt-in performance logging.

Enable by setting environment variable:
- DOOR_DETECTOR_PROFILE=1   (or true/yes/on)

This module is intentionally dependency-free and safe to import anywhere.
"""

from __future__ import annotations

import json
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
    return _truthy_env("DOOR_DETECTOR_PROFILE") or _truthy_env("DOOR_DETECTOR_PERF")


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
    if not enabled():
        return
    payload = {"name": str(name)}
    for k, v in fields.items():
        payload[str(k)] = _safe(v)
    try:
        print("[door_detector][perf]", json.dumps(payload, separators=(",", ":")))
    except Exception:
        # Never allow perf logging to break the app.
        try:
            print("[door_detector][perf]", str(payload))
        except Exception:
            pass


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
        log(name, ms=round(float(dt_ms), 3), **fields)

