"""PDF.js-backed Streamlit viewer component.

This replaces the legacy iframe+sink pan/zoom viewer with a Streamlit custom
component so Streamlit reruns update props without reloading the PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit.components.v1 as components


_COMPONENT_NAME = "door_detector_pdfjs_viewer"


def _get_build_dir() -> Path:
    return Path(__file__).resolve().parent / "frontend" / "dist"


def _declare() -> Any:
    build_dir = _get_build_dir()
    # We intentionally do not provide a dev-server URL here. This repository
    # runs the component in bundled mode.
    return components.declare_component(_COMPONENT_NAME, path=str(build_dir))


_component_func = _declare()


def pdfjs_viewer(
    *,
    file_id: str,
    height: int,
    pdf_hash: str,
    pdf_data_b64: Optional[str],
    page_number: int = 1,
    overlay_doors: Optional[list[dict[str, Any]]] = None,
    candidate_pool: Optional[list[dict[str, Any]]] = None,
    selected_door_id: Optional[str] = None,
    focus_seq: int = 0,
    focus_request_seq: int = 0,
    auto_focus: bool = True,
    cycle_candidate_id: Optional[str] = None,
    edit_mode: bool = False,
    viewer_display_mode: str = "all",
    door_state: Optional[Dict[str, Any]] = None,
    manual_overlays: Optional[Dict[str, Any]] = None,
    proposal_overlays: Optional[Dict[str, Any]] = None,
    unmatched_debug_raw: Optional[str] = None,
    last_ack_event_id: Optional[str] = None,
    key: Optional[str] = None,
) -> Any:
    """Render the PDF.js viewer and return the latest emitted event (or None).

    The frontend emits events via `Streamlit.setComponentValue(...)`, e.g.:
    - {type: "door_click", event_id, door_id, ts}
    - {type: "draw_rect", event_id, bbox_pdf_xyxy, snapped_candidate_id?, ...}
    """

    return _component_func(
        fileId=str(file_id),
        height=int(height),
        pdfHash=str(pdf_hash or ""),
        pdfDataB64=pdf_data_b64,
        pageNumber=int(page_number),
        overlayDoors=list(overlay_doors or []),
        candidatePool=list(candidate_pool or []),
        selectedDoorId=str(selected_door_id) if selected_door_id not in (None, "") else "",
        focusSeq=int(focus_seq or 0),
        focusRequestSeq=int(focus_request_seq or 0),
        autoFocus=bool(auto_focus),
        cycleCandidateId=str(cycle_candidate_id or ""),
        editMode=bool(edit_mode),
        viewerDisplayMode=str(viewer_display_mode or "all"),
        doorState=dict(door_state or {}),
        manualOverlays=dict(manual_overlays or {}),
        proposalOverlays=dict(proposal_overlays or {}),
        unmatchedDebugRaw=str(unmatched_debug_raw or ""),
        lastAckEventId=str(last_ack_event_id or ""),
        default=None,
        key=key,
    )

