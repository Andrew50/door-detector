import hashlib
import html
import json
import logging
import math
import os
import shutil
import time
import base64
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from door_detector.analysis_signature import compute_analysis_signature
from door_detector.door_features import (
    compute_iou,
    dist_point_to_point,
    fit_circle,
    get_arc_angle_span,
    get_bbox,
    sample_bezier,
)

from door_detector.library import Library
from door_detector.step1_pipeline import process_pdf
from door_detector.step1_signature import compute_step1_signature
from door_detector.step2_pipeline import run_step2
from door_detector.reweight_fit import fit_reweighter

# Log to Streamlit's server console for debugging.
logger = logging.getLogger("door_detector.review_app")

# Increase PIL pixel limit
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Door Detector: Door Detection & Review", layout="wide", initial_sidebar_state="expanded")

# --- UI Styling ---
st.markdown("""
<style>
    /* Hide Streamlit chrome (cosmetic only; not a security boundary).
       We keep the sidebar always visible and remove the top toolbar entirely. */
    /* Streamlit 1.53 reserves space for the header via CSS variables; force it to 0. */
    :root {
        --header-height: 0px !important;
        --st-header-height: 0px !important;
    }

    header,
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        /* IMPORTANT: don't `display:none` the toolbar/header; Streamlit may mount
           the sidebar "open" control there when the sidebar is collapsed, and
           we need it present so JS can force-open on page load. */
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        overflow: visible !important;
        /* Actually hide it visually (incl. deploy button), but keep it in the DOM so
           the sidebar "open" control can still exist and be clicked via JS. */
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important; /* avoid any accidental clicks */
    }
    [data-testid="stStatusWidget"] { display: none !important; }
    #MainMenu { display: none !important; }
    [data-testid="stMainMenu"] { display: none !important; }
    footer { display: none !important; }
    [data-testid="stDeployButton"] { display: none !important; }

    /* Prevent collapsing: hide any sidebar close/collapse buttons. */
    [data-testid="stSidebar"] button[aria-label="Close sidebar"],
    [data-testid="stSidebar"] button[aria-label="Collapse sidebar"],
    [data-testid="stSidebar"] button[title="Close sidebar"],
    [data-testid="stSidebar"] button[title="Collapse sidebar"],
    [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"],
    [data-testid="stSidebar"] button[kind="headerNoPadding"],
    [data-testid="stSidebar"] button[data-testid*="headerNoPadding"] {
        display: none !important;
    }

    /* Hide the collapsed-control toggle UI (we auto-open on load if needed).
       Use visibility/opacity instead of display:none so it stays in the DOM. */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[aria-label="Open sidebar"] {
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Remove Streamlit's default huge bottom spacing in main area */
    html body section.stMain {
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    html body section.stMain > div {
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    html body section.stMain .block-container {
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
    }
    /* Some Streamlit versions include an empty bottom container */
    [data-testid="stBottomBlockContainer"] { display: none !important; }

    /* Pull main content to the top (align with sidebar header) */
    [data-testid="stAppViewContainer"] {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    [data-testid="stAppViewContainer"] .main {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    [data-testid="stAppViewContainer"] .main > div:first-child {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 0rem !important;
        margin-top: 0rem !important;
        max-width: 100% !important;
    }

    /* Streamlit sometimes targets this container directly; keep it tight. */
    [data-testid="stMainBlockContainer"] {
        padding-top: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 0rem !important;
        margin-top: 0rem !important;
    }
    /* Streamlit 1.53 adds extra top padding on the first inner wrapper inside the
       main block container; remove it so the PDF title sits closer to the top. */
    [data-testid="stMainBlockContainer"] > div,
    [data-testid="stMainBlockContainer"] > div:first-child {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    /* Streamlit 1.53 uses class selectors here (not always data-testid). */
    section.stMain > div.stMainBlockContainer,
    div.stMainBlockContainer {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    section.stMain > div.stMainBlockContainer > div,
    div.stMainBlockContainer > div {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        justify-content: flex-start !important;
        align-content: flex-start !important;
    }
    /* Higher-specificity overrides for 1.53's `stMainBlockContainer.block-container.*` rules
       (important-vs-important: specificity matters). */
    section.stMain > div.stMainBlockContainer.block-container,
    div.stMainBlockContainer.block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    section.stMain > div.stMainBlockContainer.block-container > div,
    div.stMainBlockContainer.block-container > div {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        /* This wrapper is a flex column in Streamlit 1.53; reduce its default `gap`
           (often 1rem) which can look like "space above" the first visible widget. */
        gap: 0.5rem !important;
        row-gap: 0.5rem !important;
    }
    /* Some Streamlit builds nest another wrapper div that carries the top padding. */
    section.stMain > div.stMainBlockContainer.block-container > div > div,
    div.stMainBlockContainer.block-container > div > div {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    [data-testid="stAppViewContainer"] .main h1, 
    [data-testid="stAppViewContainer"] .main h3 {
        margin-top: 0 !important;
        padding-top: 0rem !important;
    }

    /* Sidebar: remove the reserved top padding that Streamlit adds for the header. */
    [data-testid="stSidebar"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Streamlit 1.53: this header block sits above the sidebar content and
       contains the collapse button + logo spacer. Remove it so "Library" can
       sit at the top. */
    [data-testid="stSidebarHeader"],
    [data-testid="stLogoSpacer"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebarHeader"] button[data-testid="stBaseButton-headerNoPadding"] {
        display: none !important;
    }

    /* Styling for the file list rows */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        padding: 2px 4px !important;
        transition: background-color 0.2s;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }

    /* Library file list: plain text buttons, single-line, clipped (no wrap) */
    [data-testid="stSidebar"] button[id^="sel_"] {
        background: none !important;
        border: none !important;
        padding: 2px 0px !important;
        margin: 0 !important;
        color: rgba(255, 255, 255, 0.8) !important;
        text-align: left !important;
        width: 100% !important;
        display: block !important;
        box-shadow: none !important;
        min-height: 0 !important;
        line-height: 1.2 !important;
        outline: none !important;
        overflow: hidden !important;
        white-space: nowrap !important;
    }

    [data-testid="stSidebar"] button[id^="sel_"]:hover {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Streamlit renders button labels via a markdown container; force it to never wrap */
    [data-testid="stSidebar"] button[id^="sel_"] [data-testid="stMarkdownContainer"] {
        overflow: hidden !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }

    [data-testid="stSidebar"] button[id^="sel_"] div {
        overflow: hidden !important;
        min-width: 0 !important;
        max-width: 100% !important;
        flex-wrap: nowrap !important;
    }

    [data-testid="stSidebar"] button[id^="sel_"] * {
        white-space: nowrap !important;
    }

    [data-testid="stSidebar"] button[id^="sel_"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] button[id^="sel_"] [data-testid="stMarkdownContainer"] span {
        overflow: hidden !important;
        text-overflow: clip !important;
        display: block !important;
        width: 100% !important;
        font-size: 13px !important;
        margin: 0 !important;
    }

    /* Selected library item: emphasize via text only (no box) */
    [data-testid="stSidebar"] button[id^="sel_"][data-testid="stBaseButton-primary"] {
        color: white !important;
        font-weight: 600 !important;
    }

    /* Streamlit/BaseUI buttons often look like: #bui3__anchor > button
       Force sidebar button labels to NEVER wrap (clip instead). */
    [data-testid="stSidebar"] div[id$="__anchor"] > button {
        max-width: 100% !important;
        width: 100% !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] div[id$="__anchor"] > button > div,
    [data-testid="stSidebar"] div[id$="__anchor"] > button > div > div {
        max-width: 100% !important;
        min-width: 0 !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] div[id$="__anchor"] > button [data-testid="stMarkdownContainer"] {
        max-width: 100% !important;
        min-width: 0 !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] div[id$="__anchor"] > button [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[id$="__anchor"] > button [data-testid="stMarkdownContainer"] span {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: clip !important;
        max-width: 100% !important;
    }

    /* File Uploader styling to look like a simple "Add" button */
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
        border: none !important;
        background: none !important;
    }
    [data-testid="stFileUploader"] section > div {
        display: none !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
        padding: 0px 12px !important;
        border-radius: 4px !important;
        font-size: 14px !important;
        height: 38px !important;
        width: 100% !important;
    }
    /* Replace Streamlit's default "Browse files" label with "Upload" */
    [data-testid="stFileUploader"] button > div > p {
        font-size: 0 !important;
        line-height: 1 !important;
        margin: 0 !important;
    }
    [data-testid="stFileUploader"] button > div > p::before {
        content: "Upload";
        font-size: 14px;
    }
    [data-testid="stFileUploader"] small {
        display: none !important;
    }

    /* Keep Search/Clear buttons looking like buttons */
    [data-testid="stSidebar"] .stButton button#open_search_btn,
    [data-testid="stSidebar"] .stButton button#close_search_btn {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 0px 12px !important;
        text-align: center !important;
        border-radius: 4px !important;
        margin-bottom: 0px !important;
        height: 38px !important;
    }

    /* File list rows: remove column gap and ensure background covers everything */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] [data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        border: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.1rem !important;
    }
    
    [data-testid="stSidebar"] h1 {
        padding-top: 0 !important;
        margin-top: 28px !important;
        margin-bottom: 24px !important; /* more space before Search/Upload row */
    }

    /* Make the X button align better with the search input */
    [data-testid="stSidebar"] div[data-testid="column"] .stButton button#close_search_btn {
        margin-top: 0px !important;
        height: 38px !important;
        width: 100% !important;
    }

    /* Make the Search button match Upload sizing */
    [data-testid="stSidebar"] div[data-testid="column"] .stButton button#open_search_btn {
        margin-top: 0px !important;
        height: 38px !important;
        width: 100% !important;
    }

    /* Main viewer: keep PDF title area a fixed two lines tall */
    .door_detector-pdf-title {
        --door_detector-title-font-size: 1.35rem;
        --door_detector-title-line-height: 1.55rem;

        /* Match existing heading spacing, but keep total height stable */
        padding-top: 0rem;
        margin: 0 0 0.75rem 0;

        /* Always reserve exactly two lines */
        height: calc(2 * var(--door_detector-title-line-height));
        overflow: hidden;
    }

    .door_detector-pdf-title > h3 {
        font-size: var(--door_detector-title-font-size) !important;
        line-height: var(--door_detector-title-line-height) !important;

        margin: 0 !important;
        padding: 0 !important; /* override global .main h3 padding-top */

        /* Clamp to two lines; container keeps the reserved space either way */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        word-break: break-word;
    }

    /* Hide internal "sink" widgets (used for box click selection + JS sync) */
    div[data-testid="stTextInput"]:has(input[aria-label^="door_click_sink_"]) {
        /* IMPORTANT: don't use display:none; Streamlit/React may ignore programmatic
           input events on elements that are not rendered. Keep it off-screen. */
        position: fixed !important;
        left: -10000px !important;
        top: -10000px !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        opacity: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: none !important;
    }
    div[data-testid="stTextInput"] input[aria-label^="door_click_sink_"] {
        position: fixed !important;
        left: -10000px !important;
        top: -10000px !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        opacity: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: none !important;
    }
    div[data-testid="stTextInput"]:has(input[aria-label^="selected_door_sink_"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"] input[aria-label^="selected_door_sink_"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"]:has(input[aria-label^="focus_seq_sink_"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"] input[aria-label^="focus_seq_sink_"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"]:has(input[aria-label^="auto_focus_sink_"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"] input[aria-label^="auto_focus_sink_"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"]:has(input[aria-label^="edit_mode_sink_"]),
    div[data-testid="stTextInput"]:has(input[aria-label^="draw_event_sink_"]),
    div[data-testid="stTextInput"]:has(input[aria-label^="manual_overlay_sink_"]),
    div[data-testid="stTextInput"]:has(input[aria-label^="door_state_sink_"]),
    div[data-testid="stTextInput"]:has(input[aria-label^="viewer_display_sink_"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"] input[aria-label^="edit_mode_sink_"],
    div[data-testid="stTextInput"] input[aria-label^="draw_event_sink_"],
    div[data-testid="stTextInput"] input[aria-label^="manual_overlay_sink_"],
    div[data-testid="stTextInput"] input[aria-label^="door_state_sink_"],
    div[data-testid="stTextInput"] input[aria-label^="viewer_display_sink_"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Door navigation: make the index input look obviously editable */
    div[data-testid="stNumberInput"]:has(input[aria-label^="door_jump_idx_"]) input {
        text-align: center !important;
        font-weight: 650 !important;
        background: rgba(17, 25, 40, 0.70) !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 10px !important;
        height: 38px !important;
        padding: 0px 10px !important;
    }

    /* Prevent Delete/Confirm/Cancel labels from wrapping (even in narrow columns)
       NOTE: keys must be valid HTML ids; we generate safe hashed keys. */
    button[id^="delete_btn_"] div[data-testid="stMarkdownContainer"] p,
    button[id^="delete_confirm_btn_"] div[data-testid="stMarkdownContainer"] p,
    button[id^="delete_cancel_btn_"] div[data-testid="stMarkdownContainer"] p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Selected door details */
    .door_detector-door-meta {
        display: flex;
        align-items: baseline;
        gap: 16px; /* spacing between items (no pipe separators) */
        padding: 6px 2px;
    }
    .door_detector-door-meta-item {
        display: inline-flex;
        align-items: baseline;
        gap: 6px;
    }
    .door_detector-door-meta-label {
        font-size: 13px;
        font-weight: 650;
        opacity: 0.8;
    }
    .door_detector-door-meta-type {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 0.2px;
        text-transform: capitalize;
    }
    .door_detector-door-meta-confidence {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 0.2px;
    }

    /* Viewer loading state (replaces the PDF viewer during analysis/re-analysis) */
    .door_detector-viewer-loading {
        width: 100%;
        background: #0e1117;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    .door_detector-viewer-loading-inner {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 14px;
        padding: 24px 18px;
        text-align: center;
        max-width: 520px;
    }
    .door_detector-spinner {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        border: 7px solid rgba(255, 255, 255, 0.12);
        border-top-color: rgba(255, 75, 75, 0.98);
        animation: door_detectorSpin 0.95s linear infinite;
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.35);
    }
    .door_detector-viewer-loading-title {
        font-size: 18px;
        font-weight: 750;
        letter-spacing: 0.2px;
        color: rgba(255, 255, 255, 0.92);
    }
    .door_detector-viewer-loading-sub {
        font-size: 13px;
        line-height: 1.4;
        color: rgba(255, 255, 255, 0.70);
    }
    @keyframes door_detectorSpin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def _sidebar_autopen_component_html() -> str:
    """JS that force-opens Streamlit's sidebar if a prior session left it collapsed.

    NOTE: This must run inside a Streamlit component iframe so it can access `window.parent`.
    """
    return """
<script>
(function () {
  try {
    const doc = window.parent && window.parent.document ? window.parent.document : document;
    function getSidebar() {
      try { return doc.querySelector('[data-testid="stSidebar"]'); } catch (_) { return null; }
    }
    function isCollapsed() {
      const sb = getSidebar();
      if (!sb) return false;
      const v = sb.getAttribute("aria-expanded");
      return v === "false";
    }
    function findExpandControl() {
      const selectors = [
        '[data-testid="collapsedControl"] button',
        '[data-testid="collapsedControl"]',
        '[data-testid="stSidebarCollapsedControl"] button',
        '[data-testid="stSidebarCollapsedControl"]',
        'button[data-testid="stBaseButton-headerNoPadding"]',
        'button[kind="headerNoPadding"]',
        'button[aria-label="Open sidebar"]',
      ];
      for (const sel of selectors) {
        try {
          const el = doc.querySelector(sel);
          if (el) return el;
        } catch (_) {}
      }
      return null;
    }
    let tries = 0;
    const maxTries = 40; // ~10s @ 250ms
    const timer = setInterval(() => {
      tries += 1;
      if (tries > maxTries) { try { clearInterval(timer); } catch (_) {} return; }
      if (!isCollapsed()) { try { clearInterval(timer); } catch (_) {} return; }
      const btn = findExpandControl();
      if (btn && btn.click) {
        try { btn.click(); } catch (_) {}
      }
    }, 250);
  } catch (_) {}
})();
</script>
"""

# --- Initialize Library ---
if "library" not in st.session_state:
    st.session_state.library = Library(Path("artifacts"))
    # NOTE: We no longer auto-import every existing `artifacts/**/meta.json` on startup,
    # because that can flood the Library with hundreds of historical runs.

if "search_visible" not in st.session_state:
    st.session_state.search_visible = False

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# File uploader widgets retain their value across reruns. If we `st.rerun()` after
# handling an upload without resetting the widget, we'll keep re-processing the
# same uploaded file and rerunning before the library list renders.
if "upload_widget_seq" not in st.session_state:
    st.session_state.upload_widget_seq = 0

# Pipeline run state (used to show an "Analyzing..." indicator during synchronous work)
if "door_detector_pipeline_task" not in st.session_state:
    # { file_id, file_dir, config_path, label, _started }
    st.session_state.door_detector_pipeline_task = None

# Debug logging toggle (server console). Used by `_debug_log(...)`.
if "debug_perf" not in st.session_state:
    st.session_state.debug_perf = False

VIEWER_TARGET_WIDTH_PX = 1200
VIEWER_ASPECT_RATIO_HW = 0.75  # height/width

lib = st.session_state.library

def _get_viewer_height_px() -> int:
    viewer_width_hint = int(VIEWER_TARGET_WIDTH_PX)
    viewer_width_hint = max(600, min(2000, viewer_width_hint))
    aspect = float(VIEWER_ASPECT_RATIO_HW)
    aspect = max(0.35, min(1.25, aspect))
    viewer_height = int(round(viewer_width_hint * aspect))
    return max(450, min(1400, viewer_height))

def _render_viewer_loading(*, height_px: int, title: str, subtitle: str) -> None:
    safe_title = html.escape(str(title or "Analyzing…"))
    safe_sub = html.escape(str(subtitle or "Running analysis…"))
    st.markdown(
        f"""
<div class="door_detector-viewer-loading" style="height: {int(height_px)}px;">
  <div class="door_detector-viewer-loading-inner" role="status" aria-live="polite">
    <div class="door_detector-spinner" aria-hidden="true"></div>
    <div class="door_detector-viewer-loading-title">{safe_title}</div>
    <div class="door_detector-viewer-loading-sub">{safe_sub}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

def _queue_pipeline_run(file_id: str, file_dir_str: str, config_path: str, label: str) -> None:
    # Queue work; it will execute on the next run so the viewer can paint a loader first.
    st.session_state.door_detector_pipeline_task = {
        "file_id": str(file_id),
        "file_dir": str(file_dir_str),
        "config_path": str(config_path),
        "label": str(label),
        "_started": False,
    }

def _debug_log(msg: str, *args: Any) -> None:
    """Debug logging gated by the sidebar perf checkbox."""
    try:
        if st.session_state.get("debug_perf"):
            logger.info(msg, *args)
    except Exception:
        return


def _coerce_bool(v: Any, *, default: bool = False) -> bool:
    """Best-effort boolean coercion for values coming from Streamlit/session_state."""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            return bool(int(v) != 0)
        except Exception:
            return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off", ""):
            return False
        return default
    return default

def _delete_library_item_and_reset_ui(file_id: str) -> None:
    lib.delete_item(file_id)
    # Clear selection + per-file UI state to avoid dangling widget keys.
    try:
        st.session_state.files.pop(file_id, None)
    except Exception:
        pass
    for k in [
        f"auto_focus_{file_id}",
        f"auto_focus_sink_{file_id}",
        f"jump_{file_id}",
        f"door_click_sink_{file_id}",
        f"selected_door_sink_{file_id}",
        f"focus_seq_sink_{file_id}",
        f"edit_mode_sink_{file_id}",
        f"draw_event_sink_{file_id}",
        f"manual_overlay_sink_{file_id}",
        f"door_state_sink_{file_id}",
        f"viewer_display_sink_{file_id}",
        f"confirm_delete_{file_id}",
    ]:
        try:
            st.session_state.pop(k, None)
        except Exception:
            pass
    st.session_state.selected_file_id = None

    st.cache_data.clear()
    try:
        st.cache_resource.clear()
    except Exception:
        pass

    # Labels schema changed (v2-only); avoid any lingering cached file artifacts.
    # (load_file_artifacts is no longer cached, but keep this for safety.)
    try:
        st.session_state.pop("_last_loaded_labels_schema_version", None)
    except Exception:
        pass

def _clear_library_and_reset_ui() -> None:
    lib.clear()
    try:
        st.session_state.files = {}
    except Exception:
        pass
    st.session_state.selected_file_id = None

    # Reset uploader widget so the UI doesn't immediately re-add a previously-selected file.
    try:
        st.session_state.upload_widget_seq = int(st.session_state.get("upload_widget_seq") or 0) + 1
    except Exception:
        st.session_state.upload_widget_seq = 1

    st.cache_data.clear()
    try:
        st.cache_resource.clear()
    except Exception:
        pass

    try:
        st.session_state.pop("_last_loaded_labels_schema_version", None)
    except Exception:
        pass

# --- Session State Helpers ---
LABELS_SCHEMA_VERSION = 2
_LEGACY_LABEL_KEYS = {"accepted_ids", "rejected_ids", "added_boxes", "notes"}
_LABELS_V2_REQUIRED_KEYS = {
    "schema_version",
    "reviewed_at",
    "confirmed_ids",
    "deleted_ids",
    "manual_additions",
    "unmatched_manual_boxes",
}


def _labels_v2_default() -> Dict[str, Any]:
    # reviewed_at is intentionally null until the first save.
    return {
        "schema_version": LABELS_SCHEMA_VERSION,
        "reviewed_at": None,
        "confirmed_ids": [],
        "deleted_ids": [],
        "manual_additions": [],
        "unmatched_manual_boxes": [],
    }


def _validate_labels_v2_or_raise(labels_data: Dict[str, Any], *, labels_path: Path) -> None:
    """Validate schema v2 labels.json. Raise with a clear migration message on failure."""
    if not isinstance(labels_data, dict):
        raise ValueError(f"Invalid labels.json (expected object): {labels_path}")

    schema_version = labels_data.get("schema_version", None)
    if schema_version != LABELS_SCHEMA_VERSION:
        raise ValueError(
            "\n".join(
                [
                    f"Unsupported labels.json schema in {labels_path}.",
                    f"Expected schema_version={LABELS_SCHEMA_VERSION}, got {schema_version!r}.",
                    "",
                    "This UI no longer supports legacy label schemas.",
                    "Please delete this labels.json (or migrate it offline) and re-review the file.",
                ]
            )
        )

    legacy_present = sorted(k for k in _LEGACY_LABEL_KEYS if k in labels_data)
    if legacy_present:
        raise ValueError(
            "\n".join(
                [
                    f"labels.json in {labels_path} contains deprecated fields: {legacy_present}",
                    "Please delete this labels.json (or migrate it offline) and re-review the file.",
                ]
            )
        )

    missing = sorted(k for k in _LABELS_V2_REQUIRED_KEYS if k not in labels_data)
    if missing:
        raise ValueError(f"labels.json in {labels_path} is missing required keys: {missing}")

    # Lightweight type checks (don’t coerce silently; fail fast).
    for lk in ["confirmed_ids", "deleted_ids", "manual_additions", "unmatched_manual_boxes"]:
        if not isinstance(labels_data.get(lk), list):
            raise ValueError(f"labels.json field {lk!r} must be a list: {labels_path}")


def _viewer_display_mode_to_sink_value(mode: str) -> str:
    """Map UI selection to a compact string the iframe JS can poll."""
    if mode == "Highlight Selected":
        return "selected"
    if mode == "Off":
        return "off"
    return "all"


def _snapshot_label_state(src: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "confirmed_ids": set(src.get("confirmed_ids", set())),
        "deleted_ids": set(src.get("deleted_ids", set())),
        "manual_additions": copy.deepcopy(list(src.get("manual_additions", []))),
        "unmatched_manual_boxes": copy.deepcopy(list(src.get("unmatched_manual_boxes", []))),
    }


def _apply_label_state(dst: Dict[str, Any], state: Dict[str, Any]) -> None:
    dst["confirmed_ids"] = set(state.get("confirmed_ids", set()))
    dst["deleted_ids"] = set(state.get("deleted_ids", set()))
    dst["manual_additions"] = copy.deepcopy(list(state.get("manual_additions", [])))
    dst["unmatched_manual_boxes"] = copy.deepcopy(list(state.get("unmatched_manual_boxes", [])))


def _get_working_label_state(fstate: Dict[str, Any]) -> Dict[str, Any]:
    """Return the active label state dict (draft while editing, else committed fstate)."""
    if bool(fstate.get("edit_mode")) and isinstance(fstate.get("_edit_draft"), dict):
        return fstate["_edit_draft"]
    return fstate


def _enter_edit_mode(fstate: Dict[str, Any]) -> None:
    if bool(fstate.get("edit_mode")) and isinstance(fstate.get("_edit_draft"), dict):
        return
    baseline = _snapshot_label_state(fstate)
    draft = _snapshot_label_state(fstate)
    fstate["edit_mode"] = True
    fstate["_edit_baseline"] = baseline
    fstate["_edit_draft"] = draft
    # Track which confirmations are attributable to manual additions so removing a
    # manual record can revert to undecided unless explicitly confirmed.
    manual_ids = set()
    for rec in draft.get("manual_additions", []):
        cid = rec.get("snapped_candidate_id")
        if cid:
            manual_ids.add(str(cid))
    fstate["_edit_manual_confirmed_ids"] = set(draft.get("confirmed_ids", set())) & manual_ids


def _cancel_edit_mode(fstate: Dict[str, Any]) -> None:
    baseline = fstate.get("_edit_baseline")
    if isinstance(baseline, dict):
        _apply_label_state(fstate, baseline)
    fstate["edit_mode"] = False
    fstate["_edit_baseline"] = None
    fstate["_edit_draft"] = None
    fstate["_edit_manual_confirmed_ids"] = set()


def _save_edit_mode(fstate: Dict[str, Any]) -> None:
    draft = fstate.get("_edit_draft")
    if isinstance(draft, dict):
        _apply_label_state(fstate, draft)
    fstate["edit_mode"] = False
    fstate["_edit_baseline"] = None
    fstate["_edit_draft"] = None
    fstate["_edit_manual_confirmed_ids"] = set()


def _bbox_center_xy(bbox_xyxy: List[float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox_xyxy
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _point_in_bbox(pt: Tuple[float, float], bbox_xyxy: List[float]) -> bool:
    x, y = pt
    x0, y0, x1, y1 = bbox_xyxy
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def _snap_to_candidate(
    drawn_bbox_xyxy: List[float],
    *,
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Return (best_candidate, iou). Candidate bboxes are assumed full-res pixels."""
    nb = _normalize_bbox_xyxy(drawn_bbox_xyxy)
    if nb is None:
        return None, 0.0
    x0, y0, x1, y1 = nb
    drawn = [x0, y0, x1, y1]
    dw = max(1.0, x1 - x0)
    dh = max(1.0, y1 - y0)
    dcenter = _bbox_center_xy(drawn)

    # Only consider candidates that overlap the selection box (IoU>0). This matches the
    # “selector rectangle” UX and prevents snapping to nearby doors outside the box.
    best_iou = -1.0
    best_by_iou: Optional[Dict[str, Any]] = None
    best_inter = -1.0
    best_by_inter: Optional[Dict[str, Any]] = None
    any_overlap = False

    for cand in candidates:
        cid = cand.get("id")
        cb = _normalize_bbox_xyxy(cand.get("bbox_xyxy"))
        if cid is None or cb is None:
            continue
        cx0, cy0, cx1, cy1 = cb
        cbox = [cx0, cy0, cx1, cy1]
        iou = float(compute_iou(drawn, cbox))
        if iou <= 0.0:
            continue
        any_overlap = True
        if iou > best_iou:
            best_iou = iou
            best_by_iou = cand
        # Track maximum intersection area as a robust fallback (IoU can be low when
        # the selector rectangle is larger than the candidate).
        inter_x0 = max(drawn[0], cbox[0])
        inter_y0 = max(drawn[1], cbox[1])
        inter_x1 = min(drawn[2], cbox[2])
        inter_y1 = min(drawn[3], cbox[3])
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter = inter_w * inter_h
        if inter > best_inter:
            best_inter = inter
            best_by_inter = cand

    if not any_overlap:
        return None, 0.0

    # Primary: max IoU.
    MIN_SNAP_IOU = 0.02
    if best_by_iou is not None and best_iou >= MIN_SNAP_IOU:
        return best_by_iou, max(0.0, float(best_iou))

    # Fallback: max intersection area among overlapping candidates.
    if best_by_inter is not None and best_inter > 0.0:
        cb = _normalize_bbox_xyxy(best_by_inter.get("bbox_xyxy"))
        if cb is not None:
            return best_by_inter, max(0.0, float(compute_iou(drawn, [cb[0], cb[1], cb[2], cb[3]])))

    return None, 0.0


def _scale_bbox_xyxy(bbox_xyxy: List[float], scale: float) -> Optional[List[float]]:
    nb = _normalize_bbox_xyxy(bbox_xyxy)
    if nb is None:
        return None
    x0, y0, x1, y1 = nb
    return [x0 * scale, y0 * scale, x1 * scale, y1 * scale]


def _clamp_bbox_xyxy(bbox_xyxy: List[float], *, w: Optional[int], h: Optional[int]) -> List[float]:
    nb = _normalize_bbox_xyxy(bbox_xyxy)
    if nb is None:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = nb
    if w is not None and w > 0:
        x0 = max(0.0, min(float(w), x0))
        x1 = max(0.0, min(float(w), x1))
    if h is not None and h > 0:
        y0 = max(0.0, min(float(h), y0))
        y1 = max(0.0, min(float(h), y1))
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _bbox_intersects(a: List[float], b: List[float]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _expand_bbox(b: List[float], pad: float) -> List[float]:
    nb = _normalize_bbox_xyxy(b)
    if nb is None:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = nb
    p = float(pad or 0.0)
    return [x0 - p, y0 - p, x1 + p, y1 + p]


def _debug_unmatched_region(
    *,
    file_dir: Path,
    drawn_bbox_full_xyxy: List[float],
    config_path: str,
) -> Optional[str]:
    """Explain why a clearly-boxed door produced no candidate overlap.

    This runs only when Debug logs is enabled and prints to the server console.
    """
    try:
        if not st.session_state.get("debug_perf"):
            return None
    except Exception:
        return None

    primitives_path = file_dir / "primitives.json"
    meta_path = file_dir / "meta.json"
    if not primitives_path.exists() or not meta_path.exists():
        _debug_log("unmatched_debug missing primitives/meta in %s", str(file_dir))
        return None

    try:
        primitives = json.loads(primitives_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        _debug_log("unmatched_debug failed to read primitives/meta: %s", str(e))
        return None

    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as e:
        _debug_log("unmatched_debug failed to read config %s: %s", str(config_path), str(e))
        return None

    if (meta or {}).get("mode") == "scan":
        _debug_log("unmatched_debug meta.mode=scan (no vector candidates).")
        return None

    swing_conf = (cfg.get("swing") or {}) if isinstance(cfg, dict) else {}
    if not bool(swing_conf.get("enabled", False)):
        _debug_log("unmatched_debug swing.enabled=false (no swing candidates).")
        return None

    arc_conf = swing_conf.get("arc") or {}
    leaf_conf = swing_conf.get("leaf") or {}
    score_conf = swing_conf.get("scoring") or {}
    out_conf = cfg.get("output") or {}

    min_confidence = float(out_conf.get("min_confidence", 0.55) or 0.55)
    bezier_pts = int(swing_conf.get("bezier_sampling_points", 17) or 17)

    nb = _normalize_bbox_xyxy(drawn_bbox_full_xyxy)
    if nb is None:
        return None
    drawn = [nb[0], nb[1], nb[2], nb[3]]
    roi = _expand_bbox(drawn, pad=250.0)

    lines = list((primitives.get("lines") or []))
    beziers = list((primitives.get("beziers") or []))

    # Pre-filter lines by ROI for speed.
    line_infos: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        try:
            p0 = (float(line["p0"]["x"]), float(line["p0"]["y"]))
            p1 = (float(line["p1"]["x"]), float(line["p1"]["y"]))
        except Exception:
            continue
        lb = [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])]
        if _bbox_intersects(lb, roi):
            line_infos.append({"idx": i, "p0": p0, "p1": p1, "bbox": lb})

    # Bezier quick bbox from control points.
    def _bez_bbox(bez: Dict[str, Any]) -> Optional[List[float]]:
        try:
            xs = [float(bez[k]["x"]) for k in ("p0", "p1", "p2", "p3")]
            ys = [float(bez[k]["y"]) for k in ("p0", "p1", "p2", "p3")]
            return [min(xs), min(ys), max(xs), max(ys)]
        except Exception:
            return None

    # Counters for why arcs/leaf matches fail.
    arc_total = 0
    arc_near = 0
    arc_fail_radius = 0
    arc_fail_rmse = 0
    arc_fail_angle = 0
    arc_pass = 0

    leaf_tested = 0
    leaf_fail_len_ratio = 0
    leaf_fail_hinge = 0
    leaf_fail_center = 0
    leaf_fail_radial = 0
    leaf_fail_tip = 0
    leaf_pass = 0
    cand_below_conf = 0

    best_candidate: Optional[Dict[str, Any]] = None
    best_candidate_below: Optional[Dict[str, Any]] = None

    # Config thresholds (mirror door_detection).
    min_radius = float(arc_conf.get("min_radius_px", 8) or 8)
    max_radius = float(arc_conf.get("max_radius_px", 220) or 220)
    max_rmse = float(arc_conf.get("max_circle_fit_rmse", 2.5) or 2.5)
    min_angle = float(arc_conf.get("min_angle_deg", 20) or 20)
    max_angle = float(arc_conf.get("max_angle_deg", 125) or 125)

    min_len_ratio = float(leaf_conf.get("min_length_ratio", 0.45) or 0.45)
    max_len_ratio = float(leaf_conf.get("max_length_ratio", 1.25) or 1.25)
    max_hinge_ratio = float(leaf_conf.get("max_hinge_dist_ratio", 0.25) or 0.25)
    require_center = bool(leaf_conf.get("require_endpoint_near_center", False))
    max_center_ratio = float(leaf_conf.get("max_center_dist_ratio", max_hinge_ratio) or max_hinge_ratio)
    max_radial_angle = leaf_conf.get("max_radial_angle_deg", None)
    max_tip_ratio = leaf_conf.get("max_tip_to_arc_ratio", None)
    try:
        max_radial_angle = float(max_radial_angle) if max_radial_angle is not None else None
    except Exception:
        max_radial_angle = None
    try:
        max_tip_ratio = float(max_tip_ratio) if max_tip_ratio is not None else None
    except Exception:
        max_tip_ratio = None

    w_fit = float(score_conf.get("w_fit", 0.45) or 0.45)
    w_angle = float(score_conf.get("w_angle", 0.25) or 0.25)
    w_prox = float(score_conf.get("w_proximity", 0.30) or 0.30)

    for b_idx, bez in enumerate(beziers):
        arc_total += 1
        bb = _bez_bbox(bez)
        if bb is None or not _bbox_intersects(bb, roi):
            continue
        arc_near += 1

        # Sample and fit circle.
        try:
            pts = sample_bezier(bez["p0"], bez["p1"], bez["p2"], bez["p3"], num_points=bezier_pts)
            center, radius, rmse = fit_circle(pts)
            angle_span = float(get_arc_angle_span(pts, center))
        except Exception:
            continue

        # Arc gates.
        if not (min_radius <= float(radius) <= max_radius):
            arc_fail_radius += 1
            continue
        if float(rmse) > max_rmse:
            arc_fail_rmse += 1
            continue
        if not (min_angle <= float(angle_span) <= max_angle):
            arc_fail_angle += 1
            continue
        arc_pass += 1

        arc_bbox = get_bbox(pts)
        query_bbox = [
            arc_bbox[0] - float(radius) * 0.5,
            arc_bbox[1] - float(radius) * 0.5,
            arc_bbox[2] + float(radius) * 0.5,
            arc_bbox[3] + float(radius) * 0.5,
        ]

        # Consider only nearby lines (already ROI-filtered) that intersect the arc query region.
        for li in line_infos:
            if not _bbox_intersects(li["bbox"], query_bbox):
                continue
            leaf_tested += 1
            p0 = li["p0"]
            p1 = li["p1"]
            l_len = dist_point_to_point(p0, p1)
            if float(radius) <= 1e-6:
                continue
            len_ratio = float(l_len) / float(radius)
            if not (min_len_ratio <= len_ratio <= max_len_ratio):
                leaf_fail_len_ratio += 1
                continue

            arc_start = pts[0]
            arc_end = pts[-1]
            d0_start = dist_point_to_point(p0, arc_start)
            d0_end = dist_point_to_point(p0, arc_end)
            d1_start = dist_point_to_point(p1, arc_start)
            d1_end = dist_point_to_point(p1, arc_end)
            d0_center = dist_point_to_point(p0, center)
            d1_center = dist_point_to_point(p1, center)

            min_hinge_dist = min(d0_start, d0_end, d1_start, d1_end)
            if float(min_hinge_dist) > float(radius) * max_hinge_ratio:
                if min(d0_center, d1_center) > float(radius) * max_hinge_ratio:
                    leaf_fail_hinge += 1
                    continue

            if require_center and min(d0_center, d1_center) > float(radius) * max_center_ratio:
                leaf_fail_center += 1
                continue

            radial_angle_deg = None
            tip_to_arc_dist = None
            if max_radial_angle is not None and max_radial_angle > 0:
                hinge_pt = p0 if d0_center <= d1_center else p1
                tip_pt = p1 if hinge_pt == p0 else p0
                if dist_point_to_point(tip_pt, arc_start) <= dist_point_to_point(tip_pt, arc_end):
                    target_pt = arc_start
                else:
                    target_pt = arc_end
                tip_to_arc_dist = dist_point_to_point(tip_pt, target_pt)
                lx, ly = (tip_pt[0] - hinge_pt[0], tip_pt[1] - hinge_pt[1])
                rx, ry = (target_pt[0] - center[0], target_pt[1] - center[1])
                ln = math.hypot(lx, ly)
                rn = math.hypot(rx, ry)
                if ln > 1e-6 and rn > 1e-6:
                    dot = (lx * rx + ly * ry) / (ln * rn)
                    dot = max(-1.0, min(1.0, dot))
                    radial_angle_deg = math.degrees(math.acos(dot))
                    if radial_angle_deg > max_radial_angle:
                        leaf_fail_radial += 1
                        continue
                if max_tip_ratio is not None and max_tip_ratio > 0 and tip_to_arc_dist is not None:
                    if tip_to_arc_dist > float(radius) * max_tip_ratio:
                        leaf_fail_tip += 1
                        continue

            # Candidate confidence (mirror door_detection).
            fit_score = max(0.0, 1.0 - (float(rmse) / max_rmse)) if max_rmse > 0 else 0.0
            min_a = float(arc_conf.get("min_angle_deg", 55.0) or 55.0)
            denom = max(1.0, 90.0 - min_a)
            angle_score = (float(angle_span) - min_a) / denom
            angle_score = max(0.0, min(1.0, angle_score))
            prox_score = max(0.0, 1.0 - (float(min_hinge_dist) / (float(radius) * max_hinge_ratio))) if max_hinge_ratio > 0 else 0.0
            conf = w_fit * fit_score + w_angle * angle_score + w_prox * prox_score

            leaf_pass += 1
            cand = {
                "b_idx": int(b_idx),
                "l_idx": int(li["idx"]),
                "confidence": float(conf),
                "radius": float(radius),
                "rmse": float(rmse),
                "angle_span": float(angle_span),
                "min_hinge_dist": float(min_hinge_dist),
                "len_ratio": float(len_ratio),
                "center_dist": float(min(d0_center, d1_center)),
                "radial_angle_deg": float(radial_angle_deg) if radial_angle_deg is not None else None,
                "tip_to_arc_dist": float(tip_to_arc_dist) if tip_to_arc_dist is not None else None,
                "arc_bbox": arc_bbox,
                "line_bbox": li["bbox"],
            }
            if best_candidate is None or cand["confidence"] > float(best_candidate.get("confidence", 0.0)):
                best_candidate = cand
            if conf < min_confidence:
                cand_below_conf += 1
                if best_candidate_below is None or cand["confidence"] > float(best_candidate_below.get("confidence", 0.0)):
                    best_candidate_below = cand

    _debug_log(
        "unmatched_debug roi=%s drawn=%s lines_near=%d beziers_total=%d beziers_near=%d arcs_pass=%d arc_fail(radius=%d rmse=%d angle=%d)",
        str([round(v, 1) for v in roi]),
        str([round(v, 1) for v in drawn]),
        len(line_infos),
        int(arc_total),
        int(arc_near),
        int(arc_pass),
        int(arc_fail_radius),
        int(arc_fail_rmse),
        int(arc_fail_angle),
    )
    _debug_log(
        "unmatched_debug leaf tested=%d pass=%d fail(len=%d hinge=%d center=%d radial=%d tip=%d) min_conf=%.3f below_conf=%d",
        int(leaf_tested),
        int(leaf_pass),
        int(leaf_fail_len_ratio),
        int(leaf_fail_hinge),
        int(leaf_fail_center),
        int(leaf_fail_radial),
        int(leaf_fail_tip),
        float(min_confidence),
        int(cand_below_conf),
    )
    if best_candidate is not None:
        _debug_log("unmatched_debug best_candidate=%s", json.dumps(best_candidate, indent=2))
        if float(best_candidate.get("confidence", 0.0)) < min_confidence:
            _debug_log("unmatched_debug NOTE best_candidate below min_confidence=%.3f", float(min_confidence))
    elif arc_near == 0:
        _debug_log("unmatched_debug NOTE: no beziers near selection (no swing arcs to match).")
    elif arc_pass == 0:
        _debug_log("unmatched_debug NOTE: beziers near selection, but none pass arc thresholds.")
    elif leaf_pass == 0:
        _debug_log("unmatched_debug NOTE: arcs pass, but no leaf line matched leaf thresholds.")

    # Also return a compact report for UI display.
    try:
        report = {
            "roi": [round(v, 1) for v in roi],
            "drawn": [round(v, 1) for v in drawn],
            "lines_near": int(len(line_infos)),
            "beziers_total": int(arc_total),
            "beziers_near": int(arc_near),
            "arc_pass": int(arc_pass),
            "arc_fail": {
                "radius": int(arc_fail_radius),
                "rmse": int(arc_fail_rmse),
                "angle": int(arc_fail_angle),
            },
            "leaf": {
                "tested": int(leaf_tested),
                "pass": int(leaf_pass),
                "fail": {
                    "len_ratio": int(leaf_fail_len_ratio),
                    "hinge": int(leaf_fail_hinge),
                    "center": int(leaf_fail_center),
                    "radial": int(leaf_fail_radial),
                    "tip": int(leaf_fail_tip),
                },
            },
            "min_confidence": float(min_confidence),
            "below_conf": int(cand_below_conf),
            "best_candidate": best_candidate,
        }
        return json.dumps(report, indent=2)
    except Exception:
        return None


def _process_draw_event_if_any(
    *,
    file_id: str,
    file_dir: Path,
    fstate: Dict[str, Any],
    doors_data: Dict[str, Any],
    preview_spec: Optional[Dict[str, Any]],
    full_dims: Optional[Tuple[int, int]],
    config_path: str,
) -> None:
    """Consume a Shift+drag draw event from the iframe (if present)."""
    if not preview_spec:
        return
    draw_key = f"draw_event_sink_{file_id}"
    raw = st.session_state.get(draw_key) or ""
    if not raw:
        return
    try:
        evt = json.loads(raw)
    except Exception:
        return
    if not isinstance(evt, dict):
        return
    if evt.get("event") != "draw_rect":
        return
    event_id = evt.get("event_id")
    bbox = evt.get("bbox_xyxy")
    snapped_candidate_id = evt.get("snapped_candidate_id")
    if not event_id or not isinstance(bbox, list) or len(bbox) != 4:
        return
    if str(event_id) == str(fstate.get("_last_draw_event_id")):
        return
    fstate["_last_draw_event_id"] = str(event_id)
    _debug_log(
        "draw_event file_id=%s event_id=%s snapped_candidate_id=%s bbox_preview=%s",
        str(file_id),
        str(event_id),
        str(snapped_candidate_id),
        str(bbox),
    )

    # JS only emits in edit mode, but guard anyway.
    if not bool(fstate.get("edit_mode")):
        _debug_log("draw_event ignored (not in edit_mode) file_id=%s", str(file_id))
        return

    _enter_edit_mode(fstate)
    draft = _get_working_label_state(fstate)

    scale_full_to_preview = float(preview_spec.get("scale", 1.0) or 1.0)
    if not (scale_full_to_preview > 0):
        return

    # Convert preview → full-res pixels.
    try:
        x0p, y0p, x1p, y1p = [float(v) for v in bbox]
    except Exception:
        return
    drawn_full = [x0p / scale_full_to_preview, y0p / scale_full_to_preview, x1p / scale_full_to_preview, y1p / scale_full_to_preview]

    full_w = full_dims[0] if full_dims else None
    full_h = full_dims[1] if full_dims else None
    drawn_full = _clamp_bbox_xyxy(drawn_full, w=full_w, h=full_h)

    candidates = list(doors_data.get("doors", []) or [])
    best = None
    iou = 0.0

    if snapped_candidate_id:
        sid = str(snapped_candidate_id)
        best = next((c for c in candidates if str(c.get("id") or "") == sid), None)
        if best is not None:
            cb = _normalize_bbox_xyxy(best.get("bbox_xyxy"))
            if cb is not None:
                iou = float(compute_iou(drawn_full, [cb[0], cb[1], cb[2], cb[3]]))
                # Reject client-proposed snaps that don't actually overlap the selection.
                if iou <= 0.0:
                    best = None
    if best is None:
        best, iou = _snap_to_candidate(drawn_full, candidates=candidates)

    if best is not None and best.get("id") is not None:
        cid = str(best["id"])
        _debug_log("draw_event snap matched file_id=%s cid=%s iou=%.4f", str(file_id), cid, float(iou))
        snapped_full = _normalize_bbox_xyxy(best.get("bbox_xyxy")) or _normalize_bbox_xyxy(drawn_full) or (0.0, 0.0, 0.0, 0.0)
        rec = {
            "drawn_bbox_xyxy": drawn_full,
            "snapped_candidate_id": cid,
            "iou": float(iou),
            "snapped_bbox_xyxy": [float(snapped_full[0]), float(snapped_full[1]), float(snapped_full[2]), float(snapped_full[3])],
        }
        draft["manual_additions"].append(rec)
        draft["confirmed_ids"].add(cid)
        draft["deleted_ids"].discard(cid)
        try:
            fstate["_edit_manual_confirmed_ids"].add(cid)
        except Exception:
            pass
        # Make the snapped door the current selection.
        try:
            fstate["selected_door_id"] = cid
            st.session_state[f"jump_{file_id}"] = cid
        except Exception:
            pass
    else:
        _debug_log("draw_event snap unmatched file_id=%s", str(file_id))
        draft["unmatched_manual_boxes"].append(
            {
                "bbox_xyxy": drawn_full,
                "note": "No candidate match",
            }
        )
        fstate["_last_unmatched_debug"] = _debug_unmatched_region(
            file_dir=file_dir,
            drawn_bbox_full_xyxy=drawn_full,
            config_path=str(config_path),
        )

    # Consume the sink value so it doesn't grow / re-trigger on reconnects.
    try:
        st.session_state[draw_key] = ""
    except Exception:
        pass


def _manual_overlay_payload_for_sink(
    *,
    fstate: Dict[str, Any],
    preview_scale: float,
) -> Dict[str, Any]:
    """Return preview-space overlays for the iframe (NOT saved to labels.json)."""
    state = _get_working_label_state(fstate)
    out_manual: List[Dict[str, Any]] = []
    out_unmatched: List[Dict[str, Any]] = []

    if not (preview_scale > 0):
        preview_scale = 1.0

    for rec in list(state.get("manual_additions", [])):
        drawn_full = rec.get("drawn_bbox_xyxy")
        if not isinstance(drawn_full, list) or len(drawn_full) != 4:
            continue
        drawn_prev = _scale_bbox_xyxy([float(v) for v in drawn_full], preview_scale)
        if drawn_prev is None:
            continue
        snapped_prev = None
        snapped_full = rec.get("snapped_bbox_xyxy")
        if isinstance(snapped_full, list) and len(snapped_full) == 4:
            snapped_prev = _scale_bbox_xyxy([float(v) for v in snapped_full], preview_scale)
        out_manual.append(
            {
                "drawn_bbox_xyxy": drawn_prev,
                "snapped_bbox_xyxy": snapped_prev,
                "snapped_candidate_id": rec.get("snapped_candidate_id"),
                "iou": rec.get("iou"),
            }
        )

    for rec in list(state.get("unmatched_manual_boxes", [])):
        bbox_full = rec.get("bbox_xyxy")
        if not isinstance(bbox_full, list) or len(bbox_full) != 4:
            continue
        bbox_prev = _scale_bbox_xyxy([float(v) for v in bbox_full], preview_scale)
        if bbox_prev is None:
            continue
        out_unmatched.append({"bbox_xyxy": bbox_prev, "note": rec.get("note")})

    return {"manual_additions": out_manual, "unmatched_manual_boxes": out_unmatched}


def init_file_state(file_id: str, doors_data: Dict, labels_data: Dict):
    if "files" not in st.session_state:
        st.session_state.files = {}
    
    if file_id not in st.session_state.files:
        st.session_state.files[file_id] = {
            "confirmed_ids": set(labels_data.get("confirmed_ids", [])),
            "deleted_ids": set(labels_data.get("deleted_ids", [])),
            "manual_additions": list(labels_data.get("manual_additions", [])),
            "unmatched_manual_boxes": list(labels_data.get("unmatched_manual_boxes", [])),
            "selected_door_id": None,
            "viewer_display_mode": "Highlight All",
            "auto_focus": True,
            "edit_mode": False,
            "_edit_baseline": None,
            "_edit_draft": None,
            "_edit_manual_confirmed_ids": set(),
            "_last_draw_event_id": None,
            "_last_unmatched_debug": None,
            "_focus_seq": 0,
            "_focus_last_id": None,
            "_last_clicked_door_id": None,
        }

# --- Data Loading ---
def load_file_artifacts(file_dir_str: str):
    file_dir = Path(file_dir_str)
    doors_path = file_dir / "doors.json"
    labels_path = file_dir / "labels.json"
    meta_path = file_dir / "meta.json"
    
    doors_data = {}
    if doors_path.exists():
        with open(doors_path) as f:
            doors_data = json.load(f)
            
    labels_data: Dict[str, Any] = _labels_v2_default()
    if labels_path.exists():
        with open(labels_path) as f:
            labels_data = json.load(f)
        _validate_labels_v2_or_raise(labels_data, labels_path=labels_path)
            
    meta_data = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta_data = json.load(f)

    return doors_data, labels_data, meta_data


def _get_full_page_dims(meta_data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Return (width, height) for the full-resolution page (page.png), if known."""
    try:
        w = int(meta_data.get("pix_width"))
        h = int(meta_data.get("pix_height"))
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        return None
    return None


def _get_preview_path(file_dir: Path) -> Path:
    # Stable filename so existing artifacts benefit after first view.
    return file_dir / "page_view.jpg"


@st.cache_resource(show_spinner=False)
def get_or_create_page_preview(
    file_dir_str: str,
    *,
    full_width: Optional[int],
    full_height: Optional[int],
    page_png_mtime_ns: int,
    preview_max_width: int = 2400,
) -> Optional[Dict[str, Any]]:
    """Return a lightweight preview image spec for UI rendering.

    Creates `page_view.jpg` on disk if missing by downscaling `page.png` once.
    Returns:
      { path, url, width, height, scale }
    Where `scale` maps full-res pixel coords -> preview coords.
    """
    file_dir = Path(file_dir_str)
    page_png_path = file_dir / "page.png"
    if not page_png_path.exists():
        return None

    preview_path = _get_preview_path(file_dir)

    # Create preview lazily (once) to avoid re-decoding huge PNG on every rerun.
    preview_is_stale = False
    if preview_path.exists():
        try:
            preview_is_stale = preview_path.stat().st_mtime_ns < int(page_png_mtime_ns)
        except Exception:
            preview_is_stale = False

    if (not preview_path.exists()) or preview_is_stale:
        src = Image.open(page_png_path)
        try:
            src_w, src_h = src.size
            # If meta is missing, fall back to the source image dimensions.
            if not full_width or not full_height:
                full_width, full_height = src_w, src_h

            scale_create = min(1.0, float(preview_max_width) / float(src_w)) if src_w else 1.0
            if scale_create < 1.0:
                out_w = max(1, int(round(src_w * scale_create)))
                out_h = max(1, int(round(src_h * scale_create)))
                prev = src.resize((out_w, out_h), Image.LANCZOS)
            else:
                # Still write a JPEG so the UI never has to decode the huge PNG.
                prev = src

            if prev.mode != "RGB":
                prev = prev.convert("RGB")

            preview_path.parent.mkdir(parents=True, exist_ok=True)
            prev.save(preview_path, "JPEG", quality=88, optimize=True, progressive=True)
        finally:
            try:
                src.close()
            except Exception:
                pass

    prev_img = Image.open(preview_path)
    try:
        prev_w, prev_h = prev_img.size
    finally:
        try:
            prev_img.close()
        except Exception:
            pass

    # Compute a stable scale factor from full-res → preview coords.
    if full_width and full_width > 0:
        scale = float(prev_w) / float(full_width)
    else:
        # Last-resort: treat preview as full-res (should be rare).
        scale = 1.0

    return {
        "path": str(preview_path),
        "width": int(prev_w),
        "height": int(prev_h),
        "scale": float(scale),
    }

def save_labels(dir_path: Path, labels_data: Dict[str, Any]):
    labels_path = dir_path / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)

def get_current_signature(config_path: str):
    try:
        return compute_analysis_signature(Path(config_path))
    except Exception:
        return None

# --- UI Components ---

def _normalize_bbox_xyxy(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return (x0, y0, x1, y1) with x0<=x1 and y0<=y1, or None if invalid."""
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def _image_path_to_streamlit_url(image_path: str) -> str:
    """Return a URL (or small data URL fallback) for an on-disk image."""
    p = Path(image_path)
    try:
        img = Image.open(p)
        try:
            from streamlit.elements.lib.image_utils import image_to_url
            from streamlit.elements.lib.layout_utils import LayoutConfig

            st_image_id = f"img|{p}|{p.stat().st_mtime_ns}"
            url = image_to_url(
                image=img,
                layout_config=LayoutConfig(width=None),
                clamp=False,
                channels="RGB",
                output_format="JPEG",
                image_id=st_image_id,
            )
            if url:
                return url
        finally:
            try:
                img.close()
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: embed the preview (should be small).
    try:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""

def _rects_to_svg(
    *,
    active_doors: List[Dict[str, Any]],
    fstate: Dict[str, Any],
    scale: float,
    img_width: int,
    img_height: int,
) -> str:
    parts: List[str] = []

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    for d in active_doors:
        did = d.get("id")

        nb = _normalize_bbox_xyxy(d.get("bbox_xyxy"))
        if nb is None:
            continue
        x0, y0, x1, y1 = nb

        # Map full-res pixels → preview pixels.
        x0p = x0 * scale
        y0p = y0 * scale
        x1p = x1 * scale
        y1p = y1 * scale

        x0p = _clamp(x0p, 0.0, float(img_width))
        y0p = _clamp(y0p, 0.0, float(img_height))
        x1p = _clamp(x1p, 0.0, float(img_width))
        y1p = _clamp(y1p, 0.0, float(img_height))

        w = max(0.0, x1p - x0p)
        h = max(0.0, y1p - y0p)
        if w <= 0.0 or h <= 0.0:
            continue

        # Styling is applied client-side (confirmed/selected/deleted/view modes) so the
        # iframe does not need to remount for label-only state changes.
        stroke = "#ffa500"  # default (orange)
        stroke_width = 2
        did_attr = html.escape(str(did), quote=True)
        parts.append(
            f'<rect x="{x0p:.2f}" y="{y0p:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'vector-effect="non-scaling-stroke" '
            f'data-door-id="{did_attr}" '
            f'data-x="{x0p:.2f}" data-y="{y0p:.2f}" data-w="{w:.2f}" data-h="{h:.2f}" '
            f'style="pointer-events: all; cursor: pointer;" />'
        )

    return "\n".join(parts)

def _panzoom_image_viewer(
    *,
    img_src: str,
    img_width: int,
    img_height: int,
    rects_svg: str,
    height: int,
    key: str,
    click_sink_aria_label: str,
    selected_sink_aria_label: str,
    focus_seq_sink_aria_label: str,
    edit_mode_sink_aria_label: str,
    draw_event_sink_aria_label: str,
    manual_overlay_sink_aria_label: str,
    door_state_sink_aria_label: str,
    viewer_display_sink_aria_label: str,
    auto_focus_sink_aria_label: str,
) -> None:
    # This viewer provides:
    # - scrollwheel zoom (centered at cursor)
    # - click+drag pan
    # - initial fit-to-container with letterboxing
    click_sink_aria_label_esc = html.escape(click_sink_aria_label, quote=True)
    selected_sink_aria_label_esc = html.escape(selected_sink_aria_label, quote=True)
    focus_seq_sink_aria_label_esc = html.escape(focus_seq_sink_aria_label, quote=True)
    edit_mode_sink_aria_label_esc = html.escape(edit_mode_sink_aria_label, quote=True)
    draw_event_sink_aria_label_esc = html.escape(draw_event_sink_aria_label, quote=True)
    manual_overlay_sink_aria_label_esc = html.escape(manual_overlay_sink_aria_label, quote=True)
    door_state_sink_aria_label_esc = html.escape(door_state_sink_aria_label, quote=True)
    viewer_display_sink_aria_label_esc = html.escape(viewer_display_sink_aria_label, quote=True)
    auto_focus_sink_aria_label_esc = html.escape(auto_focus_sink_aria_label, quote=True)
    viewer_html = f"""
{_sidebar_autopen_component_html()}
<div id="pz_root_{key}" style="width: 100%; height: {height}px; overflow: hidden; background: #0e1117; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.12);">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
    }}
    /* Scoped to this component instance */
    #pz_root_{key} .pz-reset {{
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 5;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      background: rgba(17, 25, 40, 0.65);
      color: rgba(255, 255, 255, 0.92);
      font-size: 12px;
      line-height: 1;
      cursor: pointer;
      backdrop-filter: blur(6px);
      -webkit-backdrop-filter: blur(6px);
      box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35);
      opacity: 0;
      transform: translateY(-4px);
      transition: opacity 120ms ease, transform 120ms ease, border-color 120ms ease, background 120ms ease;
      user-select: none;
      pointer-events: none; /* enabled only when visible */
    }}
    #pz_root_{key} .pz-reset.pz-reset--visible {{
      opacity: 1;
      transform: translateY(0px);
      pointer-events: auto;
    }}
    #pz_root_{key} .pz-reset:hover {{
      background: rgba(17, 25, 40, 0.82);
      border-color: rgba(255, 255, 255, 0.26);
    }}
    #pz_root_{key} .pz-reset:active {{
      transform: translateY(0px) scale(0.98);
    }}
  </style>
  <div id="pz_stage_{key}" style="width: 100%; height: 100%; position: relative; user-select: none; cursor: grab; touch-action: none;">
    <button id="pz_reset_{key}" class="pz-reset" type="button" aria-label="Reset zoom" aria-hidden="true" tabindex="-1">Reset</button>
    <div id="pz_content_{key}" style="position: absolute; left: 0; top: 0; width: {int(img_width)}px; height: {int(img_height)}px; transform-origin: 0 0; will-change: transform;">
      <img
        id="pz_img_{key}"
        src="{img_src}"
        width="{int(img_width)}"
        height="{int(img_height)}"
        style="position: absolute; left: 0; top: 0; pointer-events: none;"
      />
      <svg
        id="pz_svg_{key}"
        width="{int(img_width)}"
        height="{int(img_height)}"
        viewBox="0 0 {int(img_width)} {int(img_height)}"
        style="position: absolute; left: 0; top: 0; pointer-events: auto;"
        xmlns="http://www.w3.org/2000/svg"
      >
        {rects_svg}
      </svg>
    </div>
  </div>
</div>

<script>
(function() {{
  const root = document.getElementById("pz_root_{key}");
  const stage = document.getElementById("pz_stage_{key}");
  const content = document.getElementById("pz_content_{key}");
  const img = document.getElementById("pz_img_{key}");
  const svg = document.getElementById("pz_svg_{key}");
  const resetBtn = document.getElementById("pz_reset_{key}");
  if (!root || !stage || !content || !img) return;

  const persistKey = "door_detector_pz_state_{html.escape(str(key), quote=True)}";
  const clickSinkLabel = "{click_sink_aria_label_esc}";
  const selectedSinkLabel = "{selected_sink_aria_label_esc}";
  const focusSeqSinkLabel = "{focus_seq_sink_aria_label_esc}";
  const editModeSinkLabel = "{edit_mode_sink_aria_label_esc}";
  const drawEventSinkLabel = "{draw_event_sink_aria_label_esc}";
  const manualOverlaySinkLabel = "{manual_overlay_sink_aria_label_esc}";
  const doorStateSinkLabel = "{door_state_sink_aria_label_esc}";
  const viewerDisplaySinkLabel = "{viewer_display_sink_aria_label_esc}";

  const autoFocusSinkLabel = "{auto_focus_sink_aria_label_esc}";
  function parseBool(v, defaultValue) {{
    if (v === true) return true;
    if (v === false) return false;
    const s = String(v ?? "").trim().toLowerCase();
    if (s === "1" || s === "true" || s === "t" || s === "yes" || s === "y" || s === "on") return true;
    if (s === "0" || s === "false" || s === "f" || s === "no" || s === "n" || s === "off" || s === "") return false;
    return !!defaultValue;
  }}
  function getAutoFocus() {{
    return parseBool(readParentInputValue(autoFocusSinkLabel), true);
  }}
  let autoFocus = getAutoFocus();

  try {{
    console.log("[door_detector] pz init", {{
      key: {json.dumps(str(key))},
      autoFocus,
      persistKey,
      clickSinkLabel,
      selectedSinkLabel,
      focusSeqSinkLabel,
      editModeSinkLabel,
      drawEventSinkLabel,
      manualOverlaySinkLabel,
      doorStateSinkLabel,
      viewerDisplaySinkLabel,
      autoFocusSinkLabel,
      ts: Date.now(),
    }});
  }} catch (_) {{}}

  let scale = 1;
  let tx = 0;
  let ty = 0;

  // "Original state" = initial fit-to-container transform.
  let baseScale = 1;
  let baseTx = 0;
  let baseTy = 0;

  let dragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartTx = 0;
  let dragStartTy = 0;

  function clamp(v, lo, hi) {{
    return Math.max(lo, Math.min(hi, v));
  }}

  function applyTransform() {{
    content.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
    updateResetVisibility();
    scheduleSaveState();
  }}

  function isAtBase() {{
    const eps = 0.5; // pixels for tx/ty (and ~0.5 for scale*1000 below)
    const scaleClose = Math.abs(scale - baseScale) < 0.0005;
    const txClose = Math.abs(tx - baseTx) < eps;
    const tyClose = Math.abs(ty - baseTy) < eps;
    return scaleClose && txClose && tyClose;
  }}

  function updateResetVisibility() {{
    if (!resetBtn) return;
    const visible = !isAtBase();
    if (visible) {{
      resetBtn.classList.add("pz-reset--visible");
      resetBtn.setAttribute("aria-hidden", "false");
      resetBtn.tabIndex = 0;
    }} else {{
      resetBtn.classList.remove("pz-reset--visible");
      resetBtn.setAttribute("aria-hidden", "true");
      resetBtn.tabIndex = -1;
    }}
  }}

  function fitToContainer() {{
    const cw = root.clientWidth;
    const ch = root.clientHeight;
    // Use the known image pixel dimensions passed from Python.
    // Relying on `img.width` can be wrong if CSS/layout clamps the image.
    const iw = {int(img_width)};
    const ih = {int(img_height)};
    if (!cw || !ch || !iw || !ih) return;

    // Initial zoom should be the MAX zoom that keeps the entire image visible:
    // fit-to-container in BOTH directions (no initial cropping on extreme aspect ratios).
    const pad = 6; // subtle breathing room against the rounded border
    const cwPad = cw - pad * 2;
    const chPad = ch - pad * 2;
    if (cwPad <= 0 || chPad <= 0) return;
    scale = clamp(Math.min(cwPad / iw, chPad / ih), 0.05, 20);
    tx = (cw - iw * scale) / 2;
    ty = (ch - ih * scale) / 2;

    baseScale = scale;
    baseTx = tx;
    baseTy = ty;
    applyTransform();
  }}

  function loadState() {{
    try {{
      const raw = sessionStorage.getItem(persistKey);
      if (!raw) return null;
      const obj = JSON.parse(raw);
      if (!obj) return null;
      if (!Number.isFinite(obj.tx) || !Number.isFinite(obj.ty) || !Number.isFinite(obj.scale)) return null;
      return obj;
    }} catch (_) {{
      return null;
    }}
  }}

  function readParentInputValue(label) {{
    try {{
      const input = window.parent?.document?.querySelector(`input[aria-label="${{label}}"]`);
      return input ? String(input.value || "") : "";
    }} catch (_) {{
      return "";
    }}
  }}

  function setParentInputValue(label, value) {{
    try {{
      const doc = window.parent?.document;
      if (!doc || !label) return false;
      let input = null;
      try {{
        input = doc.querySelector(`input[aria-label="${{label}}"]`);
      }} catch (_) {{
        input = null;
      }}
      if (!input) {{
        // Fallback: Streamlit sometimes mutates aria-label; allow prefix match.
        try {{
          input = doc.querySelector(`input[aria-label^="${{label}}"]`);
        }} catch (_) {{
          input = null;
        }}
      }}
      if (!input) return false;

      // Use the native setter so React/Streamlit observe the change.
      const v = String(value);
      const prev = String(input.value || "");
      // Some Streamlit builds may only commit widget state on focused inputs.
      try {{ input.focus({{ preventScroll: true }}); }} catch (_) {{}}
      // Streamlit's frontend is React; resetting the value tracker makes "input"
      // events reliably propagate widget state updates.
      let hadTracker = false;
      try {{
        const tracker = input._valueTracker;
        if (tracker && tracker.setValue) {{
          hadTracker = true;
          tracker.setValue(prev);
        }}
      }} catch (_) {{}}
      try {{
        // IMPORTANT: dispatch events created from the *parent realm* so React's
        // event system sees them (cross-iframe Event instances can be ignored).
        const evWin = input.ownerDocument && input.ownerDocument.defaultView ? input.ownerDocument.defaultView : (window.parent || window);
        const setter = Object.getOwnPropertyDescriptor(evWin.HTMLInputElement.prototype, "value")?.set;
        if (setter) setter.call(input, v);
        else input.value = v;
      }} catch (_) {{
        input.value = v;
      }}

      // Dispatch events from the parent realm (see note above).
      let inputEvtOk = false;
      let changeEvtOk = false;
      try {{
        const evWin = input.ownerDocument && input.ownerDocument.defaultView ? input.ownerDocument.defaultView : (window.parent || window);
        const InputEv = evWin.InputEvent || evWin.Event;
        const inputEvt = evWin.InputEvent
          ? new evWin.InputEvent("input", {{ bubbles: true, cancelable: true, data: v, inputType: "insertReplacementText" }})
          : new InputEv("input", {{ bubbles: true, cancelable: true }});
        inputEvtOk = !!input.dispatchEvent(inputEvt);
        const changeEvt = new evWin.Event("change", {{ bubbles: true, cancelable: true }});
        changeEvtOk = !!input.dispatchEvent(changeEvt);
      }} catch (_) {{
        // Fallback (same-realm) if the above fails for any reason.
        try {{ inputEvtOk = !!input.dispatchEvent(new Event("input", {{ bubbles: true }})); }} catch (_) {{}}
        try {{ changeEvtOk = !!input.dispatchEvent(new Event("change", {{ bubbles: true }})); }} catch (_) {{}}
      }}
      try {{
        window.__door_detectorLastSinkWrite = {{
          ok: true,
          label,
          ariaLabel: input.getAttribute ? input.getAttribute("aria-label") : null,
          prev,
          next: v,
          after: String(input.value || ""),
          hadTracker,
          inputEvtOk,
          changeEvtOk,
          ts: Date.now(),
        }};
      }} catch (_) {{}}
      try {{ input.blur(); }} catch (_) {{}}
      return true;
    }} catch (_) {{
      try {{
        window.__door_detectorLastSinkWrite = {{ ok: false, label, ts: Date.now(), err: "exception" }};
      }} catch (_) {{}}
      return false;
    }}
  }}

  function readParentJson(label) {{
    try {{
      const raw = readParentInputValue(label);
      if (!raw) return null;
      return JSON.parse(raw);
    }} catch (_) {{
      return null;
    }}
  }}

  function getSelectedId() {{
    const v = readParentInputValue(selectedSinkLabel);
    return v ? v : null;
  }}

  function getFocusSeq() {{
    const raw = readParentInputValue(focusSeqSinkLabel);
    const n = parseInt(raw || "0", 10);
    return Number.isFinite(n) ? n : 0;
  }}

  function findRectByDoorId(doorId) {{
    if (!svg || !doorId) return null;
    const rects = svg.querySelectorAll("rect[data-door-id]");
    for (const r of rects) {{
      if (r && r.getAttribute && r.getAttribute("data-door-id") === doorId) return r;
    }}
    return null;
  }}

  function getEditEnabled() {{
    return readParentInputValue(editModeSinkLabel) === "1";
  }}

  function getViewerDisplayMode() {{
    const v = readParentInputValue(viewerDisplaySinkLabel);
    return v ? v : "all";
  }}

  let lastDoorStateRaw = null;
  let confirmedSet = new Set();
  let deletedSet = new Set();
  let lastViewerDisplay = null;
  let lastEditEnabled = null;
  let localSelectedId = null;

  function updateDoorStateFromSinks() {{
    const raw = readParentInputValue(doorStateSinkLabel);
    if (raw === lastDoorStateRaw) return false;
    lastDoorStateRaw = raw;
    const obj = readParentJson(doorStateSinkLabel) || {{}};
    const c = Array.isArray(obj.confirmed_ids) ? obj.confirmed_ids : [];
    const d = Array.isArray(obj.deleted_ids) ? obj.deleted_ids : [];
    confirmedSet = new Set(c.map(String));
    deletedSet = new Set(d.map(String));
    return true;
  }}

  function applyDoorStyles() {{
    if (!svg) return;
    // Prefer local selection for immediate feedback; Streamlit selection takes over after rerun.
    const selectedId = localSelectedId || getSelectedId();
    const viewMode = getViewerDisplayMode();

    const rects = svg.querySelectorAll("rect[data-door-id]");
    try {{
      if (!applyDoorStyles._loggedOnce) {{
        applyDoorStyles._loggedOnce = true;
        const first = rects && rects.length ? rects[0] : null;
        let bb = null;
        try {{ bb = first ? first.getBBox() : null; }} catch (_) {{ bb = null; }}

        // Compute bbox range across all candidates (helps spot coord mismatches).
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        if (rects) {{
          for (const rr of rects) {{
            let b = null;
            try {{ b = rr.getBBox(); }} catch (_) {{ b = null; }}
            if (!b) continue;
            const x0 = Number(b.x), y0 = Number(b.y);
            const x1 = x0 + Number(b.width), y1 = y0 + Number(b.height);
            if (!Number.isFinite(x0) || !Number.isFinite(y0) || !Number.isFinite(x1) || !Number.isFinite(y1)) continue;
            minX = Math.min(minX, x0);
            minY = Math.min(minY, y0);
            maxX = Math.max(maxX, x1);
            maxY = Math.max(maxY, y1);
          }}
        }}

        const payload = {{
          count: rects ? rects.length : 0,
          firstId: first ? first.getAttribute("data-door-id") : null,
          firstBBox: bb ? {{ x: Number(bb.x), y: Number(bb.y), w: Number(bb.width), h: Number(bb.height) }} : null,
          candRange: (minX !== Infinity) ? [minX, minY, maxX, maxY] : null,
          svgViewBox: svg ? svg.getAttribute("viewBox") : null,
        }};
        console.log("[door_detector] applyDoorStyles rects", payload);
        console.log("[door_detector] applyDoorStyles rects json", JSON.stringify(payload));
      }}
    }} catch (_) {{}}
    for (const r of rects) {{
      const did = r.getAttribute("data-door-id");
      if (!did) continue;

      const isSelected = selectedId && did === selectedId;
      const isDeleted = deletedSet.has(did);

      let visible = true;
      if (viewMode === "off") visible = false;
      else if (viewMode === "selected") visible = !!isSelected;
      if (isDeleted) visible = false;

      if (!visible) {{
        r.style.display = "none";
        r.style.pointerEvents = "none";
        continue;
      }}

      r.style.display = "";
      r.style.pointerEvents = "all";

      if (isSelected) {{
        r.setAttribute("stroke", "#ff4b4b");
        r.setAttribute("stroke-width", "3");
        try {{ svg.appendChild(r); }} catch (_) {{}}
      }} else if (confirmedSet.has(did)) {{
        r.setAttribute("stroke", "#00ff00");
        r.setAttribute("stroke-width", "2");
      }} else {{
        r.setAttribute("stroke", "#ffa500");
        r.setAttribute("stroke-width", "2");
      }}
    }}
  }}

  let saveTimer = null;
  function scheduleSaveState() {{
    try {{
      if (saveTimer) clearTimeout(saveTimer);
      saveTimer = setTimeout(() => {{
        saveTimer = null;
        try {{
          sessionStorage.setItem(
            persistKey,
            JSON.stringify({{ tx: tx, ty: ty, scale: scale, focusSeq: focusSeq }})
          );
        }} catch (_) {{}}
      }}, 100);
    }} catch (_) {{}}
  }}

  function easeInOut(t) {{
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }}

  function animateTo(targetTx, targetTy, targetScale, durationMs) {{
    const start = performance.now();
    const sTx = tx, sTy = ty, sScale = scale;
    const dTx = targetTx - sTx;
    const dTy = targetTy - sTy;
    const dScale = targetScale - sScale;

    function step(now) {{
      const t = clamp((now - start) / durationMs, 0, 1);
      const e = easeInOut(t);
      tx = sTx + dTx * e;
      ty = sTy + dTy * e;
      scale = sScale + dScale * e;
      applyTransform();
      if (t < 1) requestAnimationFrame(step);
    }}

    requestAnimationFrame(step);
  }}

  function focusToBBox(bbox) {{
    if (!bbox || !Array.isArray(bbox) || bbox.length !== 4) return;
    const cw = root.clientWidth;
    const ch = root.clientHeight;
    if (!cw || !ch) return;

    const x0 = bbox[0], y0 = bbox[1], x1 = bbox[2], y1 = bbox[3];
    const bx0 = Math.min(x0, x1);
    const by0 = Math.min(y0, y1);
    const bx1 = Math.max(x0, x1);
    const by1 = Math.max(y0, y1);

    const bw = Math.max(1, bx1 - bx0);
    const bh = Math.max(1, by1 - by0);
    const cx = (bx0 + bx1) / 2;
    const cy = (by0 + by1) / 2;

    // Show some context around the selected door (larger = less zoomed-in).
    const padFactor = 3.0;
    const targetScale = clamp(Math.min(cw / (bw * padFactor), ch / (bh * padFactor)), baseScale, baseScale * 6);
    const targetTx = (cw / 2) - targetScale * cx;
    const targetTy = (ch / 2) - targetScale * cy;

    try {{
      console.log("[door_detector] pz focusToBBox", {{
        key: {json.dumps(str(key))},
        focusSeq,
        bbox,
        target: {{ tx: targetTx, ty: targetTy, scale: targetScale }},
        ts: Date.now(),
      }});
    }} catch (_) {{}}
    animateTo(targetTx, targetTy, targetScale, 260);
  }}

  function focusToDoorId(doorId) {{
    const r = findRectByDoorId(doorId);
    if (!r) return;
    const x = parseFloat(r.getAttribute("data-x") || r.getAttribute("x") || "0");
    const y = parseFloat(r.getAttribute("data-y") || r.getAttribute("y") || "0");
    const w = parseFloat(r.getAttribute("data-w") || r.getAttribute("width") || "0");
    const h = parseFloat(r.getAttribute("data-h") || r.getAttribute("height") || "0");
    if (!(w > 0) || !(h > 0)) return;
    focusToBBox([x, y, x + w, y + h]);
  }}

  function zoomAt(clientX, clientY, zoomFactor) {{
    const rect = root.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;

    const oldScale = scale;
    scale = clamp(scale * zoomFactor, 0.05, 20);

    // Keep the point under the cursor stable:
    // p = t + s * q  => q = (p - t)/s
    const qx = (px - tx) / oldScale;
    const qy = (py - ty) / oldScale;
    tx = px - scale * qx;
    ty = py - scale * qy;
    applyTransform();
  }}

  // Fit once image is ready.
  function applyInitialView() {{
    // Always establish base fit first.
    fitToContainer();

    // Restore last view first so focus can animate from the current pan/zoom.
    const saved = loadState();
    if (saved) {{
      try {{
        console.log("[door_detector] pz restoreState", {{
          key: {json.dumps(str(key))},
          saved,
          ts: Date.now(),
        }});
      }} catch (_) {{}}
      scale = clamp(saved.scale, 0.05, 20);
      tx = saved.tx;
      ty = saved.ty;
      applyTransform();
    }}

    // Sync highlight and optional focus based on parent state.
    const doorId = getSelectedId();
    const seq = getFocusSeq();
    updateDoorStateFromSinks();
    applyDoorStyles();
    renderManualOverlays();
    if (doorId && autoFocus && (!saved || saved.focusSeq !== seq)) {{
      try {{
        console.log("[door_detector] pz autoFocus", {{
          key: {json.dumps(str(key))},
          fromSaved: !!saved,
          savedFocusSeq: saved ? saved.focusSeq : null,
          focusSeq: seq,
          ts: Date.now(),
        }});
      }} catch (_) {{}}
      focusToDoorId(doorId);
    }}
  }}

  if (img.complete) applyInitialView();
  else img.addEventListener("load", applyInitialView, {{ once: true }});

  // Keep the "original state" in sync on resize, but only if user hasn't deviated.
  try {{
    const ro = new ResizeObserver(() => {{
      if (isAtBase()) fitToContainer();
    }});
    ro.observe(root);
  }} catch (_) {{}}

  if (resetBtn) {{
    // Prevent pan-drag initiation when clicking the reset button.
    resetBtn.addEventListener("pointerdown", (e) => {{
      e.preventDefault();
      e.stopPropagation();
    }});
    resetBtn.addEventListener("click", (e) => {{
      e.preventDefault();
      e.stopPropagation();
      // Reset back to original fit-to-container for current container size.
      fitToContainer();
    }});
  }}

  root.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomAt(e.clientX, e.clientY, zoomFactor);
  }}, {{ passive: false }});

  // Manual overlays (server-supplied) and draw overlays (client-only until rerun).
  //
  // NOTE: These are referenced from applyInitialView(), which can run before this
  // section executes. Use `var` (function-scoped, hoisted) to avoid TDZ errors.
  var manualLayer = null;
  var tempLayer = null;
  var lastManualOverlayRaw = null;
  var suppressSvgClickUntil = 0;

  function ensureLayer(id) {{
    if (!svg) return null;
    let g = null;
    try {{ g = svg.querySelector(`#${{id}}`); }} catch (_) {{ g = null; }}
    if (g) return g;
    try {{
      g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("id", id);
      svg.appendChild(g);
      return g;
    }} catch (_) {{
      return null;
    }}
  }}

  function clearLayer(g) {{
    if (!g) return;
    try {{
      while (g.firstChild) g.removeChild(g.firstChild);
    }} catch (_) {{}}
  }}

  function drawBox(layer, bbox, stroke, strokeWidth, dashArray, opacity, titleText) {{
    if (!layer || !bbox || !Array.isArray(bbox) || bbox.length !== 4) return;
    const x0 = Math.min(bbox[0], bbox[2]);
    const y0 = Math.min(bbox[1], bbox[3]);
    const x1 = Math.max(bbox[0], bbox[2]);
    const y1 = Math.max(bbox[1], bbox[3]);
    const w = Math.max(0, x1 - x0);
    const h = Math.max(0, y1 - y0);
    if (!(w > 0) || !(h > 0)) return;

    let r = null;
    try {{
      r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", String(x0));
      r.setAttribute("y", String(y0));
      r.setAttribute("width", String(w));
      r.setAttribute("height", String(h));
      r.setAttribute("fill", "none");
      r.setAttribute("stroke", stroke || "#00ffff");
      r.setAttribute("stroke-width", String(strokeWidth || 2));
      r.setAttribute("vector-effect", "non-scaling-stroke");
      if (dashArray) r.setAttribute("stroke-dasharray", String(dashArray));
      if (Number.isFinite(opacity)) r.setAttribute("opacity", String(opacity));
      r.style.pointerEvents = "none";
      if (titleText) {{
        const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
        t.textContent = String(titleText);
        r.appendChild(t);
      }}
      layer.appendChild(r);
    }} catch (_) {{}}
  }}

  function computeIoU(a, b) {{
    if (!a || !b || a.length !== 4 || b.length !== 4) return 0;
    const ax0 = Math.min(a[0], a[2]), ay0 = Math.min(a[1], a[3]);
    const ax1 = Math.max(a[0], a[2]), ay1 = Math.max(a[1], a[3]);
    const bx0 = Math.min(b[0], b[2]), by0 = Math.min(b[1], b[3]);
    const bx1 = Math.max(b[0], b[2]), by1 = Math.max(b[1], b[3]);
    const ix0 = Math.max(ax0, bx0);
    const iy0 = Math.max(ay0, by0);
    const ix1 = Math.min(ax1, bx1);
    const iy1 = Math.min(ay1, by1);
    const iw = Math.max(0, ix1 - ix0);
    const ih = Math.max(0, iy1 - iy0);
    const inter = iw * ih;
    const aa = Math.max(0, ax1 - ax0) * Math.max(0, ay1 - ay0);
    const ba = Math.max(0, bx1 - bx0) * Math.max(0, by1 - by0);
    const denom = aa + ba - inter;
    if (!(denom > 0)) return 0;
    return inter / denom;
  }}

  function computeIntersectionArea(a, b) {{
    if (!a || !b || a.length !== 4 || b.length !== 4) return 0;
    const ax0 = Math.min(a[0], a[2]), ay0 = Math.min(a[1], a[3]);
    const ax1 = Math.max(a[0], a[2]), ay1 = Math.max(a[1], a[3]);
    const bx0 = Math.min(b[0], b[2]), by0 = Math.min(b[1], b[3]);
    const bx1 = Math.max(b[0], b[2]), by1 = Math.max(b[1], b[3]);
    const ix0 = Math.max(ax0, bx0);
    const iy0 = Math.max(ay0, by0);
    const ix1 = Math.min(ax1, bx1);
    const iy1 = Math.min(ay1, by1);
    const iw = Math.max(0, ix1 - ix0);
    const ih = Math.max(0, iy1 - iy0);
    return iw * ih;
  }}

  function bboxCenter(b) {{
    return {{ x: (b[0] + b[2]) / 2, y: (b[1] + b[3]) / 2 }};
  }}

  function snapCandidateForDraw(drawn) {{
    if (!svg) return null;
    const rects = svg.querySelectorAll("rect[data-door-id]");
    if (!rects || rects.length === 0) return null;

    const x0 = Math.min(drawn[0], drawn[2]);
    const y0 = Math.min(drawn[1], drawn[3]);
    const x1 = Math.max(drawn[0], drawn[2]);
    const y1 = Math.max(drawn[1], drawn[3]);
    const norm = [x0, y0, x1, y1];
    const dw = Math.max(1, x1 - x0);
    const dh = Math.max(1, y1 - y0);
    const dc = bboxCenter(norm);

    let best = null;
    let bestIou = -1;
    const centersInside = [];
    const centersAll = [];
    const overlap = []; // candidates with inter > 0 (i.e. actually inside selection)
    let bestCoverage = -1;
    let bestByCoverage = null;
    let bestInter = -1;
    let bestByInter = null;

    for (const r of rects) {{
      const did = r.getAttribute("data-door-id");
      if (!did) continue;
      // Use SVG geometry instead of relying on attributes (which can be missing/NaN).
      let bb = null;
      try {{ bb = r.getBBox(); }} catch (_) {{ bb = null; }}
      if (!bb) continue;
      const rw = Number(bb.width);
      const rh = Number(bb.height);
      const rx = Number(bb.x);
      const ry = Number(bb.y);
      if (!(rw > 0) || !(rh > 0) || !Number.isFinite(rx) || !Number.isFinite(ry)) continue;
      const cand = [rx, ry, rx + rw, ry + rh];
      const iou = computeIoU(norm, cand);
      const inter = computeIntersectionArea(norm, cand);
      const candArea = Math.max(0, (cand[2] - cand[0])) * Math.max(0, (cand[3] - cand[1]));
      const coverage = candArea > 0 ? (inter / candArea) : 0; // how much of candidate is inside drawn
      // Only consider candidates that overlap the selection for snapping.
      // This prevents snapping to a nearby candidate outside the drawn box.
      if (inter > 0) {{
        overlap.push({{ id: did, bbox: cand, iou: iou, inter: inter, coverage: coverage }});
        if (iou > bestIou) {{
          bestIou = iou;
          best = {{ id: did, bbox: cand, iou: iou }};
        }}
        if (coverage > bestCoverage) {{
          bestCoverage = coverage;
          bestByCoverage = {{ id: did, bbox: cand, iou: iou, inter: inter, coverage: coverage }};
        }}
        if (inter > bestInter) {{
          bestInter = inter;
          bestByInter = {{ id: did, bbox: cand, iou: iou, inter: inter, coverage: coverage }};
        }}
      }}
      const cc = bboxCenter(cand);
      const dist = Math.hypot(cc.x - dc.x, cc.y - dc.y);
      const rec = {{ id: did, bbox: cand, iou: iou, dist: dist }};
      centersAll.push(rec);
      if (cc.x >= x0 && cc.x <= x1 && cc.y >= y0 && cc.y <= y1) centersInside.push(rec);
    }}

    // Snapping only considers candidates that overlap the selection.
    const MIN_SNAP_IOU = 0.02; // selector rectangles can be larger than candidates
    const MIN_CAND_COVERAGE = 0.20;
    try {{
      const closest = centersAll.length ? centersAll.slice().sort((a, b) => (a.dist - b.dist) || (b.iou - a.iou))[0] : null;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      let sample = [];
      for (let i = 0; i < Math.min(3, overlap.length); i++) {{
        const o = overlap[i];
        sample.push({{ id: o.id, bbox: o.bbox, iou: o.iou, inter: o.inter, coverage: o.coverage }});
      }}
      // Compute candidate coordinate range (using bbox geometry).
      for (const rr of rects) {{
        let bb = null;
        try {{ bb = rr.getBBox(); }} catch (_) {{ bb = null; }}
        if (!bb) continue;
        const rx0 = Number(bb.x), ry0 = Number(bb.y);
        const rx1 = rx0 + Number(bb.width), ry1 = ry0 + Number(bb.height);
        if (!Number.isFinite(rx0) || !Number.isFinite(ry0) || !Number.isFinite(rx1) || !Number.isFinite(ry1)) continue;
        minX = Math.min(minX, rx0);
        minY = Math.min(minY, ry0);
        maxX = Math.max(maxX, rx1);
        maxY = Math.max(maxY, ry1);
      }}

      // Also compute the nearest candidate to the drawn box (even if no overlap).
      let nearest = null;
      let bestDist = Infinity;
      for (const rr of rects) {{
        const did = rr.getAttribute("data-door-id");
        if (!did) continue;
        let bb = null;
        try {{ bb = rr.getBBox(); }} catch (_) {{ bb = null; }}
        if (!bb) continue;
        const rx0 = Number(bb.x), ry0 = Number(bb.y);
        const rx1 = rx0 + Number(bb.width), ry1 = ry0 + Number(bb.height);
        if (!Number.isFinite(rx0) || !Number.isFinite(ry0) || !Number.isFinite(rx1) || !Number.isFinite(ry1)) continue;
        const cand = [rx0, ry0, rx1, ry1];
        const cc = bboxCenter(cand);
        const dist = Math.hypot(cc.x - dc.x, cc.y - dc.y);
        if (dist < bestDist) {{
          bestDist = dist;
          nearest = {{ id: did, bbox: cand, dist }};
        }}
      }}

      // Compute max intersection among all candidates (diagnostic).
      let maxInterAny = 0;
      let maxInterAnyId = null;
      let maxInterAnyBBox = null;
      for (const rr of rects) {{
        const did = rr.getAttribute("data-door-id");
        if (!did) continue;
        let bb = null;
        try {{ bb = rr.getBBox(); }} catch (_) {{ bb = null; }}
        if (!bb) continue;
        const rx0 = Number(bb.x), ry0 = Number(bb.y);
        const rx1 = rx0 + Number(bb.width), ry1 = ry0 + Number(bb.height);
        if (!Number.isFinite(rx0) || !Number.isFinite(ry0) || !Number.isFinite(rx1) || !Number.isFinite(ry1)) continue;
        const cand = [rx0, ry0, rx1, ry1];
        const inter = computeIntersectionArea(norm, cand);
        if (inter > maxInterAny) {{
          maxInterAny = inter;
          maxInterAnyId = did;
          maxInterAnyBBox = cand;
        }}
      }}

      console.log("[door_detector] snapCandidateForDraw", {{
        drawn: norm,
        drawnWH: [dw, dh],
        candRange: (minX !== Infinity) ? [minX, minY, maxX, maxY] : null,
        candidates: rects.length,
        overlapCandidates: overlap.length,
        bestByIoU: best ? {{ id: best.id, iou: best.iou }} : null,
        bestByCoverage: bestByCoverage ? {{ id: bestByCoverage.id, coverage: bestByCoverage.coverage, iou: bestByCoverage.iou, inter: bestByCoverage.inter }} : null,
        bestByInter: bestByInter ? {{ id: bestByInter.id, inter: bestByInter.inter, coverage: bestByInter.coverage, iou: bestByInter.iou }} : null,
        centersInside: centersInside.length,
        closest: closest ? {{ id: closest.id, dist: closest.dist, iou: closest.iou }} : null,
        thresholds: {{ MIN_SNAP_IOU, MIN_CAND_COVERAGE }},
        overlapSample: sample,
        nearestCandidate: nearest,
        maxInterAny: {{ id: maxInterAnyId, inter: maxInterAny, bbox: maxInterAnyBBox }},
      }});
      console.log(
        "[door_detector] snapCandidateForDraw json",
        JSON.stringify(
          {{
            drawn: norm,
            drawnWH: [dw, dh],
            candRange: (minX !== Infinity) ? [minX, minY, maxX, maxY] : null,
            candidates: rects.length,
            overlapCandidates: overlap.length,
            bestByIoU: best ? {{ id: best.id, iou: best.iou }} : null,
            bestByCoverage: bestByCoverage ? {{ id: bestByCoverage.id, coverage: bestByCoverage.coverage, iou: bestByCoverage.iou, inter: bestByCoverage.inter }} : null,
            bestByInter: bestByInter ? {{ id: bestByInter.id, inter: bestByInter.inter, coverage: bestByInter.coverage, iou: bestByInter.iou }} : null,
            closest: closest ? {{ id: closest.id, dist: closest.dist, iou: closest.iou }} : null,
            thresholds: {{ MIN_SNAP_IOU, MIN_CAND_COVERAGE }},
            nearestCandidate: nearest,
            maxInterAny: {{ id: maxInterAnyId, inter: maxInterAny, bbox: maxInterAnyBBox }},
          }}
        )
      );
    }} catch (_) {{}}

    if (!overlap.length) {{
      try {{ console.log("[door_detector] snapCandidateForDraw no match (no overlap)"); }} catch (_) {{}}
      return null;
    }}

    if (best && best.iou >= MIN_SNAP_IOU) {{
      try {{ console.log("[door_detector] snapCandidateForDraw chosen", {{ reason: "iou", id: best.id, iou: best.iou }}); }} catch (_) {{}}
      return best;
    }}

    // If IoU is too low, fall back to maximum overlap (intersection area).
    if (bestByInter) {{
      try {{ console.log("[door_detector] snapCandidateForDraw chosen", {{ reason: "max_intersection", id: bestByInter.id, inter: bestByInter.inter, iou: bestByInter.iou }}); }} catch (_) {{}}
      return bestByInter;
    }}

    // Fallback: if the drawn rectangle covers a substantial fraction of a candidate,
    // treat it as a match even when IoU is low.
    if (bestByCoverage && bestByCoverage.coverage >= MIN_CAND_COVERAGE) {{
      try {{ console.log("[door_detector] snapCandidateForDraw chosen", {{ reason: "coverage", id: bestByCoverage.id, coverage: bestByCoverage.coverage, iou: bestByCoverage.iou }}); }} catch (_) {{}}
      return bestByCoverage;
    }}

    try {{ console.log("[door_detector] snapCandidateForDraw no match (overlap too weak)", {{ bestByIoU: best ? {{ id: best.id, iou: best.iou }} : null, bestByInter: bestByInter ? {{ id: bestByInter.id, inter: bestByInter.inter }} : null }}); }} catch (_) {{}}
    return null;
  }}

  function renderManualOverlays() {{
    const raw = readParentInputValue(manualOverlaySinkLabel);
    if (raw === lastManualOverlayRaw) return false;
    lastManualOverlayRaw = raw;
    try {{ console.log("[door_detector] renderManualOverlays update", {{ bytes: raw ? raw.length : 0 }}); }} catch (_) {{}}

    const obj = readParentJson(manualOverlaySinkLabel) || {{}};
    const manual = Array.isArray(obj.manual_additions) ? obj.manual_additions : [];
    const unmatched = Array.isArray(obj.unmatched_manual_boxes) ? obj.unmatched_manual_boxes : [];

    manualLayer = ensureLayer("pz_manual");
    tempLayer = ensureLayer("pz_temp");
    clearLayer(manualLayer);
    // Once server overlays update, drop any client-only temp boxes to avoid duplicates.
    clearLayer(tempLayer);
    try {{ console.log("[door_detector] renderManualOverlays cleared temp layer"); }} catch (_) {{}}

    for (const m of manual) {{
      const drawn = m.drawn_bbox_xyxy;
      const snapped = m.snapped_bbox_xyxy;
      const iou = m.iou;
      const cid = m.snapped_candidate_id;
      drawBox(
        manualLayer,
        drawn,
        "rgba(0,255,255,0.85)",
        2,
        "6,4",
        0.55,
        cid ? `drawn (→ ${{cid}}, iou=${{iou}})` : "drawn"
      );
      if (snapped && Array.isArray(snapped) && snapped.length === 4) {{
        drawBox(
          manualLayer,
          snapped,
          "rgba(0,255,0,0.9)",
          3,
          "4,3",
          0.85,
          cid ? `snapped (${{cid}}, iou=${{iou}})` : "snapped"
        );
      }}
    }}

    for (const u of unmatched) {{
      const bbox = u.bbox_xyxy;
      const note = u.note || "unmatched";
      drawBox(manualLayer, bbox, "rgba(255,0,255,0.9)", 2, "6,4", 0.7, String(note));
    }}
    return true;
  }}

  function clientToImage(clientX, clientY) {{
    // Robust mapping based on the transformed content element’s DOM rect.
    // This avoids subtle drift/mismatch if tx/ty/scale bookkeeping ever diverges.
    try {{
      const cr = content.getBoundingClientRect();
      if (!(cr.width > 0) || !(cr.height > 0)) throw new Error("bad content rect");
      const iw = {int(img_width)};
      const ih = {int(img_height)};
      const qx = (clientX - cr.left) * (iw / cr.width);
      const qy = (clientY - cr.top) * (ih / cr.height);
      clientToImage._lastMethod = "domRect";
      return {{ x: qx, y: qy }};
    }} catch (_) {{
      // Fallback to internal transform math.
      const rect = root.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      const qx = (px - tx) / scale;
      const qy = (py - ty) / scale;
      clientToImage._lastMethod = "txScale";
      return {{ x: qx, y: qy }};
    }}
  }}

  let drawing = false;
  let drawStart = null;
  let drawRect = null;

  root.addEventListener("pointerdown", (e) => {{
    if (resetBtn && resetBtn.contains(e.target)) return;
    if (e.button !== 0) return;
    // Shift+drag draws a selector rectangle in Edit Doors mode.
    if (getEditEnabled() && e.shiftKey) {{
      drawing = true;
      stage.style.cursor = "crosshair";
      drawStart = clientToImage(e.clientX, e.clientY);
      tempLayer = ensureLayer("pz_temp");
      if (tempLayer) {{
        try {{
          drawRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          drawRect.setAttribute("fill", "rgba(0,255,255,0.12)");
          drawRect.setAttribute("stroke", "rgba(0,255,255,0.9)");
          drawRect.setAttribute("stroke-width", "2");
          drawRect.setAttribute("stroke-dasharray", "6,4");
          drawRect.setAttribute("vector-effect", "non-scaling-stroke");
          drawRect.style.pointerEvents = "none";
          tempLayer.appendChild(drawRect);
        }} catch (_) {{}}
      }}
      suppressSvgClickUntil = performance.now() + 250;
      e.preventDefault();
      e.stopPropagation();
      try {{ root.setPointerCapture(e.pointerId); }} catch (_) {{}}
      return;
    }}

    // Default: pan drag.
    dragging = true;
    stage.style.cursor = "grabbing";
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragStartTx = tx;
    dragStartTy = ty;
    try {{ root.setPointerCapture(e.pointerId); }} catch (_) {{}}
  }});

  root.addEventListener("pointermove", (e) => {{
    if (drawing && drawStart) {{
      const p = clientToImage(e.clientX, e.clientY);
      const x0 = Math.min(drawStart.x, p.x);
      const y0 = Math.min(drawStart.y, p.y);
      const x1 = Math.max(drawStart.x, p.x);
      const y1 = Math.max(drawStart.y, p.y);
      const w = Math.max(0, x1 - x0);
      const h = Math.max(0, y1 - y0);
      if (drawRect) {{
        try {{
          drawRect.setAttribute("x", String(x0));
          drawRect.setAttribute("y", String(y0));
          drawRect.setAttribute("width", String(w));
          drawRect.setAttribute("height", String(h));
        }} catch (_) {{}}
      }}
      e.preventDefault();
      return;
    }}
    if (!dragging) return;
    tx = dragStartTx + (e.clientX - dragStartX);
    ty = dragStartTy + (e.clientY - dragStartY);
    applyTransform();
  }});

  function endDrag(e) {{
    if (drawing) {{
      drawing = false;
      stage.style.cursor = "grab";
      const end = clientToImage(e.clientX, e.clientY);
      const x0 = Math.min(drawStart?.x ?? end.x, end.x);
      const y0 = Math.min(drawStart?.y ?? end.y, end.y);
      const x1 = Math.max(drawStart?.x ?? end.x, end.x);
      const y1 = Math.max(drawStart?.y ?? end.y, end.y);
      drawStart = null;
      // Keep the drawn rect faintly visible until the rerun overlays arrive.
      if (drawRect) {{
        try {{
          drawRect.setAttribute("fill", "rgba(0,255,255,0.08)");
          drawRect.setAttribute("stroke", "rgba(0,255,255,0.65)");
        }} catch (_) {{}}
      }}
      drawRect = null;

      // Emit draw event to Streamlit.
      const w = Math.abs(x1 - x0);
      const h = Math.abs(y1 - y0);
      if (w >= 2 && h >= 2) {{
        const drawn = [x0, y0, x1, y1];
        const snap = snapCandidateForDraw(drawn);
        try {{
          const n = svg ? svg.querySelectorAll("rect[data-door-id]").length : 0;
          const rr = root ? root.getBoundingClientRect() : null;
          const cr = content ? content.getBoundingClientRect() : null;
          const used = clientToImage._lastMethod || "unknown";
          // Compare the two mapping methods at the end point (diagnostic).
          const rect = rr || {{ left: 0, top: 0 }};
          const px = e.clientX - rect.left;
          const py = e.clientY - rect.top;
          const qx_math = (px - tx) / scale;
          const qy_math = (py - ty) / scale;
          const qx_dom = cr && cr.width ? ((e.clientX - cr.left) * ({int(img_width)} / cr.width)) : null;
          const qy_dom = cr && cr.height ? ((e.clientY - cr.top) * ({int(img_height)} / cr.height)) : null;
          console.log("[door_detector] draw_rect endDrag", {{
            drawn,
            drawnRounded: drawn.map((v) => Math.round(v)),
            drawnJson: JSON.stringify(drawn),
            candidates: n,
            snap: snap ? {{ id: snap.id, iou: snap.iou, bbox: snap.bbox }} : null,
            editEnabled: getEditEnabled(),
            viewMode: getViewerDisplayMode(),
            tx, ty, scale,
            rootRect: rr ? {{ left: rr.left, top: rr.top, w: rr.width, h: rr.height }} : null,
            contentRect: cr ? {{ left: cr.left, top: cr.top, w: cr.width, h: cr.height }} : null,
            mapCompare: {{ used, math: [qx_math, qy_math], dom: [qx_dom, qy_dom] }},
          }});
        }} catch (_) {{}}
        if (snap && tempLayer) {{
          try {{
            drawBox(
              tempLayer,
              snap.bbox,
              "rgba(0,255,0,0.9)",
              3,
              "4,3",
              0.85,
              `snapped (${{snap.id}}, iou=${{snap.iou}})`
            );
            try {{ console.log("[door_detector] draw_rect drew snap overlay", {{ id: snap.id }}); }} catch (_) {{}}
          }} catch (_) {{}}
        }} else {{
          try {{ console.log("[door_detector] draw_rect no snap overlay drawn", {{ hasSnap: !!snap, hasTempLayer: !!tempLayer }}); }} catch (_) {{}}
          // Show an explicit unmatched marker immediately so the user isn't left
          // wondering if anything happened.
          if (tempLayer) {{
            try {{
              drawBox(
                tempLayer,
                drawn,
                "rgba(255,0,255,0.9)",
                2,
                "6,4",
                0.75,
                "unmatched (no overlapping candidate)"
              );
              console.log("[door_detector] draw_rect drew unmatched overlay");
            }} catch (_) {{}}
          }}
        }}
        const payload = {{
          event: "draw_rect",
          event_id: `${{Date.now()}}_${{Math.random().toString(16).slice(2)}}`,
          bbox_xyxy: drawn,
          snapped_candidate_id: snap ? snap.id : null,
          iou: snap ? snap.iou : null,
          snapped_bbox_xyxy: snap ? snap.bbox : null,
          ts: Date.now(),
        }};
        try {{
          const ok = setParentInputValue(drawEventSinkLabel, JSON.stringify(payload));
          // Some Streamlit builds don't rerun reliably on a hidden text_input change.
          // As a robust fallback, also poke the existing click sink with a sentinel
          // value that will not match any door id, solely to trigger the rerun.
          if (ok) {{
            setParentInputValue(clickSinkLabel, `__draw_event__${{payload.event_id}}`);
          }}
        }} catch (_) {{}}
      }}
      return;
    }}

    if (!dragging) return;
    dragging = false;
    stage.style.cursor = "grab";
  }}

  root.addEventListener("pointerup", endDrag);
  root.addEventListener("pointercancel", endDrag);
  root.addEventListener("pointerleave", endDrag);

  function setSelectedDoorId(doorId) {{
    if (!doorId) return;
    const ok = setParentInputValue(clickSinkLabel, String(doorId));
    try {{
      console.log("[door_detector] click->sink", {{
        doorId,
        ok,
        clickSinkLabel,
        parentValueAfter: readParentInputValue(clickSinkLabel),
        lastSinkWrite: window.__door_detectorLastSinkWrite || null,
        ts: Date.now(),
      }});
    }} catch (_) {{}}
    // Diagnostic: if Streamlit ingests the sink update, it should rerun and either:
    // - clear the click sink (we consume it server-side), and/or
    // - update selectedSinkLabel / focusSeqSinkLabel.
    try {{
      const t0 = Date.now();
      setTimeout(() => {{
        try {{
          console.log("[door_detector] click->sink followup", {{
            doorId,
            dtMs: Date.now() - t0,
            clickSinkNow: readParentInputValue(clickSinkLabel),
            selectedSinkNow: readParentInputValue(selectedSinkLabel),
            focusSeqNow: readParentInputValue(focusSeqSinkLabel),
            ts: Date.now(),
          }});
        }} catch (_) {{}}
      }}, 450);
    }} catch (_) {{}}
  }}

  // Watch selection changes coming from Streamlit (right panel).
  let lastSelectedId = null;
  let lastFocusSeq = null;
  function pollSelection() {{
    const did = getSelectedId();
    const seq = getFocusSeq();
    const viewMode = getViewerDisplayMode();
    const editEnabled = getEditEnabled();
    const doorStateChanged = updateDoorStateFromSinks();
    renderManualOverlays();

    const newAutoFocus = getAutoFocus();
    if (newAutoFocus !== autoFocus) {{
      autoFocus = newAutoFocus;
      // If the user just enabled auto-focus, focus immediately to the current selection.
      if (autoFocus && did) {{
        try {{ focusToDoorId(did); }} catch (_) {{}}
        lastFocusSeq = seq;
      }}
    }}

    let needsStyle = false;
    if (did !== lastSelectedId) {{
      lastSelectedId = did;
      localSelectedId = null;
      needsStyle = true;
    }}
    if (viewMode !== lastViewerDisplay) {{
      lastViewerDisplay = viewMode;
      needsStyle = true;
    }}
    if (editEnabled !== lastEditEnabled) {{
      lastEditEnabled = editEnabled;
      needsStyle = true;
    }}
    if (doorStateChanged) needsStyle = true;
    if (needsStyle) applyDoorStyles();

    if (autoFocus && did) {{
      if (lastFocusSeq === null) {{
        lastFocusSeq = seq;
      }} else if (seq !== lastFocusSeq) {{
        lastFocusSeq = seq;
        focusToDoorId(did);
      }}
    }}
  }}
  try {{ setInterval(pollSelection, 120); }} catch (_) {{}}

  if (svg) {{
    // If a door bbox is clicked, select it (and don't start a pan drag).
    svg.addEventListener("pointerdown", (e) => {{
      // In Edit Doors mode, Shift+drag should start drawing even if you start on a door box.
      if (getEditEnabled() && e.shiftKey) return;
      const t = e.target;
      if (t && t.getAttribute && t.getAttribute("data-door-id")) {{
        e.preventDefault();
        e.stopPropagation();
      }}
    }}, true);

    svg.addEventListener("click", (e) => {{
      if (performance.now && performance.now() < suppressSvgClickUntil) return;
      const t = e.target;
      if (!t || !t.getAttribute) return;
      const did = t.getAttribute("data-door-id");
      if (!did) return;
      e.preventDefault();
      e.stopPropagation();
      // Immediate UX: highlight/focus locally without waiting for the rerun.
      localSelectedId = did;
      applyDoorStyles();
      if (getAutoFocus()) focusToDoorId(did);
      setSelectedDoorId(did);
    }});
  }}
}})();
</script>
"""
    try:
        viewer_hash = hashlib.sha256(viewer_html.encode("utf-8")).hexdigest()[:12]
        _debug_log("viewer key=%s html_hash=%s", str(key), viewer_hash)
    except Exception:
        pass
    components.html(viewer_html, height=height, scrolling=False)

def sidebar_library():
    st.sidebar.title("Library")

    # Server-console debug logs (selection sync, timing, etc.)
    try:
        st.sidebar.checkbox("Debug logs", key="debug_perf", help="Log selection sync + perf to the server console.")
    except Exception:
        pass

    # Manage actions removed (Import existing artifacts / Clear library).
    # If a prior run left the confirm flag around, clear it so it doesn't linger.
    try:
        st.session_state.pop("confirm_clear_library", None)
    except Exception:
        pass

    # Search and Add Area
    if not st.session_state.search_visible:
        col_search, col_add = st.sidebar.columns(2)
        with col_search:
            if st.button("Search", key="open_search_btn", help="Open search", use_container_width=True):
                st.session_state.search_visible = True
                st.rerun()
        with col_add:
            upload_key = f"upload_pdf_{int(st.session_state.get('upload_widget_seq') or 0)}"
            uploaded_file = st.file_uploader(
                "Upload",
                type=["pdf"],
                label_visibility="collapsed",
                key=upload_key,
            )
            if uploaded_file:
                file_id = lib.add_file(uploaded_file.name, uploaded_file.getvalue())
                # Auto-select the newly added file, and reset the uploader so we
                # don't re-add it on the next rerun.
                st.session_state.selected_file_id = file_id
                st.session_state.upload_widget_seq = int(st.session_state.get("upload_widget_seq") or 0) + 1
                st.rerun()
    else:
        col_input, col_close = st.sidebar.columns([5, 1])
        with col_input:
            search_val = st.text_input(
                "Search",
                value=st.session_state.search_query,
                label_visibility="collapsed",
                key="search_input_widget",
            )
            if search_val != st.session_state.search_query:
                st.session_state.search_query = search_val
                st.rerun()
        with col_close:
            if st.button("X", key="close_search_btn", help="Clear search"):
                st.session_state.search_query = ""
                st.session_state.search_visible = False
                st.rerun()

    st.sidebar.divider()

    items = lib.get_items()
    if st.session_state.search_query:
        items = [i for i in items if st.session_state.search_query.lower() in i["original_name"].lower()]

    if not items:
        st.sidebar.info("No files in library.")
    else:
        for item in items:
            is_selected = (st.session_state.get("selected_file_id") == item["id"])
            label = item["original_name"]

            if st.sidebar.button(
                label,
                key=f"sel_{item['id']}",
                help=item["original_name"],
                type="primary" if is_selected else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected_file_id = item["id"]
                st.rerun()

def main_viewer_canvas(
    item: Dict,
    *,
    preview_spec: Optional[Dict[str, Any]],
    full_dims: Optional[Tuple[int, int]],
    doors_data: Dict,
    fstate: Dict,
    active_doors: List[Dict[str, Any]],
    click_sink_label: str,
):
    file_id = item["id"]
    file_dir = Path(item["path"])
    
    if preview_spec:
        # Door click sink: JS writes the clicked door id into this hidden widget,
        # which triggers a rerun; on rerun we sync it into the "Jump to door" control.
        click_sink_key = click_sink_label
        st.text_input(click_sink_label, key=click_sink_key, label_visibility="collapsed")

        # Auto-focus sink: viewer polls this so toggling the checkbox doesn't remount the iframe.
        auto_focus_key = f"auto_focus_{file_id}"
        auto_focus_sink_label = f"auto_focus_sink_{file_id}"
        try:
            # Canonicalize early (main_viewer_controls may render after the viewer).
            auto_focus_val = _coerce_bool(
                st.session_state.get(auto_focus_key),
                default=_coerce_bool(fstate.get("auto_focus", True), default=True),
            )
            fstate["auto_focus"] = auto_focus_val
            st.session_state[auto_focus_sink_label] = "1" if auto_focus_val else "0"
        except Exception:
            pass
        st.text_input(auto_focus_sink_label, key=auto_focus_sink_label, label_visibility="collapsed")

        # Selection/focus sinks: the viewer polls these so selection changes don't require
        # changing the viewer HTML (reduces iframe reload flicker).
        selected_sink_label = f"selected_door_sink_{file_id}"
        focus_seq_sink_label = f"focus_seq_sink_{file_id}"
        try:
            st.session_state[selected_sink_label] = str(fstate.get("selected_door_id") or "")
            st.session_state[focus_seq_sink_label] = str(int(fstate.get("_focus_seq") or 0))
        except Exception:
            pass
        st.text_input(selected_sink_label, key=selected_sink_label, label_visibility="collapsed")
        st.text_input(focus_seq_sink_label, key=focus_seq_sink_label, label_visibility="collapsed")

        # Edit-mode + overlay sinks: the viewer polls these so we can enable Shift+drag
        # drawing and update styles/overlays without remounting the iframe.
        edit_mode_sink_label = f"edit_mode_sink_{file_id}"
        draw_event_sink_label = f"draw_event_sink_{file_id}"
        manual_overlay_sink_label = f"manual_overlay_sink_{file_id}"
        door_state_sink_label = f"door_state_sink_{file_id}"
        viewer_display_sink_label = f"viewer_display_sink_{file_id}"

        # Server → iframe values (updated every run).
        try:
            working = _get_working_label_state(fstate)
            st.session_state[edit_mode_sink_label] = "1" if bool(fstate.get("edit_mode")) else "0"
            st.session_state[viewer_display_sink_label] = _viewer_display_mode_to_sink_value(
                str(fstate.get("viewer_display_mode") or "Highlight All")
            )
            st.session_state[door_state_sink_label] = json.dumps(
                {
                    "confirmed_ids": sorted(list(working.get("confirmed_ids", set()))),
                    "deleted_ids": sorted(list(working.get("deleted_ids", set()))),
                },
                separators=(",", ":"),
            )
            # Manual overlay data is supplied via a separate sink (preview-space bboxes).
            st.session_state[manual_overlay_sink_label] = json.dumps(
                _manual_overlay_payload_for_sink(
                    fstate=fstate,
                    preview_scale=float(preview_spec.get("scale", 1.0) or 1.0),
                ),
                separators=(",", ":"),
            )
        except Exception:
            pass

        # Iframe → server events (do not overwrite once set by JS).
        if draw_event_sink_label not in st.session_state:
            st.session_state[draw_event_sink_label] = ""

        st.text_input(edit_mode_sink_label, key=edit_mode_sink_label, label_visibility="collapsed")
        st.text_input(draw_event_sink_label, key=draw_event_sink_label, label_visibility="collapsed")
        st.text_input(manual_overlay_sink_label, key=manual_overlay_sink_label, label_visibility="collapsed")
        st.text_input(door_state_sink_label, key=door_state_sink_label, label_visibility="collapsed")
        st.text_input(viewer_display_sink_label, key=viewer_display_sink_label, label_visibility="collapsed")

        viewer_width_hint = int(VIEWER_TARGET_WIDTH_PX)
        viewer_width_hint = max(600, min(2000, viewer_width_hint))
        aspect = float(VIEWER_ASPECT_RATIO_HW)
        aspect = max(0.35, min(1.25, aspect))

        # Viewer height derived from width and a fixed aspect ratio.
        viewer_height = int(round(viewer_width_hint * aspect))
        viewer_height = max(450, min(1400, viewer_height))

        # NOTE: Don't wrap this in `st.container(height=...)` because Streamlit makes that
        # container scrollable (adds a scrollbar) which steals scroll/drag interactions.
        rects_svg = _rects_to_svg(
            active_doors=active_doors,
            fstate=fstate,
            scale=float(preview_spec.get("scale", 1.0)),
            img_width=int(preview_spec.get("width", 1)),
            img_height=int(preview_spec.get("height", 1)),
        )

        img_src = _image_path_to_streamlit_url(str(preview_spec.get("path", "")))

        _panzoom_image_viewer(
            img_src=img_src,
            img_width=int(preview_spec.get("width", 1)),
            img_height=int(preview_spec.get("height", 1)),
            rects_svg=rects_svg,
            height=viewer_height,
            key=str(file_id),
            click_sink_aria_label=click_sink_label,
            selected_sink_aria_label=selected_sink_label,
            focus_seq_sink_aria_label=focus_seq_sink_label,
            edit_mode_sink_aria_label=edit_mode_sink_label,
            draw_event_sink_aria_label=draw_event_sink_label,
            manual_overlay_sink_aria_label=manual_overlay_sink_label,
            door_state_sink_aria_label=door_state_sink_label,
            viewer_display_sink_aria_label=viewer_display_sink_label,
            auto_focus_sink_aria_label=auto_focus_sink_label,
        )
        return None, active_doors
    else:
        # Still run the sidebar auto-open logic even when the viewer isn't mounted.
        components.html(_sidebar_autopen_component_html(), height=0, scrolling=False)
        st.info("Analyze to see results.")
        return None, []

def main_viewer_controls(
    item: Dict,
    *,
    full_dims: Optional[Tuple[int, int]],
    doors_data: Dict,
    fstate: Dict,
):
    file_id = item["id"]
    file_dir = Path(item["path"])
    # Some discovered file ids can include characters like '(' which are not valid
    # HTML element ids; Streamlit may then omit the button id attribute. Use a
    # safe hashed suffix for widget keys that we want to style via CSS.
    key_suffix = hashlib.md5(str(file_id).encode("utf-8")).hexdigest()[:12]
    
    # Grid for main controls
    c1, c2, c_del = st.columns([2, 2, 1])
    with c1:
        status = item.get("status", "not_processed")
        if status == "processing":
            # Streamlit runs the pipeline synchronously; the UI won't render mid-run.
            # So a persisted "processing" status is stale and should not block running.
            doors_path = file_dir / "doors.json"
            if doors_path.exists():
                lib.update_status(file_id, "done")
                status = "done"
            else:
                lib.update_status(file_id, "not_processed")
                status = "not_processed"
            st.cache_data.clear()
            st.rerun()

        config_path = "configs/door_rules.json"
        current_sig = get_current_signature(config_path)
        stored_sig = doors_data.get("analysis_signature")
        is_out_of_date = stored_sig and current_sig and stored_sig != current_sig
        
        label = "Re-analyze" if status == "done" else "Analyze"
        if is_out_of_date:
            label = f"{label} (!)"

        task = st.session_state.get("door_detector_pipeline_task")
        is_running_for_file = bool(task and task.get("file_id") == str(file_id))
        analysis_label = f"Analyzing {item.get('original_name', '')}".strip() or "Analyzing…"

        st.button(
            label,
            type="primary" if not status == "done" else "secondary",
            use_container_width=True,
            disabled=is_running_for_file,
            on_click=_queue_pipeline_run,
            args=(str(file_id), str(file_dir), str(config_path), analysis_label),
        )
    with c2:
        modes = ["Highlight All", "Highlight Selected", "Off"]
        fstate["viewer_display_mode"] = st.selectbox(
            "Mode", 
            modes,
            index=modes.index(fstate.get("viewer_display_mode")) if fstate.get("viewer_display_mode") in modes else 0,
            label_visibility="collapsed"
        )
    with c_del:
        confirm_key = f"confirm_delete_{file_id}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False

        if not st.session_state[confirm_key]:
            if st.button(
                "Delete",
                key=f"delete_btn_{key_suffix}",
                help="Remove this PDF from the library (deletes source.pdf and all artifacts).",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state[confirm_key] = True
                st.rerun()
        else:
            # Stack buttons vertically to avoid extreme wrapping in narrow layouts.
            if st.button(
                "Confirm",
                key=f"delete_confirm_btn_{key_suffix}",
                use_container_width=True,
                type="primary",
            ):
                _delete_library_item_and_reset_ui(file_id)
                st.rerun()
            if st.button(
                "Cancel",
                key=f"delete_cancel_btn_{key_suffix}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state[confirm_key] = False
                st.rerun()

    c3, c4 = st.columns(2)
    with c3:
        if not bool(fstate.get("edit_mode")):
            if st.button("Edit Doors", use_container_width=True, type="secondary"):
                _enter_edit_mode(fstate)
                st.rerun()
        else:
            col_save, col_cancel = st.columns(2)
            if col_save.button("Save", use_container_width=True, type="primary"):
                _save_edit_mode(fstate)
                save_current_labels(file_id, file_dir)
                st.rerun()
            if col_cancel.button("Cancel", use_container_width=True, type="secondary"):
                _cancel_edit_mode(fstate)
                st.rerun()
            st.caption("Shift+drag to add rectangles (snap-to-candidate).")
    with c4:
        auto_focus_key = f"auto_focus_{file_id}"
        # Keep widget state and per-file fstate in sync.
        if auto_focus_key not in st.session_state:
            st.session_state[auto_focus_key] = bool(fstate.get("auto_focus", True))
        fstate["auto_focus"] = st.checkbox("Auto-focus", key=auto_focus_key)

def _sync_selected_door_for_run(
    *,
    file_id: str,
    fstate: Dict[str, Any],
    all_visible: List[Dict[str, Any]],
) -> None:
    """Sync selected door state BEFORE rendering the main viewer.

    This reduces visible flicker on reruns because the viewer can render early
    while still reflecting the newest selection (Next/Prev/selectbox/box-click).
    """
    if not all_visible:
        fstate["selected_door_id"] = None
        return

    door_ids = [d.get("id") for d in all_visible if d.get("id") is not None]
    if not door_ids:
        fstate["selected_door_id"] = None
        return

    jump_key = f"jump_{file_id}"
    idx_key = f"{jump_key}__idx"
    click_sink_key = f"door_click_sink_{file_id}"
    _debug_log(
        "sync_select file_id=%s start click_raw=%r jump_raw=%r idx_raw=%r fstate_sel=%r n_visible=%d",
        str(file_id),
        st.session_state.get(click_sink_key),
        st.session_state.get(jump_key),
        st.session_state.get(idx_key),
        fstate.get("selected_door_id"),
        len(door_ids),
    )

    def _coerce_idx(v: Any) -> Optional[int]:
        try:
            # st.number_input is usually int, but can be float depending on args.
            n = int(v)
        except Exception:
            return None
        return n

    # Establish current selection.
    # Priority: clicked bbox → explicit index → previous selection → first.
    current_id: Optional[Any] = None
    clicked_id = st.session_state.get(click_sink_key)
    # Consume internal sentinel events (used to force reruns from the viewer).
    if isinstance(clicked_id, str) and clicked_id.startswith("__draw_event__"):
        try:
            st.session_state[click_sink_key] = ""
        except Exception:
            pass
        clicked_id = ""
    if clicked_id in door_ids:
        # Treat the click sink as an event: consume once, then clear so it doesn't
        # keep overriding other controls on subsequent runs.
        fstate["_last_clicked_door_id"] = clicked_id
        current_id = clicked_id
        _debug_log("sync_select file_id=%s click_match clicked_id=%r", str(file_id), clicked_id)
        try:
            st.session_state[click_sink_key] = ""
        except Exception:
            pass
    else:
        if clicked_id not in (None, ""):
            _debug_log("sync_select file_id=%s click_ignored clicked_id=%r", str(file_id), clicked_id)
        # If the user typed an index, treat it as the requested selection.
        idx_req = _coerce_idx(st.session_state.get(idx_key))
        if isinstance(idx_req, int):
            # Wrap-around navigation:
            # - minus at 1 -> 0 -> last
            # - plus at last -> N+1 -> first
            if idx_req <= 0:
                current_id = door_ids[-1]
            elif idx_req > len(door_ids):
                current_id = door_ids[0]
            else:
                current_id = door_ids[idx_req - 1]
            _debug_log("sync_select file_id=%s idx_select idx=%r id=%r", str(file_id), idx_req, current_id)
        elif st.session_state.get(jump_key) in door_ids:
            current_id = st.session_state[jump_key]
            _debug_log("sync_select file_id=%s jump_select id=%r", str(file_id), current_id)
        elif fstate.get("selected_door_id") in door_ids:
            current_id = fstate["selected_door_id"]
            _debug_log("sync_select file_id=%s fstate_select id=%r", str(file_id), current_id)
        else:
            current_id = door_ids[0]
            _debug_log("sync_select file_id=%s default_select id=%r", str(file_id), current_id)

    # Make selection canonical for the rest of this run.
    if st.session_state.get(jump_key) != current_id:
        st.session_state[jump_key] = current_id
    try:
        st.session_state[idx_key] = door_ids.index(current_id) + 1
    except Exception:
        # If anything is off, fall back to 1 so the widget doesn't break.
        st.session_state[idx_key] = 1
    fstate["selected_door_id"] = current_id
    _debug_log(
        "sync_select file_id=%s final current_id=%r jump_now=%r idx_now=%r",
        str(file_id),
        current_id,
        st.session_state.get(jump_key),
        st.session_state.get(idx_key),
    )

    # Bump focus sequence when selection changes (so the viewer auto-focuses only on changes).
    if current_id != fstate.get("_focus_last_id"):
        fstate["_focus_last_id"] = current_id
        try:
            fstate["_focus_seq"] = int(fstate.get("_focus_seq") or 0) + 1
        except Exception:
            fstate["_focus_seq"] = 1

def right_panel_review(
    item: Dict,
    *,
    preview_spec: Optional[Dict[str, Any]],
    doors_data: Dict,
    fstate: Dict,
    active_doors: List,
    all_active_doors: Optional[List] = None,
):
    file_id = item["id"]
    file_dir = Path(item["path"])

    # Don't show "Doors (0)" until analysis has been run at least once.
    status = item.get("status", "not_processed")
    has_run = (status == "done") or (file_dir / "doors.json").exists()
    if not has_run:
        st.info("Analyze to see doors.")
        return
    
    # Use pre-calculated active_doors so the main viewer + right panel stay in sync.
    all_visible = active_doors.copy()
    all_visible.sort(key=lambda x: x["confidence"], reverse=True)
    
    total_doors = len(all_active_doors) if isinstance(all_active_doors, list) else len(all_visible)
    door_type_filter_key = f"door_type_filter_{file_id}"
    current_filter = str(st.session_state.get(door_type_filter_key) or "All")
    if current_filter and current_filter != "All":
        st.subheader(f"Doors ({len(all_visible)} of {total_doors})")
    else:
        st.subheader(f"Doors ({len(all_visible)})")
    
    if not all_visible:
        return

    # Filter panel (stored in session_state; applied before rendering the viewer).
    all_for_filter = (all_active_doors if isinstance(all_active_doors, list) else active_doors) or []
    type_values = sorted(
        {
            str(d.get("type")).strip()
            for d in all_for_filter
            if d.get("type") is not None and str(d.get("type")).strip()
        }
    )
    type_options = ["All"] + type_values
    if door_type_filter_key not in st.session_state:
        st.session_state[door_type_filter_key] = "All"
    if str(st.session_state.get(door_type_filter_key) or "All") not in type_options:
        st.session_state[door_type_filter_key] = "All"

    f_lbl, f_sel = st.columns([1, 3])
    f_lbl.markdown("<div style='line-height: 38px; opacity: 0.85;'>Type</div>", unsafe_allow_html=True)
    f_sel.selectbox("Door type", type_options, key=door_type_filter_key, label_visibility="collapsed")

    # Door navigation (index input + Prev/Next).
    door_ids = [d["id"] for d in all_visible if d.get("id") is not None]
    if not door_ids:
        return

    jump_key = f"jump_{file_id}"
    idx_key = f"{jump_key}__idx"
    idx_label = f"door_jump_idx_{file_id}"

    current_id = st.session_state.get(jump_key) if st.session_state.get(jump_key) in door_ids else door_ids[0]
    selected_idx = door_ids.index(current_id) if current_id in door_ids else 0
    # Keep the editable index input in sync with selection (must happen before widget instantiation).
    if st.session_state.get(idx_key) != (selected_idx + 1):
        st.session_state[idx_key] = selected_idx + 1

    # Prev/Next (wrap-around).
    #
    # Avoid callbacks: Streamlit may run callbacks after widget instantiation,
    # which can trigger "cannot be modified..." if we touch `st.session_state[jump_key]`.
    col_idx = st.container()
    with col_idx:
        c_in, c_total = st.columns([1, 1])
        c_in.number_input(
            idx_label,
            min_value=0,
            max_value=len(door_ids) + 1,
            step=1,
            key=idx_key,
            label_visibility="collapsed",
        )
        c_total.markdown(
            f"<div style='text-align: left; line-height: 38px; padding-left: 6px;'>/ {len(all_visible)}</div>",
            unsafe_allow_html=True,
        )
    # NOTE: Focus sequence is bumped in `_sync_selected_door_for_run(...)` earlier in the run,
    # so the main viewer highlight/focus stays in sync.

    st.divider()
    
    # Details of selected
    # Selection is canonicalized earlier by `_sync_selected_door_for_run(...)`.
    current_id = fstate.get("selected_door_id")
    if current_id not in door_ids:
        current_id = door_ids[0]
    selected_idx = door_ids.index(current_id) if current_id in door_ids else 0
    selected_door = all_visible[selected_idx]
    did = selected_door["id"]
    door_type = html.escape(str(selected_door.get("type", "")))
    try:
        conf = float(selected_door.get("confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    conf_pct = int(round(conf * 100))

    st.markdown(
        f"""
<div class="door_detector-door-meta">
  <div class="door_detector-door-meta-item">
    <span class="door_detector-door-meta-label">Type</span>
    <span class="door_detector-door-meta-type">{door_type}</span>
  </div>
  <div class="door_detector-door-meta-item">
    <span class="door_detector-door-meta-label">Confidence</span>
    <span class="door_detector-door-meta-confidence">{conf_pct}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    
    # Zoom
    if preview_spec:
        image = Image.open(preview_spec["path"])
        bbox = selected_door["bbox_xyxy"]
        nb = _normalize_bbox_xyxy(bbox)
        if nb is None:
            st.warning("Selected door has an invalid bbox; preview unavailable.")
            return
        x0, y0, x1, y1 = nb
        scale = float(preview_spec.get("scale", 1.0))
        pad_full = 100.0
        pad = pad_full * scale

        left = max(0, int(math.floor(x0 * scale - pad)))
        upper = max(0, int(math.floor(y0 * scale - pad)))
        right = min(image.width, int(math.ceil(x1 * scale + pad)))
        lower = min(image.height, int(math.ceil(y1 * scale + pad)))
        if right <= left or lower <= upper:
            st.warning("Selected door bbox is degenerate after clamping; preview unavailable.")
            logger.info(
                "Degenerate preview crop file_id=%s door_id=%s bbox=%s nb=%s scale=%.6f pad_full=%.1f crop=(%d,%d,%d,%d) preview=%dx%d shift=%s",
                file_id,
                did,
                bbox,
                nb,
                scale,
                pad_full,
                left,
                upper,
                right,
                lower,
                image.width,
                image.height,
                doors_data.get("_bbox_transform_fix") or doors_data.get("_bbox_origin_shift"),
            )
        else:
            st.image(image.crop((left, upper, right, lower)), use_container_width=True)

    # Actions
    working = _get_working_label_state(fstate)
    is_editing = bool(fstate.get("edit_mode"))
    c1, c2, c3 = st.columns(3)
    if c1.button("Confirm door", use_container_width=True):
        working["confirmed_ids"].add(did)
        working["deleted_ids"].discard(did)
        # Treat as explicit confirmation (so removing a manual-add record won't unconfirm).
        if is_editing:
            try:
                fstate["_edit_manual_confirmed_ids"].discard(did)
            except Exception:
                pass
        else:
            save_current_labels(file_id, file_dir)
        st.rerun()
    if c2.button("Delete / Not a door", use_container_width=True):
        working["deleted_ids"].add(did)
        working["confirmed_ids"].discard(did)
        if is_editing:
            try:
                fstate["_edit_manual_confirmed_ids"].discard(did)
            except Exception:
                pass
            # If the user marks a candidate as not-a-door, drop any manual-add records
            # that snapped to it (they are no longer meaningful).
            working["manual_additions"] = [
                r for r in list(working.get("manual_additions", [])) if str(r.get("snapped_candidate_id") or "") != str(did)
            ]
        else:
            save_current_labels(file_id, file_dir)
        fstate["selected_door_id"] = None  # Move to next
        st.rerun()
    if c3.button("Skip", use_container_width=True):
        if selected_idx < len(all_visible) - 1:
            next_id = all_visible[selected_idx + 1]["id"]
            fstate["selected_door_id"] = next_id
            st.session_state[jump_key] = next_id
            st.rerun()

    st.divider()
    # Show stats for the currently active label state (draft while editing).
    st.write(
        f"**Stats:** "
        f"{len(working.get('confirmed_ids', set()))} confirmed, "
        f"{len(working.get('deleted_ids', set()))} deleted, "
        f"{len(working.get('manual_additions', []))} manual-added, "
        f"{len(working.get('unmatched_manual_boxes', []))} unmatched"
    )

    if is_editing:
        st.divider()
        st.subheader("Edit Doors")
        st.caption("Shift+drag in the main viewer to add. Save/Cancel are in the top controls.")

        manual_adds = list(working.get("manual_additions", []))
        if manual_adds:
            st.markdown(f"**Manual additions ({len(manual_adds)})**")
            for idx, rec in enumerate(manual_adds):
                cid = rec.get("snapped_candidate_id")
                iou = rec.get("iou")
                label = f"{idx+1}. {cid or '(unmatched?)'}  iou={iou:.3f}" if isinstance(iou, (int, float)) else f"{idx+1}. {cid or '(unmatched?)'}"
                cols = st.columns([5, 1])
                cols[0].write(label)
                if cols[1].button("Remove", key=f"rm_manual_{file_id}_{idx}", use_container_width=True):
                    try:
                        removed = working["manual_additions"].pop(idx)
                    except Exception:
                        removed = None
                    removed_cid = str((removed or {}).get("snapped_candidate_id") or "")
                    if removed_cid:
                        # If this confirmation was only due to manual-add, revert to undecided.
                        try:
                            manual_confirmed = set(fstate.get("_edit_manual_confirmed_ids", set()))
                        except Exception:
                            manual_confirmed = set()
                        still_refs = any(
                            str(r.get("snapped_candidate_id") or "") == removed_cid
                            for r in list(working.get("manual_additions", []))
                        )
                        if (removed_cid in manual_confirmed) and (not still_refs):
                            working["confirmed_ids"].discard(removed_cid)
                            try:
                                fstate["_edit_manual_confirmed_ids"].discard(removed_cid)
                            except Exception:
                                pass
                    st.rerun()
        else:
            st.markdown("**Manual additions (0)**")

        unmatched = list(working.get("unmatched_manual_boxes", []))
        if unmatched:
            st.markdown(f"**Unmatched manual boxes ({len(unmatched)})**")
            for idx, rec in enumerate(unmatched):
                note = str(rec.get("note") or "unmatched")
                cols = st.columns([5, 1])
                cols[0].write(f"{idx+1}. {note}")
                if cols[1].button("Remove", key=f"rm_unmatched_{file_id}_{idx}", use_container_width=True):
                    try:
                        working["unmatched_manual_boxes"].pop(idx)
                    except Exception:
                        pass
                    st.rerun()
        else:
            st.markdown("**Unmatched manual boxes (0)**")

        # Show the most recent unmatched-debug report in the UI (when enabled).
        if st.session_state.get("debug_perf") and fstate.get("_last_unmatched_debug"):
            with st.expander("Why was this unmatched?", expanded=False):
                st.code(str(fstate.get("_last_unmatched_debug")), language="json")
    
    # Train badge
    total_overrides = len(working.get("confirmed_ids", set())) + len(working.get("deleted_ids", set()))
    if (not is_editing) and total_overrides >= 5:
        if st.button("Train Model", use_container_width=True):
            with st.spinner("Training..."):
                fit_reweighter(Path("artifacts"), Path("models/reweighter_v1.json"))
                st.success("Model updated!")
                st.cache_data.clear()
                st.rerun()

def run_pipeline(file_id: str, file_dir: Path, config_path: str):
    lib.update_status(file_id, "processing")
    try:
        primitives_path = file_dir / "primitives.json"
        meta_path = file_dir / "meta.json"
        image_path = file_dir / "page.png"
        pdf_path = file_dir / "source.pdf"

        has_step1_artifacts = primitives_path.exists() and meta_path.exists() and image_path.exists()
        can_run_step1 = pdf_path.exists()

        step1_dpi = 400
        step1_page_index = 0

        if can_run_step1:
            should_run_step1 = True
            if has_step1_artifacts:
                try:
                    stored_meta = json.loads(meta_path.read_text())
                except Exception:
                    stored_meta = {}

                stored_sig = (stored_meta.get("step1") or {}).get("signature")
                try:
                    current_sig = compute_step1_signature(
                        pdf_path=pdf_path, dpi=step1_dpi, page_index=step1_page_index
                    ).get("signature")
                except Exception:
                    current_sig = None

                if stored_sig and current_sig and stored_sig == current_sig:
                    should_run_step1 = False
                    logger.info("Skipping Step 1 for %s (signature match)", str(file_dir))

            if should_run_step1:
                process_pdf(pdf_path, file_dir, dpi=step1_dpi, page_index=step1_page_index)
        elif not has_step1_artifacts:
            raise FileNotFoundError(
                f"Missing Step 1 artifacts in {file_dir} and no source.pdf found; cannot rerun Step 1."
            )

        run_step2(file_dir, Path(config_path))
        lib.update_status(file_id, "done")
        st.cache_data.clear()
        try:
            st.cache_resource.clear()
        except Exception:
            pass
    except Exception as e:
        lib.update_status(file_id, "error", str(e))

def save_current_labels(file_id: str, file_dir: Path):
    fstate = st.session_state.files[file_id]
    labels_to_save = {
        "schema_version": LABELS_SCHEMA_VERSION,
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "confirmed_ids": sorted(list(fstate.get("confirmed_ids", set()))),
        "deleted_ids": sorted(list(fstate.get("deleted_ids", set()))),
        "manual_additions": list(fstate.get("manual_additions", [])),
        "unmatched_manual_boxes": list(fstate.get("unmatched_manual_boxes", [])),
    }
    save_labels(file_dir, labels_to_save)

# --- Layout ---

sidebar_library()

col_app = st.container()

with col_app:
    if "selected_file_id" in st.session_state and st.session_state.selected_file_id:
        items = lib.get_items()
        selected_item = next((i for i in items if i["id"] == st.session_state.selected_file_id), None)

        if selected_item:
            file_id = selected_item["id"]
            file_dir = Path(selected_item["path"])
            perf: Dict[str, float] = {}

            # If a pipeline run is queued for this file, replace the viewer with a loader,
            # then run the pipeline synchronously (so the title stays constant).
            task = st.session_state.get("door_detector_pipeline_task")
            if task and task.get("file_id") == str(file_id) and not bool(task.get("_started")):
                try:
                    task["_started"] = True
                    st.session_state.door_detector_pipeline_task = task
                except Exception:
                    pass
                try:
                    title = html.escape(str(selected_item.get("original_name", "")))
                    st.markdown(f"<div class='door_detector-pdf-title'><h3>{title}</h3></div>", unsafe_allow_html=True)

                    col_main, col_review = st.columns([2, 1])
                    with col_main:
                        _render_viewer_loading(
                            height_px=_get_viewer_height_px(),
                            title="Analyzing…",
                            subtitle="Updating detections and refreshing results. This can take a moment.",
                        )
                    with col_review:
                        st.info("Analysis is running…")

                    run_pipeline(
                        str(file_id),
                        Path(str(task.get("file_dir") or str(file_dir))),
                        str(task.get("config_path") or "configs/door_rules.json"),
                    )
                finally:
                    st.session_state.door_detector_pipeline_task = None
                st.rerun()

            t0 = time.perf_counter()
            try:
                doors_data, labels_data, meta_data = load_file_artifacts(str(file_dir))
            except Exception as e:
                st.error(str(e))
                st.stop()
            perf["load_file_artifacts_ms"] = (time.perf_counter() - t0) * 1000.0
            init_file_state(file_id, doors_data, labels_data)
            fstate = st.session_state.files[file_id]

            full_dims = _get_full_page_dims(meta_data)
            t1 = time.perf_counter()
            page_png_path = file_dir / "page.png"
            try:
                page_png_mtime_ns = page_png_path.stat().st_mtime_ns
            except Exception:
                page_png_mtime_ns = 0
            preview_spec = get_or_create_page_preview(
                str(file_dir),
                full_width=full_dims[0] if full_dims else None,
                full_height=full_dims[1] if full_dims else None,
                page_png_mtime_ns=page_png_mtime_ns,
            )
            perf["get_or_create_page_preview_ms"] = (time.perf_counter() - t1) * 1000.0

            # Consume any Shift+drag events before computing the visible list/overlay.
            _process_draw_event_if_any(
                file_id=str(file_id),
                file_dir=file_dir,
                fstate=fstate,
                doors_data=doors_data,
                preview_spec=preview_spec,
                full_dims=full_dims,
                config_path=str(doors_data.get("config_path") or "configs/door_rules.json"),
            )

            title = html.escape(str(selected_item.get("original_name", "")))
            st.markdown(f"<div class='door_detector-pdf-title'><h3>{title}</h3></div>", unsafe_allow_html=True)

            col_main, col_review = st.columns([2, 1])

            # Compute active doors once so the main viewer + right panel stay in perfect sync.
            detections = doors_data.get("doors", [])
            deleted_ids = _get_working_label_state(fstate).get("deleted_ids", set())
            overlay_doors: List[Dict[str, Any]] = [d for d in detections if d.get("id") is not None]
            # Right panel / navigation list excludes deleted; overlay hides deleted via JS state sink.
            active_doors_all: List[Dict[str, Any]] = [d for d in overlay_doors if d.get("id") not in deleted_ids]

            # Door type filter (affects navigation list + right panel; overlay still includes all).
            door_type_filter_key = f"door_type_filter_{file_id}"
            if door_type_filter_key not in st.session_state:
                st.session_state[door_type_filter_key] = "All"
            selected_type = str(st.session_state.get(door_type_filter_key) or "All")
            id_to_type = {
                str(d.get("id")): str(d.get("type") or "").strip()
                for d in active_doors_all
                if d.get("id") is not None
            }

            # If the user clicked a door of a different type, switch filter to that type so
            # the click "works" while keeping filter semantics.
            click_sink_label = f"door_click_sink_{file_id}"
            clicked_id = st.session_state.get(click_sink_label)
            if clicked_id is not None:
                clicked_type = id_to_type.get(str(clicked_id))
                if clicked_type and selected_type != "All" and clicked_type != selected_type:
                    st.session_state[door_type_filter_key] = clicked_type
                    selected_type = clicked_type

            if selected_type and selected_type != "All":
                active_doors: List[Dict[str, Any]] = [
                    d for d in active_doors_all if str(d.get("type") or "").strip() == selected_type
                ]
            else:
                active_doors = active_doors_all

            click_sink_label = f"door_click_sink_{file_id}"

            # Per-run debug context (helps diagnose viewer flashing / re-mounting).
            run_key = "_door_detector_run_seq"
            try:
                st.session_state[run_key] = int(st.session_state.get(run_key) or 0) + 1
            except Exception:
                st.session_state[run_key] = 1
            run_seq = int(st.session_state.get(run_key) or 0)

            # Sync selection state before rendering the viewer (reduces 1-run lag and flicker).
            all_visible = active_doors.copy()
            all_visible.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
            _debug_log(
                "run=%d file_id=%s mode=%s pre_sync selected=%s focus_seq=%s filter=%s",
                run_seq,
                file_id,
                fstate.get("viewer_display_mode"),
                fstate.get("selected_door_id"),
                fstate.get("_focus_seq"),
                str(st.session_state.get(door_type_filter_key) or "All"),
            )
            _sync_selected_door_for_run(file_id=file_id, fstate=fstate, all_visible=all_visible)
            _debug_log(
                "run=%d file_id=%s post_sync selected=%s focus_seq=%s auto_focus=%s",
                run_seq,
                file_id,
                fstate.get("selected_door_id"),
                fstate.get("_focus_seq"),
                fstate.get("auto_focus"),
            )

            # Render main viewer first so it appears earlier during reruns (less blank flash).
            with col_main:
                t2 = time.perf_counter()
                _debug_log("run=%d file_id=%s render_main start", run_seq, file_id)
                _canvas_result, _ = main_viewer_canvas(
                    selected_item,
                    preview_spec=preview_spec,
                    full_dims=full_dims,
                    doors_data=doors_data,
                    fstate=fstate,
                    active_doors=overlay_doors,
                    click_sink_label=click_sink_label,
                )
                perf["render_main_panel_ms"] = (time.perf_counter() - t2) * 1000.0
                _debug_log(
                    "run=%d file_id=%s render_main done (%.1f ms)",
                    run_seq,
                    file_id,
                    perf["render_main_panel_ms"],
                )

            with col_review:
                t3 = time.perf_counter()
                _debug_log("run=%d file_id=%s render_right start", run_seq, file_id)
                main_viewer_controls(
                    selected_item,
                    full_dims=full_dims,
                    doors_data=doors_data,
                    fstate=fstate,
                )
                st.divider()
                right_panel_review(
                    selected_item,
                    preview_spec=preview_spec,
                    doors_data=doors_data,
                    fstate=fstate,
                    active_doors=active_doors,
                    all_active_doors=active_doors_all,
                )
                perf["render_right_panel_ms"] = (time.perf_counter() - t3) * 1000.0
                _debug_log(
                    "run=%d file_id=%s render_right done (%.1f ms)",
                    run_seq,
                    file_id,
                    perf["render_right_panel_ms"],
                )
        else:
            # Keep sidebar accessible even if a prior session persisted a collapsed state.
            components.html(_sidebar_autopen_component_html(), height=0, scrolling=False)
            st.info("Select a file from the library to begin.")
    else:
        # Keep sidebar accessible even if a prior session persisted a collapsed state.
        components.html(_sidebar_autopen_component_html(), height=0, scrolling=False)
        st.info("Select a file from the library to begin.")
