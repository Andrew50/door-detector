"""UI assets (CSS + small HTML/JS snippets)."""

from __future__ import annotations

import streamlit as st


GLOBAL_STYLE_HTML = r"""
<style>
    /* Hide Streamlit chrome (cosmetic only; not a security boundary).
       We keep the sidebar always visible and remove the top toolbar entirely. */
    /* Streamlit 1.53 reserves space for the header via CSS variables; force it to 0. */
    :root {
        --header-height: 0px !important;
        --st-header-height: 0px !important;
    }

    /* Streamlit's thin colored "decoration" bar at the very top. */
    [data-testid="stDecoration"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
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

    /* Main viewer: keep PDF title tight to the viewer (top-justified) */
    .door_detector-pdf-title {
        --door_detector-title-font-size: 1.35rem;
        --door_detector-title-line-height: 1.55rem;

        /* Keep spacing minimal so the viewer starts right under the title */
        padding-top: 0rem;
        margin: 0 0 0.15rem 0;
        overflow: hidden;
    }

    .door_detector-pdf-title > h3 {
        font-size: var(--door_detector-title-font-size) !important;
        line-height: var(--door_detector-title-line-height) !important;

        margin: 0 !important;
        padding: 0 !important; /* override global .main h3 padding-top */

        /* Clamp to two lines without reserving extra vertical space */
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

    div[data-testid="stTextInput"]:has(input[aria-label^="unmatched_debug_sink_"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"] input[aria-label^="unmatched_debug_sink_"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"]:has(input[aria-label^="candidate_pool_sink_"]) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stTextInput"] input[aria-label^="candidate_pool_sink_"] {
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
    .door_detector-door-meta-confirmed {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.2px;
        line-height: 1.2;
        white-space: nowrap;
        color: rgba(135, 255, 190, 0.95);
        background: rgba(0, 200, 83, 0.18);
        border: 1px solid rgba(0, 200, 83, 0.35);
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
"""


def inject_global_styles() -> None:
    """Inject global CSS overrides for the Streamlit UI."""
    st.markdown(GLOBAL_STYLE_HTML, unsafe_allow_html=True)


def sidebar_autopen_component_html() -> str:
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

