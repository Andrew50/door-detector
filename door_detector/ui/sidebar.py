"""Sidebar components for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

from door_detector.library import Library


def sidebar_library(lib: Library) -> None:
    st.sidebar.title("Library")

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
            is_selected = st.session_state.get("selected_file_id") == item["id"]
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

