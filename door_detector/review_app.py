import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from door_detector.step2_pipeline import run_step2

# Increase PIL pixel limit
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Door Detector Door Review", layout="wide")

st.title("Door Detector: Door Detection Review")

# 1. Sidebar for navigation
artifacts_root = Path("artifacts")
if not artifacts_root.exists():
    st.error(f"Artifacts directory not found: {artifacts_root}")
    st.stop()

# Find all artifact directories that have meta.json (Step 1 complete)
artifact_dirs = sorted([
    p.parent for p in artifacts_root.glob("**/meta.json")
])

if not artifact_dirs:
    st.info("No artifact directories found. Run Step 1 first.")
    st.stop()

selected_dir = st.sidebar.selectbox(
    "Select Artifact Directory",
    artifact_dirs,
    format_func=lambda x: str(x.relative_to(artifacts_root))
)

if st.sidebar.button("Rerun Step 2 Detection"):
    with st.spinner("Running detection..."):
        try:
            config_path = Path("configs/door_rules.json")
            run_step2(selected_dir, config_path)
            st.sidebar.success("Detection complete!")
            st.cache_data.clear() # Force reload of data
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# 2. Load data
@st.cache_data
def load_data(dir_path: Path):
    doors_path = dir_path / "doors.json"
    image_path = dir_path / "page.png"
    
    doors_data = None
    if doors_path.exists():
        with open(doors_path) as f:
            doors_data = json.load(f)
    
    image = None
    if image_path.exists():
        image = Image.open(image_path)
        
    return doors_data, image

doors_data, image = load_data(selected_dir)

if image is None:
    st.error(f"Missing page.png in {selected_dir}")
    st.stop()

if doors_data is None:
    st.warning("No door detections found for this directory. Click 'Rerun Step 2 Detection' in the sidebar to run detection.")
    detections = []
else:
    detections = doors_data.get("doors", [])

# 3. Labeling state
labels_path = selected_dir / "labels.json"
if labels_path.exists():
    with open(labels_path) as f:
        existing_labels = json.load(f)
else:
    existing_labels = {
        "accepted_ids": [],
        "rejected_ids": [],
        "notes": ""
    }

# Initialize session state for labels if not already there or if dir changed
if "current_dir" not in st.session_state or st.session_state.current_dir != selected_dir:
    st.session_state.current_dir = selected_dir
    st.session_state.accepted = set(existing_labels.get("accepted_ids", []))
    st.session_state.rejected = set(existing_labels.get("rejected_ids", []))
    st.session_state.notes = existing_labels.get("notes", "")

# 4. Main interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Floor Plan Overlay")
    
    # Create a display image with current selections
    display_img = image.copy()
    draw = ImageDraw.Draw(display_img)
    
    for i, door in enumerate(detections):
        did = door.get("id", f"d_{i:06d}")
        bbox = door["bbox_xyxy"]
        
        if did in st.session_state.accepted:
            color = (0, 255, 0) # Green for accepted
        elif did in st.session_state.rejected:
            color = (255, 0, 0) # Red for rejected
        else:
            color = (255, 165, 0) # Orange for pending
            
        draw.rectangle(bbox, outline=color, width=5)
    
    st.image(display_img, use_container_width=True)

with col2:
    st.subheader("Detections")
    st.write(f"Total: {len(detections)}")
    
    # Selection buttons
    if st.button("Accept All Pending"):
        for i, d in enumerate(detections):
            did = d.get("id", f"d_{i:06d}")
            if did not in st.session_state.rejected:
                st.session_state.accepted.add(did)
        st.rerun()

    if st.button("Reject All Pending"):
        for i, d in enumerate(detections):
            did = d.get("id", f"d_{i:06d}")
            if did not in st.session_state.accepted:
                st.session_state.rejected.add(did)
        st.rerun()

    st.divider()
    
    # Individual detection list
    for i, door in enumerate(detections):
        did = door.get("id", f"d_{i:06d}")
        dtype = door["type"]
        conf = door["confidence"]
        
        status = "Pending"
        if did in st.session_state.accepted:
            status = "Accepted"
        elif did in st.session_state.rejected:
            status = "Rejected"
            
        with st.expander(f"{i}: {dtype} ({conf:.2f}) - {status}"):
            c1, c2 = st.columns(2)
            if c1.button("Accept", key=f"acc_{did}"):
                st.session_state.accepted.add(did)
                st.session_state.rejected.discard(did)
                st.rerun()
            if c2.button("Reject", key=f"rej_{did}"):
                st.session_state.rejected.add(did)
                st.session_state.accepted.discard(did)
                st.rerun()

    st.divider()
    st.session_state.notes = st.text_area("Notes", value=st.session_state.notes)
    
    if st.button("Save Labels", type="primary"):
        labels_to_save = {
            "schema_version": 1,
            "page_id": doors_data["page_id"] if doors_data else selected_dir.name,
            "reviewed_at": None, # Could add timestamp
            "accepted_ids": list(st.session_state.accepted),
            "rejected_ids": list(st.session_state.rejected),
            "added_boxes": [], # Future work
            "notes": st.session_state.notes
        }
        with open(labels_path, "w") as f:
            json.dump(labels_to_save, f, indent=2)
        st.success(f"Saved to {labels_path}")

st.sidebar.divider()
st.sidebar.info("""
**Legend:**
- Orange: Pending
- Green: Accepted
- Red: Rejected
""")

