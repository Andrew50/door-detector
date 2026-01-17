import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from door_detector.library import Library
from door_detector.step1_pipeline import process_pdf
from door_detector.step2_pipeline import run_step2
from door_detector.reweight_fit import fit_reweighter

# Increase PIL pixel limit
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Door Detector: Door Detection & Review", layout="wide", initial_sidebar_state="expanded")

# --- UI Styling ---
st.markdown("""
<style>
    [data-testid="stSidebar"] .stButton button {
        height: 28px;
        padding-top: 0px;
        padding-bottom: 0px;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] .stButton p {
        font-size: 13px;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Library ---
if "library" not in st.session_state:
    st.session_state.library = Library(Path("artifacts"))
    # One-time discovery of existing artifacts
    st.session_state.library.discover_existing()

lib = st.session_state.library

# --- Session State Helpers ---
def init_file_state(file_id: str, doors_data: Dict, labels_data: Dict):
    if "files" not in st.session_state:
        st.session_state.files = {}
    
    if file_id not in st.session_state.files:
        st.session_state.files[file_id] = {
            "accepted": set(labels_data.get("accepted_ids", [])),
            "rejected": set(labels_data.get("rejected_ids", [])),
            "added_boxes": labels_data.get("added_boxes", []),
            "notes": labels_data.get("notes", ""),
            "selected_door_id": None,
            "viewer_mode": "Highlight All",
            "overlay_opacity": 0.5,
        }

# --- Data Loading ---
@st.cache_data
def load_file_artifacts(file_dir_str: str):
    file_dir = Path(file_dir_str)
    image_path = file_dir / "page.png"
    doors_path = file_dir / "doors.json"
    labels_path = file_dir / "labels.json"
    meta_path = file_dir / "meta.json"

    image = Image.open(image_path) if image_path.exists() else None
    
    doors_data = {}
    if doors_path.exists():
        with open(doors_path) as f:
            doors_data = json.load(f)
            
    labels_data = {
        "accepted_ids": [],
        "rejected_ids": [],
        "added_boxes": [],
        "notes": ""
    }
    if labels_path.exists():
        with open(labels_path) as f:
            labels_data = json.load(f)
            
    meta_data = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta_data = json.load(f)

    return image, doors_data, labels_data, meta_data

def save_labels(dir_path: Path, labels_data: Dict[str, Any]):
    labels_path = dir_path / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)

def get_current_signature(config_path: str):
    try:
        with open(config_path, "rb") as f:
            config_bytes = f.read()
        config = json.loads(config_bytes)
        sig_content = config_bytes
        if "reweighter_path" in config:
            re_path = Path(config["reweighter_path"])
            if re_path.exists():
                with open(re_path, "rb") as f:
                    sig_content += b"|" + f.read()
        return hashlib.sha256(sig_content).hexdigest()
    except Exception:
        return None

# --- UI Components ---

def sidebar_library():
    col1, col2 = st.sidebar.columns([4, 1])
    col1.title("Library")
    if col2.button("X", key="collapse_sidebar", help="Collapse sidebar"):
        st.info("Use the arrow at the top left to collapse/expand the sidebar.")

    # Search
    search_query = st.sidebar.text_input("Search files...", "").lower()
    
    st.sidebar.divider()
    
    # Upload
    uploaded_file = st.sidebar.file_uploader("Upload Floor Plan PDF", type=["pdf"])
    if uploaded_file:
        if st.sidebar.button("Add to Library"):
            file_id = lib.add_file(uploaded_file.name, uploaded_file.getvalue())
            st.rerun()

    items = lib.get_items()
    if search_query:
        items = [i for i in items if search_query in i["original_name"].lower()]
    
    if not items:
        st.sidebar.info("No files in library.")
    else:
        for item in items:
            col_sel, col_del = st.sidebar.columns([5, 1])
            
            display_name = item["original_name"]
            # Manual truncation to keep it on one line
            if len(display_name) > 22:
                display_name = display_name[:19] + "..."
            
            status = item.get("status", "not_processed")
            status_tag = ""
            if status == "processing":
                status_tag = "(P) "
            elif status == "error":
                status_tag = "(E) "
            elif status == "done":
                status_tag = "(D) "
            
            label = f"{status_tag}{display_name}"
            
            if col_sel.button(label, key=f"sel_{item['id']}", use_container_width=True, help=item["original_name"]):
                st.session_state.selected_file_id = item["id"]
                st.rerun()
                
            if col_del.button("X", key=f"del_{item['id']}", help="Delete file"):
                lib.delete_item(item["id"])
                if st.session_state.get("selected_file_id") == item["id"]:
                    st.session_state.selected_file_id = None
                st.rerun()

def main_viewer(item: Dict):
    file_id = item["id"]
    file_dir = Path(item["path"])
    
    image, doors_data, labels_data, meta_data = load_file_artifacts(str(file_dir))
    init_file_state(file_id, doors_data, labels_data)
    fstate = st.session_state.files[file_id]
    
    st.title(item['original_name'])
    
    # Controls bar
    col_run, col_mode, col_add = st.columns([1, 1, 1])
    
    config_path = "configs/door_rules.json" # Default
    current_sig = get_current_signature(config_path)
    stored_sig = doors_data.get("analysis_signature")
    is_out_of_date = stored_sig and current_sig and stored_sig != current_sig
    
    with col_run:
        status = item.get("status", "not_processed")
        if status == "processing":
            st.button("Processing...", disabled=True)
        else:
            label = "Rerun Analysis" if status == "done" else "Run Analysis"
            if is_out_of_date:
                label = f"{label} (Out of Date)"
                st.warning("Analysis config changed. Rerun recommended.")
            
            if st.button(label, type="primary" if not status == "done" else "secondary"):
                run_pipeline(file_id, file_dir, config_path)
                st.rerun()

    with col_mode:
        fstate["viewer_mode"] = st.selectbox(
            "Overlay Mode", 
            ["Highlight All", "Highlight Selected", "Off"],
            index=["Highlight All", "Highlight Selected", "Off"].index(fstate["viewer_mode"])
        )

    with col_add:
        if st.button("Add Door"):
            fstate["viewer_mode"] = "Add Door"

    # Display Image / Canvas
    if image:
        detections = doors_data.get("doors", [])
        
        # Filter detections: keep undecided and accepted, exclude rejected
        active_doors = [d for d in detections if d["id"] not in fstate["rejected"]]
        # Add user added boxes as pseudo-detections
        for box in fstate["added_boxes"]:
            bbox = box["bbox_xyxy"]
            # Stable ID for added box based on coordinates
            box_id = f"u_{int(bbox[0])}_{int(bbox[1])}"
            active_doors.append({
                "id": box_id,
                "type": "added",
                "bbox_xyxy": bbox,
                "confidence": 1.0,
                "is_user_added": True
            })

        if fstate["viewer_mode"] == "Add Door":
            st.write("Draw a rectangle for the missed door.")
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 255, 0.3)",
                stroke_width=2,
                stroke_color="#00ffff",
                background_image=image,
                update_streamlit=True,
                height=int(image.height * (1000 / image.width)) if image.width > 1000 else image.height,
                width=1000 if image.width > 1000 else image.width,
                drawing_mode="rect",
                key=f"canvas_{file_id}",
            )
            
            if st.button("Save Added Door"):
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    scale_x = image.width / (1000 if image.width > 1000 else image.width)
                    scale_y = image.height / (int(image.height * (1000 / image.width)) if image.width > 1000 else image.height)
                    
                    for obj in objects:
                        if obj["type"] == "rect":
                            x0 = obj["left"] * scale_x
                            y0 = obj["top"] * scale_y
                            x1 = (obj["left"] + obj["width"]) * scale_x
                            y1 = (obj["top"] + obj["height"]) * scale_y
                            fstate["added_boxes"].append({"bbox_xyxy": [x0, y0, x1, y1]})
                    
                    save_current_labels(file_id, file_dir)
                    fstate["viewer_mode"] = "Highlight All"
                    st.rerun()
        else:
            # Normal viewing with click capture
            display_img = image.copy()
            draw = ImageDraw.Draw(display_img)
            
            if fstate["viewer_mode"] != "Off":
                for d in active_doors:
                    is_selected = d["id"] == fstate["selected_door_id"]
                    if fstate["viewer_mode"] == "Highlight Selected" and not is_selected:
                        continue
                        
                    bbox = d["bbox_xyxy"]
                    color = (0, 255, 0) if d["id"] in fstate["accepted"] else (255, 165, 0)
                    if d.get("is_user_added"):
                        color = (0, 255, 255)
                    
                    width = 4
                    if is_selected:
                        color = (255, 255, 255)
                        width = 8
                    
                    draw.rectangle(bbox, outline=color, width=width)

            # Use canvas just to capture clicks
            canvas_click = st_canvas(
                background_image=display_img,
                update_streamlit=True,
                height=int(image.height * (1000 / image.width)) if image.width > 1000 else image.height,
                width=1000 if image.width > 1000 else image.width,
                drawing_mode="point",
                display_toolbar=False,
                key=f"viewer_{file_id}",
            )
            
            if canvas_click.json_data and canvas_click.json_data["objects"]:
                last_point = canvas_click.json_data["objects"][-1]
                if last_point["type"] == "circle":
                    scale_x = image.width / (1000 if image.width > 1000 else image.width)
                    scale_y = image.height / (int(image.height * (1000 / image.width)) if image.width > 1000 else image.height)
                    click_x = last_point["left"] * scale_x
                    click_y = last_point["top"] * scale_y
                    
                    # Find clicked door
                    clicked_id = None
                    for d in active_doors:
                        b = d["bbox_xyxy"]
                        if b[0] <= click_x <= b[2] and b[1] <= click_y <= b[3]:
                            clicked_id = d["id"]
                            break
                    
                    if clicked_id:
                        fstate["selected_door_id"] = clicked_id
                        st.rerun()

    else:
        st.info("Run analysis to see results.")

def right_panel_review(item: Dict):
    file_id = item["id"]
    file_dir = Path(item["path"])
    image, doors_data, labels_data, meta_data = load_file_artifacts(str(file_dir))
    
    if file_id not in st.session_state.files:
        return # Not initialized yet
        
    fstate = st.session_state.files[file_id]
    detections = doors_data.get("doors", [])
    
    # Filter and sort
    all_visible = []
    for d in detections:
        if d["id"] not in fstate["rejected"]:
            all_visible.append(d)
    for box in fstate["added_boxes"]:
        bbox = box["bbox_xyxy"]
        box_id = f"u_{int(bbox[0])}_{int(bbox[1])}"
        all_visible.append({
            "id": box_id,
            "type": "added",
            "bbox_xyxy": bbox,
            "confidence": 1.0,
            "is_user_added": True
        })
    
    all_visible.sort(key=lambda x: x["confidence"], reverse=True)
    
    st.subheader(f"Doors ({len(all_visible)})")
    
    if not all_visible:
        st.write("No doors detected.")
        return

    # Selection sync
    selected_idx = -1
    if fstate["selected_door_id"]:
        for i, d in enumerate(all_visible):
            if d["id"] == fstate["selected_door_id"]:
                selected_idx = i
                break
    
    if selected_idx == -1 and all_visible:
        selected_idx = 0
        fstate["selected_door_id"] = all_visible[0]["id"]

    # Prev/Next
    col_p, col_idx, col_n = st.columns([1, 2, 1])
    if col_p.button("Prev", disabled=selected_idx <= 0):
        fstate["selected_door_id"] = all_visible[selected_idx - 1]["id"]
        st.rerun()
    col_idx.write(f"{selected_idx + 1} / {len(all_visible)}")
    if col_n.button("Next", disabled=selected_idx >= len(all_visible) - 1):
        fstate["selected_door_id"] = all_visible[selected_idx + 1]["id"]
        st.rerun()

    st.divider()
    
    # Details of selected
    selected_door = all_visible[selected_idx]
    did = selected_door["id"]
    
    st.write(f"**ID:** `{did}`")
    st.write(f"**Type:** {selected_door['type']}")
    st.write(f"**Confidence:** {selected_door['confidence']:.3f}")
    
    # Zoom
    if image:
        bbox = selected_door["bbox_xyxy"]
        pad = 100
        crop_box = (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(image.width, bbox[2] + pad),
            min(image.height, bbox[3] + pad)
        )
        st.image(image.crop(crop_box), use_container_width=True)

    # Actions
    c1, c2, c3 = st.columns(3)
    if c1.button("Accept", use_container_width=True):
        fstate["accepted"].add(did)
        fstate["rejected"].discard(did)
        save_current_labels(file_id, file_dir)
        st.rerun()
    if c2.button("Reject", use_container_width=True):
        if selected_door.get("is_user_added"):
            # If it's user added, "Reject" means remove it entirely
            # Find by ID
            fstate["added_boxes"] = [b for b in fstate["added_boxes"] if f"u_{int(b['bbox_xyxy'][0])}_{int(b['bbox_xyxy'][1])}" != did]
        else:
            fstate["rejected"].add(did)
            fstate["accepted"].discard(did)
        save_current_labels(file_id, file_dir)
        fstate["selected_door_id"] = None # Move to next
        st.rerun()
    if c3.button("Skip", use_container_width=True):
        if selected_idx < len(all_visible) - 1:
            fstate["selected_door_id"] = all_visible[selected_idx + 1]["id"]
            st.rerun()

    st.divider()
    st.write(f"**Stats:** {len(fstate['accepted'])} Accepted, {len(fstate['rejected'])} Rejected, {len(fstate['added_boxes'])} Added")
    
    # Train badge
    total_overrides = len(fstate["accepted"]) + len(fstate["rejected"]) + len(fstate["added_boxes"])
    if total_overrides >= 5:
        st.success("Ready to retrain!")
        if st.button("Train Reweighter"):
            with st.spinner("Training..."):
                fit_reweighter(Path("artifacts"), Path("models/reweighter_v1.json"))
                st.success("Model updated!")
                st.cache_data.clear()
                st.rerun()

def run_pipeline(file_id: str, file_dir: Path, config_path: str):
    lib.update_status(file_id, "processing")
    try:
        pdf_path = file_dir / "source.pdf"
        process_pdf(pdf_path, file_dir, dpi=400, page_index=0)
        run_step2(file_dir, Path(config_path))
        lib.update_status(file_id, "done")
        st.cache_data.clear()
    except Exception as e:
        lib.update_status(file_id, "error", str(e))

def save_current_labels(file_id: str, file_dir: Path):
    fstate = st.session_state.files[file_id]
    labels_to_save = {
        "schema_version": 1,
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accepted_ids": list(fstate["accepted"]),
        "rejected_ids": list(fstate["rejected"]),
        "added_boxes": fstate["added_boxes"],
        "notes": fstate["notes"]
    }
    save_labels(file_dir, labels_to_save)

# --- Layout ---

sidebar_library()

if "selected_file_id" in st.session_state and st.session_state.selected_file_id:
    items = lib.get_items()
    selected_item = next((i for i in items if i["id"] == st.session_state.selected_file_id), None)
    
    if selected_item:
        col_main, col_review = st.columns([2, 1])
        
        with col_main:
            main_viewer(selected_item)
            
        with col_review:
            right_panel_review(selected_item)
else:
    st.info("Select a file from the library to begin.")
