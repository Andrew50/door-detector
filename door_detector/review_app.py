import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from door_detector.step1_pipeline import process_pdf
from door_detector.step2_pipeline import run_step2
from door_detector.reweight_fit import fit_reweighter

# Increase PIL pixel limit
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Door Detector: Door Detection & Review", layout="wide")

# --- Helpers ---

def get_artifact_dirs(root: Path) -> List[Path]:
    """Find all directories that contain meta.json (Step 1 output)."""
    if not root.exists():
        return []
    return sorted([p.parent for p in root.glob("**/meta.json")])

@st.cache_data
def load_artifact_bundle(dir_path: Path):
    """Load image, detections, and existing labels."""
    image_path = dir_path / "page.png"
    doors_path = dir_path / "doors.json"
    labels_path = dir_path / "labels.json"
    meta_path = dir_path / "meta.json"

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

# --- UI Setup ---

st.title("Door Detector: Door Detection & Review")

tab_run, tab_review, tab_learn = st.tabs(["🚀 Run Pipeline", "🔍 Review Detections", "🧠 Train Reweighter"])

# --- Tab 1: Run Pipeline ---

with tab_run:
    st.header("Run Detection Pipeline")
    
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_files = st.file_uploader("Upload Floor Plan PDFs", type=["pdf"], accept_multiple_files=True)
        dpi = st.slider("Rendering DPI", 100, 600, 400)
        page_index = st.number_input("Page Index", min_value=0, value=0)
        
    with col_b:
        config_path = st.text_input("Config Path", value="configs/door_rules.json")
        debug_overlay = st.checkbox("Generate Debug Overlay", value=True)
        
    if st.button("Run Full Pipeline", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            upload_root = Path("artifacts/_uploads")
            upload_root.mkdir(parents=True, exist_ok=True)
            
            results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                pdf_path = upload_root / uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Output dir naming: artifacts/<pdf_stem>/p<page_index>/
                out_dir = Path("artifacts") / pdf_path.stem / f"p{page_index}"
                
                status_text.text(f"Processing {uploaded_file.name} (Page {page_index})...")
                
                try:
                    # Step 1
                    process_pdf(
                        pdf_path=pdf_path,
                        output_dir=out_dir,
                        dpi=dpi,
                        page_index=page_index,
                        enable_debug_overlay=debug_overlay
                    )
                    
                    # Step 2
                    run_step2(
                        artifacts_dir=out_dir,
                        config_path=Path(config_path)
                    )
                    
                    results.append({
                        "File": uploaded_file.name,
                        "Page": page_index,
                        "Status": "✅ Success",
                        "Output": str(out_dir)
                    })
                except Exception as e:
                    results.append({
                        "File": uploaded_file.name,
                        "Page": page_index,
                        "Status": f"❌ Error: {str(e)}",
                        "Output": str(out_dir)
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.table(results)
            st.cache_data.clear() # Refresh directories in Review tab

# --- Tab 2: Review Detections ---

with tab_review:
    st.header("Review & Feedback")
    
    artifact_dirs = get_artifact_dirs(Path("artifacts"))
    
    if not artifact_dirs:
        st.info("No artifacts found. Run the pipeline first.")
    else:
        # Sidebar-like selection within the tab for better layout
        selected_dir = st.selectbox(
            "Select Artifact Directory",
            artifact_dirs,
            format_func=lambda x: str(x.relative_to(Path("artifacts")))
        )
        
        image, doors_data, labels_data, meta_data = load_artifact_bundle(selected_dir)
        
        if image is None:
            st.error(f"Image not found in {selected_dir}")
        else:
            detections = doors_data.get("doors", [])
            
            # Initialize session state for this directory
            if "current_dir" not in st.session_state or st.session_state.current_dir != selected_dir:
                st.session_state.current_dir = selected_dir
                st.session_state.accepted = set(labels_data.get("accepted_ids", []))
                st.session_state.rejected = set(labels_data.get("rejected_ids", []))
                st.session_state.added_boxes = labels_data.get("added_boxes", [])
                st.session_state.notes = labels_data.get("notes", "")
                st.session_state.door_idx = 0 if detections else -1
            
            # Filters & Controls
            st.sidebar.title("Navigation & Filters")
            
            if detections:
                # Ensure door_idx is valid if we have detections
                if st.session_state.door_idx < 0:
                    st.session_state.door_idx = 0
                
                st.session_state.door_idx = st.sidebar.number_input(
                    "Door Index", 0, len(detections)-1, st.session_state.door_idx
                )
                
                col_nav1, col_nav2 = st.sidebar.columns(2)
                if col_nav1.button("⬅️ Previous"):
                    st.session_state.door_idx = max(0, st.session_state.door_idx - 1)
                    st.rerun()
                if col_nav2.button("Next ➡️"):
                    st.session_state.door_idx = min(len(detections) - 1, st.session_state.door_idx + 1)
                    st.rerun()

            st.sidebar.divider()
            viewer_mode = st.sidebar.radio("Viewer Mode", ["Review Overlay", "Model Overlay", "Original Image", "Add Missed Doors"])
            
            # Main View
            col_view, col_info = st.columns([3, 1])
            
            with col_view:
                if viewer_mode == "Add Missed Doors":
                    st.write("Draw rectangles to mark missed doors. Double-click to remove a box.")
                    # Canvas for drawing boxes
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 255, 0, 0.3)",
                        stroke_width=3,
                        stroke_color="#00ff00",
                        background_image=image,
                        update_streamlit=True,
                        height=int(image.height * (800 / image.width)) if image.width > 800 else image.height,
                        width=800 if image.width > 800 else image.width,
                        drawing_mode="rect",
                        key="canvas",
                    )
                    
                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data["objects"]
                        if st.button("Save Added Boxes"):
                            new_boxes = []
                            # Scale coordinates back to original image size
                            scale_x = image.width / 800 if image.width > 800 else 1.0
                            scale_y = image.height / (image.height * (800 / image.width)) if image.width > 800 else 1.0
                            
                            for obj in objects:
                                if obj["type"] == "rect":
                                    x0_raw = obj["left"] * scale_x
                                    y0_raw = obj["top"] * scale_y
                                    x1_raw = (obj["left"] + obj["width"]) * scale_x
                                    y1_raw = (obj["top"] + obj["height"]) * scale_y
                                    
                                    # Normalize coordinates to ensure x0 <= x1 and y0 <= y1
                                    x0 = min(x0_raw, x1_raw)
                                    x1 = max(x0_raw, x1_raw)
                                    y0 = min(y0_raw, y1_raw)
                                    y1 = max(y0_raw, y1_raw)
                                    
                                    new_boxes.append({"bbox_xyxy": [x0, y0, x1, y1], "note": "Added via UI"})
                            
                            st.session_state.added_boxes = new_boxes
                            st.success(f"Captured {len(new_boxes)} boxes.")
                
                else:
                    # Rendering Logic
                    display_img = image.copy()
                    draw = ImageDraw.Draw(display_img)
                    
                    if viewer_mode == "Review Overlay":
                        for i, d in enumerate(detections):
                            did = d.get("id", f"d_{i:06d}")
                            bbox = d["bbox_xyxy"]
                            
                            # Normalize for PIL
                            x0 = min(bbox[0], bbox[2])
                            y0 = min(bbox[1], bbox[3])
                            x1 = max(bbox[0], bbox[2])
                            y1 = max(bbox[1], bbox[3])
                            norm_bbox = [x0, y0, x1, y1]
                            
                            if did in st.session_state.accepted:
                                color = (0, 255, 0) # Green
                                width = 3
                            elif did in st.session_state.rejected:
                                color = (255, 0, 0) # Red
                                width = 3
                            else:
                                color = (255, 165, 0) # Orange
                                width = 3
                            
                            # Emphasize current door
                            if i == st.session_state.door_idx:
                                color = (255, 255, 255)
                                width = 8
                            
                            draw.rectangle(norm_bbox, outline=color, width=width)
                        
                        for box in st.session_state.added_boxes:
                            bbox = box["bbox_xyxy"]
                            norm_bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])]
                            draw.rectangle(norm_bbox, outline=(0, 255, 255), width=5) # Cyan for additions

                    elif viewer_mode == "Model Overlay":
                        # Load from doors_overlay.png if it exists, otherwise draw
                        overlay_path = selected_dir / "doors_overlay.png"
                        if overlay_path.exists():
                            display_img = Image.open(overlay_path)
                        else:
                            for d in detections:
                                draw.rectangle(d["bbox_xyxy"], outline=(0, 255, 0), width=3)

                    st.image(display_img, use_container_width=True)

            with col_info:
                st.subheader("Details")
                if st.session_state.door_idx >= 0 and st.session_state.door_idx < len(detections):
                    door = detections[st.session_state.door_idx]
                    did = door.get("id", f"d_{st.session_state.door_idx:06d}")
                    st.write(f"**ID:** {did}")
                    st.write(f"**Type:** {door.get('type')}")
                    st.write(f"**Confidence:** {door.get('confidence'):.3f}")
                    
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Accept", key=f"acc_{did}"):
                        st.session_state.accepted.add(did)
                        st.session_state.rejected.discard(did)
                    if c2.button("❌ Reject", key=f"rej_{did}"):
                        st.session_state.rejected.add(did)
                        st.session_state.accepted.discard(did)
                        
                    with st.expander("Geometric Features"):
                        st.json(door.get("features", {}))
                        
                    # Zoomed crop
                    bbox = door["bbox_xyxy"]
                    # Ensure bbox is [x0, y0, x1, y1] with x0 <= x1 and y0 <= y1
                    x0 = min(bbox[0], bbox[2])
                    x1 = max(bbox[0], bbox[2])
                    y0 = min(bbox[1], bbox[3])
                    y1 = max(bbox[1], bbox[3])
                    
                    pad = 100
                    left = max(0, x0 - pad)
                    top = max(0, y0 - pad)
                    right = min(image.width, x1 + pad)
                    bottom = min(image.height, y1 + pad)
                    
                    # Check if the detection is entirely outside the image
                    is_off_screen = (x1 < 0 or x0 > image.width or y1 < 0 or y0 > image.height)
                    
                    if not is_off_screen and right > left and bottom > top:
                        crop_box = (left, top, right, bottom)
                        st.image(image.crop(crop_box), caption="Zoomed View")
                    elif is_off_screen:
                        st.warning(f"Detection is outside the rendered page area (Image: {image.width}x{image.height}, BBox: [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}])")
                    else:
                        st.warning("Could not generate valid zoom view for this detection.")
                
                st.divider()
                st.write(f"**Accepted:** {len(st.session_state.accepted)}")
                st.write(f"**Rejected:** {len(st.session_state.rejected)}")
                st.write(f"**Added:** {len(st.session_state.added_boxes)}")
                
                st.session_state.notes = st.text_area("Notes", value=st.session_state.notes)
                
                if st.button("💾 Save Labels", type="primary"):
                    labels_to_save = {
                        "schema_version": 1,
                        "page_id": meta_data.get("id", selected_dir.name),
                        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "accepted_ids": list(st.session_state.accepted),
                        "rejected_ids": list(st.session_state.rejected),
                        "added_boxes": st.session_state.added_boxes,
                        "notes": st.session_state.notes
                    }
                    save_labels(selected_dir, labels_to_save)
                    st.success("Labels saved!")

# --- Tab 3: Learn ---

with tab_learn:
    st.header("Train Door Reweighter")
    st.write("Improve detection accuracy by training a logistic regression model on your reviewed labels.")
    
    artifacts_root = st.text_input("Artifacts Root", value="artifacts")
    model_out = st.text_input("Model Output Path", value="models/reweighter_v1.json")
    
    if st.button("🔥 Fit Reweighter"):
        with st.spinner("Training..."):
            try:
                fit_reweighter(Path(artifacts_root), Path(model_out))
                
                if Path(model_out).exists():
                    st.success(f"Model saved to {model_out}")
                    with open(model_out) as f:
                        model_data = json.load(f)
                    
                    st.subheader("Learned Weights")
                    weights_df = {
                        "Feature": model_data["feature_order"],
                        "Weight": model_data["weights"]
                    }
                    st.table(weights_df)
                    st.write(f"**Bias:** {model_data['bias']:.4f}")
                    
                    st.info(f"""
                    To use this model, update your `{config_path}`:
                    ```json
                    {{
                      "reweighter_path": "{model_out}",
                      ...
                    }}
                    ```
                    """)
                else:
                    st.error("No labels found to train on. Review some doors first!")
            except Exception as e:
                st.error(f"Training failed: {e}")
