"""Learn door detection weights from reviewer feedback."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def fit_reweighter(artifacts_root: Path, output_model: Path) -> None:
    """Read all labels and doors, fit a logistic regression, and save weights."""
    
    # 1. Collect data
    features_list = []
    labels_list = []
    
    # We'll use these features in this order
    feature_names = ["rmse", "radius", "angle_span", "hinge_dist", "len_ratio"]
    
    # We'll look into artifacts/library as well
    label_files = list(artifacts_root.glob("**/labels.json"))
    if not label_files:
        print("No labels.json files found. Go review some detections first!")
        return

    from door_detector.door_features import compute_iou

    for label_path in label_files:
        dir_path = label_path.parent
        doors_path = dir_path / "doors.json"
        
        if not doors_path.exists():
            continue
            
        with open(label_path) as f:
            labels_data = json.load(f)
        with open(doors_path) as f:
            doors_data = json.load(f)
            
        accepted = set(labels_data.get("accepted_ids", []))
        rejected = set(labels_data.get("rejected_ids", []))
        added_boxes = labels_data.get("added_boxes", [])
        
        all_candidates = doors_data.get("doors", [])
        
        # 1a. Process accepted/rejected from doors.json
        for door in all_candidates:
            did = door["id"]
            if did in accepted or did in rejected:
                feats = [door["features"].get(n, 0.0) for n in feature_names]
                features_list.append(feats)
                labels_list.append(1 if did in accepted else 0)

        # 1b. Process added boxes: try to match them to a candidate that was NOT in the final list
        # (or even one that was, but wasn't explicitly accepted/rejected yet)
        for box in added_boxes:
            best_iou = 0
            best_cand = None
            for cand in all_candidates:
                iou = compute_iou(box["bbox_xyxy"], cand["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_cand = cand
            
            if best_iou > 0.5: # If we matched a candidate
                # Check if we already added this candidate via 'accepted'
                if best_cand["id"] in accepted:
                    continue # Already added
                
                feats = [best_cand["features"].get(n, 0.0) for n in feature_names]
                features_list.append(feats)
                labels_list.append(1) # It was an added box, so it's a positive

    if not features_list:
        print("No reviewed detections found in labels.json files.")
        return

    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"Training on {len(X)} samples ({sum(y)} positive, {len(y)-sum(y)} negative)")

    # 2. Preprocessing (Standard Scaling)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    X_scaled = (X - means) / (stds + 1e-8)

    # 3. Fit Logistic Regression (Simple Gradient Descent)
    weights = np.zeros(X.shape[1])
    bias = 0.0
    learning_rate = 0.1
    epochs = 1000
    
    for _ in range(epochs):
        z = np.dot(X_scaled, weights) + bias
        probs = 1.0 / (1.0 + np.exp(-z))
        
        dw = np.dot(X_scaled.T, (probs - y)) / len(y)
        db = np.sum(probs - y) / len(y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db

    # 4. Save model
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "schema_version": 1,
        "model_type": "logreg",
        "feature_order": feature_names,
        "scaler": {
            "mean": means.tolist(),
            "std": stds.tolist()
        },
        "weights": weights.tolist(),
        "bias": float(bias)
    }
    
    with open(output_model, "w") as f:
        json.dump(model_data, f, indent=2)
    
    print(f"✓ Model saved to {output_model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit door reweighter from feedback")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Root artifacts directory")
    parser.add_argument("--out", type=Path, default=Path("models/reweighter_v1.json"), help="Output model path")
    
    args = parser.parse_args()
    fit_reweighter(args.artifacts, args.out)


if __name__ == "__main__":
    main()


