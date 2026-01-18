"""Learn door detection weights from reviewer feedback."""

import argparse
import json
from pathlib import Path
from typing import List

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

    for label_path in label_files:
        dir_path = label_path.parent
        doors_path = dir_path / "doors.json"
        
        if not doors_path.exists():
            continue
            
        with open(label_path) as f:
            labels_data = json.load(f)
        with open(doors_path) as f:
            doors_data = json.load(f)

        # v2 labels only
        if labels_data.get("schema_version") != 2:
            print(f"Skipping non-v2 labels.json: {label_path}")
            continue

        confirmed = set(labels_data.get("confirmed_ids", []))
        deleted = set(labels_data.get("deleted_ids", []))

        all_candidates = doors_data.get("candidates", doors_data.get("doors", [])) or []

        # Train on candidate feature vectors only.
        for door in all_candidates:
            did = door.get("id")
            if did is None:
                continue
            if did not in confirmed and did not in deleted:
                continue

            feats = [door.get("features", {}).get(n, 0.0) for n in feature_names]
            features_list.append(feats)
            labels_list.append(1 if did in confirmed else 0)

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


