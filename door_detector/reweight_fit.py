"""Learn door detection weights from reviewer feedback."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


FEATURE_NAMES_V2 = [
    "rmse",
    "radius",
    "angle_span",
    "hinge_dist",
    "len_ratio",
    "center_dist",
    "radial_angle_deg",
    "tip_to_arc_dist",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _load_prior_model(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with open(path) as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _convert_prior_to_current_scaler(
    *,
    prior: Dict[str, Any],
    feature_names: List[str],
    means_new: np.ndarray,
    stds_new: np.ndarray,
) -> Tuple[np.ndarray, float, bool]:
    """Return (w0_scaled, b0, ok) in the *current* scaled space.

    The prior model is assumed to compute:
      z = w_prior^T ((x - mu_prior)/std_prior) + b_prior
    We convert it to an equivalent representation under the new scaler:
      z = w0^T ((x - mu_new)/std_new) + b0
    """
    try:
        prior_order = list(prior.get("feature_order") or [])
        prior_weights = np.array(prior.get("weights") or [], dtype=float)
        prior_bias = float(prior.get("bias", 0.0))
        prior_scaler = dict(prior.get("scaler") or {})
        mu_prior = np.array(prior_scaler.get("mean") or [], dtype=float)
        std_prior = np.array(prior_scaler.get("std") or [], dtype=float)

        if not prior_order or prior_weights.ndim != 1:
            return np.zeros(len(feature_names), dtype=float), 0.0, False
        if len(prior_order) != int(prior_weights.shape[0]):
            return np.zeros(len(feature_names), dtype=float), 0.0, False
        if mu_prior.shape != prior_weights.shape or std_prior.shape != prior_weights.shape:
            return np.zeros(len(feature_names), dtype=float), 0.0, False

        prior_idx = {str(n): i for i, n in enumerate(prior_order)}

        # Convert prior from scaled space to raw-x space.
        w_raw = np.zeros(len(feature_names), dtype=float)
        b_raw = float(prior_bias)
        for j, fname in enumerate(feature_names):
            i = prior_idx.get(fname)
            if i is None:
                continue
            s = float(std_prior[i]) + 1e-8
            wj_scaled = float(prior_weights[i])
            muj = float(mu_prior[i])
            w_raw[j] = wj_scaled / s
            b_raw -= (wj_scaled * muj) / s

        # Convert raw-x space into current scaled space.
        w0_scaled = w_raw * (stds_new + 1e-8)
        b0 = float(b_raw + float(np.dot(w_raw, means_new)))
        return w0_scaled, b0, True
    except Exception:
        return np.zeros(len(feature_names), dtype=float), 0.0, False


def fit_reweighter(
    artifacts_root: Path,
    output_model: Path,
    *,
    min_samples: int = 20,
    min_pos: int = 3,
    min_neg: int = 3,
    base_reg_lambda: float = 1.0,
    reg_lambda_bias: float = 0.25,
    learning_rate: float = 0.1,
) -> None:
    """Read all labels and doors, fit a conservative logistic regression, and save weights."""
    
    # 1. Collect data
    features_list = []
    labels_list = []
    
    # We'll use these features in this order
    feature_names = list(FEATURE_NAMES_V2)
    
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
            did = str(did)
            legacy_ids = set()
            try:
                for lid in (door.get("legacy_ids") or []):
                    if lid is None:
                        continue
                    legacy_ids.add(str(lid))
            except Exception:
                legacy_ids = set()

            is_pos = (did in confirmed) or bool(legacy_ids & confirmed)
            is_neg = (did in deleted) or bool(legacy_ids & deleted)
            if not is_pos and not is_neg:
                continue
            if is_pos and is_neg:
                # Conflicting labels; ignore to avoid poisoning.
                continue

            feats = [door.get("features", {}).get(n, 0.0) for n in feature_names]
            features_list.append(feats)
            labels_list.append(1 if is_pos else 0)

    if not features_list:
        print("No reviewed detections found in labels.json files.")
        return

    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"Training on {len(X)} samples ({sum(y)} positive, {len(y)-sum(y)} negative)")

    n = int(len(X))
    n_pos = int(np.sum(y))
    n_neg = int(n - n_pos)
    if n < int(min_samples) or n_pos < int(min_pos) or n_neg < int(min_neg):
        print(
            f"Not enough labeled data for a conservative update "
            f"(need N>={min_samples} and pos>={min_pos} and neg>={min_neg}). "
            f"Got N={n}, pos={n_pos}, neg={n_neg}. Not writing a new model."
        )
        return

    # 2. Preprocessing (Standard Scaling)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    X_scaled = (X - means) / (stds + 1e-8)

    # 3. Fit Logistic Regression (warm-started + regularized toward prior)
    prior = _load_prior_model(output_model)
    w0, b0, warm_ok = (np.zeros(X.shape[1], dtype=float), 0.0, False)
    if isinstance(prior, dict):
        w0, b0, warm_ok = _convert_prior_to_current_scaler(
            prior=prior, feature_names=feature_names, means_new=means, stds_new=stds
        )

    weights = np.array(w0, dtype=float)
    bias = float(b0)

    # Stronger regularization when data is small (conservative updates).
    reg_lambda = float(base_reg_lambda) * (float(min_samples) / float(max(1, n)))

    # Epoch schedule.
    if n < 50:
        epochs = 300
        patience = 30
    elif n < 200:
        epochs = 600
        patience = 40
    else:
        epochs = 1000
        patience = 60

    best_loss = float("inf")
    best_w = weights.copy()
    best_b = float(bias)
    no_improve = 0
    
    for _ in range(int(epochs)):
        z = np.dot(X_scaled, weights) + bias
        probs = _sigmoid(z)
        
        dw = np.dot(X_scaled.T, (probs - y)) / len(y)
        db = np.sum(probs - y) / len(y)

        # Regularize *toward* the prior (or toward zeros if no prior).
        dw = dw + reg_lambda * (weights - w0)
        db = float(db + float(reg_lambda_bias) * (bias - b0))
        
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Track regularized loss for early stopping.
        eps = 1e-9
        ce = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        reg = 0.5 * reg_lambda * float(np.sum((weights - w0) ** 2)) + 0.5 * float(reg_lambda_bias) * float((bias - b0) ** 2)
        loss = float(ce + reg)
        if loss + 1e-6 < best_loss:
            best_loss = loss
            best_w = weights.copy()
            best_b = float(bias)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(patience):
                break

    weights = best_w
    bias = best_b

    # 4. Save model
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "schema_version": 2,
        "model_type": "logreg",
        "feature_order": feature_names,
        "scaler": {
            "type": "zscore",
            "mean": means.tolist(),
            "std": stds.tolist()
        },
        "weights": weights.tolist(),
        "bias": float(bias),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_samples": int(n),
        "num_pos": int(n_pos),
        "num_neg": int(n_neg),
        "warm_start_used": bool(warm_ok),
        "prior_path": str(output_model) if bool(warm_ok) else None,
        "reg_lambda": float(reg_lambda),
        "reg_lambda_bias": float(reg_lambda_bias),
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


