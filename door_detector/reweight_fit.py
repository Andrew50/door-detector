"""Learn door detection weights from reviewer feedback."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from door_detector.doors.types import DOOR_TYPES, normalize_door_type


FEATURES_BY_TYPE: Dict[str, List[str]] = {
    # Swing candidate features (existing)
    "swing": [
        "rmse",
        "radius",
        "angle_span",
        "hinge_dist",
        "len_ratio",
        "center_dist",
        "radial_angle_deg",
        "tip_to_arc_dist",
    ],
    # Double candidates are formed by pairing swing candidates.
    "double": [
        "center_dist",
        "radius_ratio",
        "avg_swing_conf",
        "pair_score",
    ],
    # Pocket candidates are dashed “track-like” lines.
    "pocket": [
        "track_length_px",
        "is_dashed",
        "dash_len",
        "gap_len",
        "stroke_width",
    ],
    # Bi-fold candidates are zig-zag chains of short segments.
    "bifold": [
        "num_segments",
        "avg_turn_angle_deg",
        "path_length_px",
        "compactness",
    ],
}


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
    output_dir: Path,
    *,
    door_type: Optional[str] = None,
    min_samples: int = 20,
    min_pos: int = 3,
    min_neg: int = 3,
    base_reg_lambda: float = 1.0,
    reg_lambda_bias: float = 0.25,
    learning_rate: float = 0.1,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Fit one (or all) per-type binary reweighters from reviewer feedback.

    Returns a best-effort training report dict suitable for showing in the UI.
    """
    types: List[str]
    if door_type:
        types = [normalize_door_type(door_type, default=str(door_type))]
    else:
        types = list(DOOR_TYPES)

    started_at = time.time()
    report: Dict[str, Any] = {
        "schema_version": 1,
        "artifacts_root": str(artifacts_root),
        "output_dir": str(output_dir),
        "types": list(types),
        "thresholds": {
            "min_samples": int(min_samples),
            "min_pos": int(min_pos),
            "min_neg": int(min_neg),
        },
        "label_files_found": 0,
        "models_written": 0,
        "by_type": {t: {"door_type": t, "status": "unknown"} for t in types},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": None,
    }

    def _emit(ev: Dict[str, Any]) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(dict(ev))
        except Exception:
            return

    # 1) Collect labeled examples by candidate type.
    X_by_type: Dict[str, List[List[float]]] = {t: [] for t in types}
    y_by_type: Dict[str, List[int]] = {t: [] for t in types}

    label_files = list(artifacts_root.glob("**/labels.json"))
    report["label_files_found"] = int(len(label_files))
    if not label_files:
        print("No labels.json files found. Go review some detections first!")
        for t in types:
            report["by_type"][t] = {
                "door_type": t,
                "status": "skipped",
                "reason": "no_labels",
                "message": "No labels found.",
                "num_samples": 0,
                "num_pos": 0,
                "num_neg": 0,
                "model_written": False,
                "output_path": str(output_dir / f"reweighter_{t}_v1.json"),
            }
        report["duration_s"] = float(max(0.0, time.time() - started_at))
        return report

    for label_path in label_files:
        dir_path = label_path.parent
        doors_path = dir_path / "doors.json"
        if not doors_path.exists():
            continue

        try:
            labels_data = json.loads(label_path.read_text(encoding="utf-8"))
            doors_data = json.loads(doors_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        schema_v = labels_data.get("schema_version")
        confirmed_by_type: Dict[str, set[str]] = {t: set() for t in DOOR_TYPES}
        deleted: set[str] = set()
        rejected_by_type: Dict[str, set[str]] = {t: set() for t in DOOR_TYPES}

        if schema_v == 2:
            # Legacy: untyped confirmations are treated as swing.
            confirmed_by_type["swing"] = {str(x) for x in (labels_data.get("confirmed_ids") or []) if x is not None}
            deleted = {str(x) for x in (labels_data.get("deleted_ids") or []) if x is not None}
        elif schema_v == 3:
            cbt = labels_data.get("confirmed_by_type") or {}
            if isinstance(cbt, dict):
                for t in DOOR_TYPES:
                    confirmed_by_type[t] = {str(x) for x in (cbt.get(t) or []) if x is not None}
            deleted = {str(x) for x in (labels_data.get("deleted_ids") or []) if x is not None}
        elif schema_v == 4:
            cbt = labels_data.get("confirmed_by_type") or {}
            if isinstance(cbt, dict):
                for t in DOOR_TYPES:
                    confirmed_by_type[t] = {str(x) for x in (cbt.get(t) or []) if x is not None}
            rbt = labels_data.get("rejected_by_type") or {}
            if isinstance(rbt, dict):
                for t in DOOR_TYPES:
                    rejected_by_type[t] = {str(x) for x in (rbt.get(t) or []) if x is not None}
            deleted = {str(x) for x in (labels_data.get("deleted_ids") or []) if x is not None}
        else:
            # Ignore unknown schema versions.
            continue

        confirmed_any: set[str] = set().union(*[confirmed_by_type[t] for t in DOOR_TYPES])

        all_candidates = doors_data.get("candidates", doors_data.get("doors", [])) or []
        if not isinstance(all_candidates, list):
            continue

        for cand in all_candidates:
            if not isinstance(cand, dict):
                continue
            did_raw = cand.get("id")
            if did_raw is None:
                continue
            did = str(did_raw)

            cand_type = normalize_door_type(cand.get("type"), default=str(cand.get("type") or ""))
            if cand_type not in X_by_type:
                continue

            legacy_ids: set[str] = set()
            try:
                for lid in (cand.get("legacy_ids") or []):
                    if lid is None:
                        continue
                    legacy_ids.add(str(lid))
            except Exception:
                legacy_ids = set()

            pos_ids = confirmed_by_type.get(cand_type, set())
            neg_ids = set(deleted) | set(rejected_by_type.get(cand_type, set())) | (confirmed_any - set(pos_ids))

            is_pos = (did in pos_ids) or bool(legacy_ids & pos_ids)
            is_neg = (did in neg_ids) or bool(legacy_ids & neg_ids)
            if not is_pos and not is_neg:
                continue
            if is_pos and is_neg:
                continue

            feature_names = FEATURES_BY_TYPE.get(cand_type, [])
            feats = [float((cand.get("features") or {}).get(n, 0.0) or 0.0) for n in feature_names]
            X_by_type[cand_type].append(feats)
            y_by_type[cand_type].append(1 if is_pos else 0)

    # 2) Train one model per requested type.
    total_types = int(len(types))
    for i, t in enumerate(types):
        _emit({"stage": "start_type", "door_type": t, "i": int(i), "total": total_types})
        X_list = X_by_type.get(t) or []
        y_list = y_by_type.get(t) or []
        if not X_list:
            print(f"[{t}] No labeled samples; skipping.")
            report["by_type"][t] = {
                "door_type": t,
                "status": "skipped",
                "reason": "no_labeled_samples",
                "message": "No samples.",
                "num_samples": 0,
                "num_pos": 0,
                "num_neg": 0,
                "model_written": False,
                "output_path": str(output_dir / f"reweighter_{t}_v1.json"),
            }
            _emit({"stage": "end_type", "door_type": t, "i": int(i), "total": total_types, "status": "skipped"})
            continue

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        n = int(len(X))
        n_pos = int(np.sum(y))
        n_neg = int(n - n_pos)
        print(f"[{t}] Training on {n} samples ({n_pos} positive, {n_neg} negative)")

        if n < int(min_samples) or n_pos < int(min_pos) or n_neg < int(min_neg):
            print(
                f"[{t}] Not enough labeled data for a conservative update "
                f"(need N>={min_samples} and pos>={min_pos} and neg>={min_neg}). "
                f"Got N={n}, pos={n_pos}, neg={n_neg}. Not writing a new model."
            )
            report["by_type"][t] = {
                "door_type": t,
                "status": "skipped",
                "reason": "not_enough_samples",
                "message": f"Not enough samples (N={n}, pos={n_pos}, neg={n_neg}).",
                "num_samples": int(n),
                "num_pos": int(n_pos),
                "num_neg": int(n_neg),
                "model_written": False,
                "output_path": str(output_dir / f"reweighter_{t}_v1.json"),
            }
            _emit({"stage": "end_type", "door_type": t, "i": int(i), "total": total_types, "status": "skipped"})
            continue

        feature_names = list(FEATURES_BY_TYPE.get(t, []))
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        X_scaled = (X - means) / (stds + 1e-8)

        out_path = output_dir / f"reweighter_{t}_v1.json"
        prior = _load_prior_model(out_path)
        w0, b0, warm_ok = (np.zeros(X.shape[1], dtype=float), 0.0, False)
        if isinstance(prior, dict):
            w0, b0, warm_ok = _convert_prior_to_current_scaler(
                prior=prior, feature_names=feature_names, means_new=means, stds_new=stds
            )

        weights = np.array(w0, dtype=float)
        bias = float(b0)

        reg_lambda = float(base_reg_lambda) * (float(min_samples) / float(max(1, n)))

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

            dw = dw + reg_lambda * (weights - w0)
            db = float(db + float(reg_lambda_bias) * (bias - b0))

            weights -= learning_rate * dw
            bias -= learning_rate * db

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

        # Small, UI-friendly training metrics ("score").
        try:
            z_final = np.dot(X_scaled, weights) + bias
            probs_final = _sigmoid(z_final)
            pred = (probs_final >= 0.5).astype(int)
            train_acc = float(np.mean(pred == y))
            eps = 1e-9
            train_logloss = float(-np.mean(y * np.log(probs_final + eps) + (1 - y) * np.log(1 - probs_final + eps)))
        except Exception:
            train_acc = None
            train_logloss = None

        output_dir.mkdir(parents=True, exist_ok=True)
        model_data = {
            "schema_version": 2,
            "model_type": "logreg",
            "door_type": t,
            "feature_order": feature_names,
            "scaler": {
                "type": "zscore",
                "mean": means.tolist(),
                "std": stds.tolist(),
            },
            "weights": weights.tolist(),
            "bias": float(bias),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_samples": int(n),
            "num_pos": int(n_pos),
            "num_neg": int(n_neg),
            "train_accuracy": train_acc,
            "train_logloss": train_logloss,
            "warm_start_used": bool(warm_ok),
            "prior_path": str(out_path) if bool(warm_ok) else None,
            "reg_lambda": float(reg_lambda),
            "reg_lambda_bias": float(reg_lambda_bias),
        }

        out_path.write_text(json.dumps(model_data, indent=2), encoding="utf-8")
        print(f"[{t}] ✓ Model saved to {out_path}")
        report["models_written"] = int(report.get("models_written") or 0) + 1
        report["by_type"][t] = {
            "door_type": t,
            "status": "trained",
            "reason": None,
            "message": "Trained.",
            "num_samples": int(n),
            "num_pos": int(n_pos),
            "num_neg": int(n_neg),
            "model_written": True,
            "output_path": str(out_path),
            "warm_start_used": bool(warm_ok),
            "train_accuracy": train_acc,
            "train_logloss": train_logloss,
        }
        _emit({"stage": "end_type", "door_type": t, "i": int(i), "total": total_types, "status": "trained"})

    report["duration_s"] = float(max(0.0, time.time() - started_at))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit door reweighter from feedback")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Root artifacts directory")
    parser.add_argument("--out-dir", type=Path, default=Path("models"), help="Output models directory")
    parser.add_argument("--type", type=str, default=None, help="Optional door type to train (default: all)")
    
    args = parser.parse_args()
    fit_reweighter(args.artifacts, args.out_dir, door_type=args.type)


if __name__ == "__main__":
    main()


