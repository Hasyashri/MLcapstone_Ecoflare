# services/evaluation/14_compare_models.py

"""
STEP 14 — MULTI-MODEL COMPARISON

Compares:
- Detection model (binary)
- Cause model (multi-class)
- Spread model (regression)

Outputs:
- reports/model_comparison/model_comparison_summary.csv
What it does

Loads fire_training_master_clean.csv

Rebuilds three tasks on the same split:

fire_occurred (detection – binary)

fire_cause (cause – multi-class, if available)

spread_ha (spread – regression, only where spread_ha > 0)

Loads your 3 models:

models/detection_model.pkl

models/cause_model.pkl (if exists)

models/spread_regressor.pkl

Computes metrics and saves:

`reports/model_comparison/model_comparison_summary.csv`

with rows: detection, cause, spread.

"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# --- logger setup (matches steps 10–12) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from services.training.logger import get_logger

log = get_logger("STEP_14_MODEL_COMPARISON")

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")

DETECTION_MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "detection_model.pkl")
CAUSE_MODEL_FILE     = os.path.join(PROJECT_ROOT, "models", "cause_model.pkl")
SPREAD_MODEL_FILE    = os.path.join(PROJECT_ROOT, "models", "spread_regressor.pkl")

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "model_comparison")
os.makedirs(REPORT_DIR, exist_ok=True)


def build_features(df: pd.DataFrame, target_col: str):
    """
    Generic feature builder: drop target + obvious non-feature columns.
    DO NOT drop 'spread_ha' by default, because detection model used it as a feature.
    """
    drop_cols = [target_col, "timestamp", "split"]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include="number")
    return X


def safe_train_test_split(X, y, random_state=42, test_size=0.25, stratify=True):
    """Utility to keep split config consistent."""
    if stratify and len(np.unique(y)) > 1:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_detection(df: pd.DataFrame):
    log.info("=== Evaluating DETECTION model ===")

    if not os.path.exists(DETECTION_MODEL_FILE):
        log.warning("Detection model file not found, skipping detection.")
        return None

    y = df["fire_occurred"].astype(int)
    X = build_features(df, "fire_occurred")

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, stratify=True)

    model = joblib.load(DETECTION_MODEL_FILE)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.25).astype(int)  # default threshold used in your pipeline

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    acc     = accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    log.info(f"DET: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}, ACC={acc:.4f}")
    log.info(f"DET confusion (thr=0.25): TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return {
        "task": "detection",
        "type": "classification_binary",
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_score": f1,
        "accuracy": acc,
        "r2": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "n_test": len(y_test)
    }


def evaluate_cause(df: pd.DataFrame):
    """
    Cause model evaluation (multi-class).
    If you don't have cause_model.pkl yet, this will just log and return None.
    """
    log.info("=== Evaluating CAUSE model ===")

    if not os.path.exists(CAUSE_MODEL_FILE):
        log.warning("Cause model file not found, skipping cause.")
        return None

    if "fire_cause" not in df.columns:
        log.warning("Column fire_cause not found in dataset, skipping cause.")
        return None

    df_c = df.dropna(subset=["fire_cause"]).copy()
    if df_c.empty:
        log.warning("No rows with fire_cause present, skipping cause.")
        return None

    # Simple encoding: treat fire_cause as categorical labels
    y = df_c["fire_cause"].astype(str)
    X = build_features(df_c, "fire_cause")

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, stratify=True)

    model = joblib.load(CAUSE_MODEL_FILE)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    log.info(f"CAUSE: ACC={acc:.4f}, Macro-F1={macro_f1:.4f}")
    log.info("CAUSE classification report:\n" + classification_report(y_test, y_pred))

    return {
        "task": "cause",
        "type": "classification_multi",
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "f1_score": macro_f1,
        "accuracy": acc,
        "r2": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "n_test": len(y_test)
    }


def evaluate_spread(df: pd.DataFrame):
    log.info("=== Evaluating SPREAD model ===")

    if not os.path.exists(SPREAD_MODEL_FILE):
        log.warning("Spread model file not found, skipping spread.")
        return None

    if "spread_ha" not in df.columns:
        log.warning("Column spread_ha not found, skipping spread.")
        return None

    # Only fire rows with positive spread, same logic as step 9
    df_s = df[df["spread_ha"] > 0].copy()
    if df_s.empty:
        log.warning("No rows with spread_ha > 0, skipping spread.")
        return None

    y = df_s["spread_ha"].astype(float)
    X = build_features(df_s, "spread_ha")

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, stratify=False)

    model = joblib.load(SPREAD_MODEL_FILE)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    log.info(f"SPREAD: RMSE={rmse:.4f} ha, MAE={mae:.4f} ha, R2={r2:.4f}")

    return {
        "task": "spread",
        "type": "regression",
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "f1_score": np.nan,
        "accuracy": np.nan,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "n_test": len(y_test)
    }


def main():
    log.info("===== MODEL COMPARISON STARTED =====")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    log.info(f"Dataset shape: {df.shape}")

    rows = []

    det_row = evaluate_detection(df)
    if det_row is not None:
        rows.append(det_row)

    cause_row = evaluate_cause(df)
    if cause_row is not None:
        rows.append(cause_row)

    spread_row = evaluate_spread(df)
    if spread_row is not None:
        rows.append(spread_row)

    if not rows:
        log.warning("No models evaluated – nothing to save.")
        return

    summary_df = pd.DataFrame(rows)
    out_csv = os.path.join(REPORT_DIR, "model_comparison_summary.csv")
    summary_df.to_csv(out_csv, index=False)

    log.info(f"Saved comparison summary -> {out_csv}")
    log.info("===== MODEL COMPARISON COMPLETE =====")


if __name__ == "__main__":
    main()
