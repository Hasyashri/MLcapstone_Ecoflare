"""
STEP 10 — DETECTION MODEL EVALUATION & THRESHOLD TUNING
------------------------------------------------------

- Recreates the same train/test split as Step 7
- Computes ROC & Precision-Recall curves
- Explores thresholds for early warning configurations
- Saves curve data + threshold table to CSV for dashboards

FULLY STABILIZED VERSION:
✅ Feature mismatch fixed
✅ Confusion matrix safety
✅ Logger path stability
✅ ROC/PR dataframe shape safety
✅ Windows path compatibility
"""

# -------------------------------------------------------------------
# PATH BOOTSTRAP — REQUIRED FOR WINDOWS + SPACES IN FOLDERS
# -------------------------------------------------------------------
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
SERVICES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if SERVICES_DIR not in sys.path:
    sys.path.append(SERVICES_DIR)

from training.logger import get_logger
log = get_logger("STEP_10_DETECTION_EVAL")

# -------------------------------------------------------------------
# STANDARD IMPORTS
# -------------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_FILE  = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "detection_model.pkl")

EVAL_DIR = os.path.join(PROJECT_ROOT, "reports", "detection_eval")
os.makedirs(EVAL_DIR, exist_ok=True)

# -------------------------------------------------------------------

def build_features_and_target(df: pd.DataFrame):
    """
    EXACT feature replication from Step 7 training.

    ❗ spread_ha MUST remain included (model expects it)
    """
    target_col = "fire_occurred"
    y = df[target_col]

    drop_cols = [target_col, "fire_cause", "timestamp", "split"]

    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include="number")

    return X, y

# -------------------------------------------------------------------

def safe_confusion_counts(y_true, y_pred):
    """Guarantees (tn, fp, fn, tp) return even if a class is missing."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape != (2, 2):
        return 0, 0, 0, 0

    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

# -------------------------------------------------------------------

def threshold_sweep(y_true, y_prob, thresholds=None):
    """
    Evaluate precision / recall / F1 across thresholds
    without crashing for edge conditions.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tn, fp, fn, tp = safe_confusion_counts(y_true, y_pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        far       = fp / (tn + fp) if (tn + fp) > 0 else 0

        rows.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarm_rate": far,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })

    df = pd.DataFrame(rows).sort_values("threshold")
    return df

# -------------------------------------------------------------------

def choose_early_warning_thresholds(df):
    """
    Identify business operating points.
    """
    picks = {}

    # Best F1
    f1_star = df.loc[df["f1"].idxmax()]
    picks["BALANCED"] = f1_star.to_dict()

    # Max recall under acceptable FPR
    high_recall = df[df["recall"] >= .80]
    if not high_recall.empty:
        picks["HIGH_RECALL"] = high_recall.loc[
            high_recall["f1"].idxmax()
        ].to_dict()

    # Conservative warning system
    low_fpr = df[df["false_alarm_rate"] <= .10]
    if not low_fpr.empty:
        picks["LOW_FALSE_ALARM"] = low_fpr.loc[
            low_fpr["recall"].idxmax()
        ].to_dict()

    return pd.DataFrame.from_dict(picks, orient="index")

# -------------------------------------------------------------------

def main():
    log.info("===== DETECTION THRESHOLD EVALUATION STARTED =====")

    df = pd.read_csv(DATA_FILE, low_memory=False)
    X, y = build_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.25,
        random_state=42,
    )

    log.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    model = joblib.load(MODEL_FILE)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ----------------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------------
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    log.info(f"ROC-AUC = {roc_auc:.4f}")
    log.info(f"PR-AUC  = {pr_auc:.4f}")

    DEFAULT_THRESHOLD = 0.25
    y_pred_def = (y_prob >= DEFAULT_THRESHOLD).astype(int)

    log.info("\n=== DEFAULT THRESHOLD PERFORMANCE (0.25) ===")
    log.info("\n" + classification_report(y_test, y_pred_def))

    tn, fp, fn, tp = safe_confusion_counts(y_test, y_pred_def)
    np.savetxt(
        os.path.join(EVAL_DIR, "confusion_matrix_default_0_25.txt"),
        [[tn, fp], [fn, tp]],
        fmt="%d"
    )

    # ----------------------------------------------------------------
    # ROC CURVE SAVE (SAFE)
    # ----------------------------------------------------------------
    fpr, tpr, roc_thr = roc_curve(y_test, y_prob)
    roc_pad = np.full(len(fpr), np.nan)
    roc_pad[:len(roc_thr)] = roc_thr

    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": roc_pad
    }).to_csv(os.path.join(EVAL_DIR, "roc_curve_detection.csv"), index=False)

    # ----------------------------------------------------------------
    # PR CURVE SAVE (SAFE)
    # ----------------------------------------------------------------
    precision, recall, pr_thr = precision_recall_curve(y_test, y_prob)
    pr_pad = np.full(len(precision), np.nan)
    pr_pad[:len(pr_thr)] = pr_thr

    pd.DataFrame({
        "precision":  precision,
        "recall": recall,
        "threshold": pr_pad
    }).to_csv(os.path.join(EVAL_DIR, "pr_curve_detection.csv"), index=False)

    # ----------------------------------------------------------------
    # THRESHOLD SWEEP
    # ----------------------------------------------------------------
    thr_df = threshold_sweep(y_test, y_prob)
    thr_df.to_csv(
        os.path.join(EVAL_DIR, "threshold_sweep_detection.csv"),
        index=False
    )

    ew_df = choose_early_warning_thresholds(thr_df)
    ew_df.to_csv(
        os.path.join(EVAL_DIR, "early_warning_thresholds_detection.csv")
    )

    log.info("\n=== EARLY WARNING THRESHOLD CANDIDATES ===")
    log.info("\n" + ew_df.to_string())

    # ----------------------------------------------------------------
    # SUMMARY JSON
    # ----------------------------------------------------------------
    pd.Series({
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "n_test_samples": int(len(y_test))
    }).to_json(
        os.path.join(EVAL_DIR, "summary_metrics_detection.json")
    )

    log.info("===== DETECTION THRESHOLD EVALUATION COMPLETE =====")

# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
