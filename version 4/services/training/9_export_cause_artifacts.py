"""
STEP 9 — EXPORT CAUSE MODEL EVALUATION ARTIFACTS

This script:
✅ Loads trained cause model (cause_classifier.pkl)
✅ Reloads full dataset
✅ Recreates train/test split EXACTLY as used in training
✅ Recomputes evaluation metrics
✅ Exports dashboard artifacts WITHOUT retraining

Outputs:
- reports/cause_eval/summary_metrics_cause.json
- reports/cause_eval/confusion_matrix_cause.txt
- reports/cause_eval/class_report_cause.csv

Safe to run multiple times.
No changes to original training code required.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "cause_classifier.pkl")

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "cause_eval")
os.makedirs(REPORT_DIR, exist_ok=True)

SUMMARY_JSON = os.path.join(REPORT_DIR, "summary_metrics_cause.json")
CONFUSION_TXT = os.path.join(REPORT_DIR, "confusion_matrix_cause.txt")
CLASSREPORT_CSV = os.path.join(REPORT_DIR, "class_report_cause.csv")

TARGET_COL = "fire_cause"


def main():

    print("===== EXPORTING CAUSE MODEL ARTIFACTS =====")

    # -------------------------------------------------------
    # LOAD DATA & MODEL
    # -------------------------------------------------------
    df = pd.read_csv(DATA_FILE, low_memory=False)

    if TARGET_COL not in df.columns:
        raise RuntimeError("fire_cause column not found!")

    X = df.drop(columns=[TARGET_COL, "fire_occurred", "timestamp", "split"], errors="ignore")
    X = X.select_dtypes(include="number")

    y = df[TARGET_COL]

    if not os.path.exists(MODEL_FILE):
        raise RuntimeError("cause_classifier.pkl not found! Train model first.")

    model = joblib.load(MODEL_FILE)
    print("✅ Loaded cause model")

    # -------------------------------------------------------
    # RECREATE TRAIN/TEST SPLIT
    # Must match original training split exactly
    # -------------------------------------------------------
    _, X_test, _, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.25,
        random_state=42
    )

    # -------------------------------------------------------
    # PREDICT
    # -------------------------------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------------------------------
    # METRICS
    # -------------------------------------------------------
    cls_report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    accuracy = round(cls_report["accuracy"], 4)
    macro_f1 = round(cls_report["macro avg"]["f1-score"], 4)
    weighted_f1 = round(cls_report["weighted avg"]["f1-score"], 4)

    classes = list(cm.shape)
    labels = sorted(y.unique().tolist())

    # -------------------------------------------------------
    # SAVE SUMMARY JSON
    # -------------------------------------------------------
    summary = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_test": len(y_test),
        "classes": labels,
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=4)

    print("✅ Summary JSON exported")

    # -------------------------------------------------------
    # SAVE CONFUSION MATRIX TXT
    # -------------------------------------------------------
    with open(CONFUSION_TXT, "w") as f:
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")

    print("✅ Confusion Matrix TXT exported")

    # -------------------------------------------------------
    # SAVE CLASS REPORT CSV
    # -------------------------------------------------------
    rows = []

    for cls in labels:
        rows.append({
            "class": cls,
            "precision": round(cls_report[cls]["precision"], 4),
            "recall": round(cls_report[cls]["recall"], 4),
            "f1": round(cls_report[cls]["f1-score"], 4),
            "support": int(cls_report[cls]["support"])
        })

    pd.DataFrame(rows).to_csv(CLASSREPORT_CSV, index=False)

    print("✅ Class Report CSV exported")

    print("\n🎉 CAUSE MODEL ARTIFACT EXPORT COMPLETE!")
    print("-----------------------------------------")
    print("Saved files:")
    print("-", SUMMARY_JSON)
    print("-", CONFUSION_TXT)
    print("-", CLASSREPORT_CSV)


if __name__ == "__main__":
    main()
