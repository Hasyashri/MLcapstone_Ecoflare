# services/evaluation/16_ensemble_decision_rules.py

"""
STEP 16 — ENSEMBLE DECISION RULES FOR FIRE ALERTS

Compares:
1) Pure model-based detection (thr = 0.25 by default)
2) Hybrid rule:

   alert = (prob >= thr)
        OR (hotspot_count >= 2)
        OR (activefire_count >= 1)

Outputs:
- reports/detection_eval/ensemble_rules_summary.csv

Your idea:

“If more than 2 sources tell us there is fire → fire”

We’ll define an ensemble alert:

alert = (detection_prob >= threshold)
    OR (hotspot_count >= 2)
    OR (activefire_count >= 1)


File: services/evaluation/16_ensemble_decision_rules.py

Output:

reports/detection_eval/ensemble_rules_summary.csv
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from services.training.logger import get_logger

log = get_logger("STEP_16_ENSEMBLE_RULES")

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "detection_model.pkl")

OUT_CSV = os.path.join(PROJECT_ROOT, "reports", "detection_eval", "ensemble_rules_summary.csv")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)


def build_features(df: pd.DataFrame, target_col: str):
    drop_cols = [target_col, "timestamp", "split"]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include="number")
    return X


def compute_metrics(y_true, y_pred, name: str):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)

    log.info(f"[{name}] ACC={acc:.4f}, PREC={prec:.4f}, RECALL={rec:.4f}, F1={f1:.4f}, "
             f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return {
        "strategy": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def main():
    log.info("===== ENSEMBLE DECISION RULES STARTED =====")

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df.dropna(subset=["fire_occurred"])
    df["fire_occurred"] = df["fire_occurred"].astype(int)

    # require sensor fields to exist, else fill with zeros (safe fallback)
    for col in ["hotspot_count", "activefire_count"]:
        if col not in df.columns:
            log.warning(f"Column {col} not found; filling with 0.")
            df[col] = 0

    X = build_features(df, "fire_occurred")
    y = df["fire_occurred"].values

    model = joblib.load(MODEL_FILE)
    y_prob = model.predict_proba(X)[:, 1]

    default_thr = 0.25

    # Strategy 1: detection-only
    y_pred_model = (y_prob >= default_thr).astype(int)

    # Strategy 2: ensemble rule
    rule_hotspots = df["hotspot_count"] >= 2
    rule_active   = df["activefire_count"] >= 1

    y_pred_ensemble = (
        (y_prob >= default_thr) |
        rule_hotspots |
        rule_active
    ).astype(int)

    rows = []
    rows.append(compute_metrics(y, y_pred_model, "det_only_thr_0_25"))
    rows.append(compute_metrics(y, y_pred_ensemble, "ensemble_model+hotspots+activefire"))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    log.info(f"Saved ensemble comparison -> {OUT_CSV}")
    log.info("===== ENSEMBLE DECISION RULES COMPLETE =====")


if __name__ == "__main__":
    main()
