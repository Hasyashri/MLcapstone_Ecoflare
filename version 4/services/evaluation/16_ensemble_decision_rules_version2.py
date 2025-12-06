"""
STEP 16 — ENSEMBLE DECISION RULES v2 (EARLY WARNING)

Purpose
-------------------------------------------------------
Combine multiple weak & strong signals into operational
fire decisions using rule-based ensemble voting.

Signals used:

✅ Detection model probability
✅ Hotspot clusters (satellite detections)
✅ Active fire reports
✅ Weather ignition risk (Wind + VPD)
✅ Predicted spread size

Ensemble Strategies Evaluated
-------------------------------------------------------
1) STRICT_CONFIRMATION
   - High confidence only (very low false alarms)

2) BALANCED_RESPONSE
   - Model threshold + hotspot confirmation

3) EARLY_WARNING
   - OR-logic across multi-signals
   - Maximizes recall for safety systems

Outputs:
-------------------------------------------------------
CSV: reports/detection_eval/ensemble_rules_comparison_v2.csv
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# ✅ FIX LOGGER IMPORT FOR WINDOWS DIRECT SCRIPT RUN
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from services.training.logger import get_logger


log = get_logger("STEP_16_ENSEMBLE_V2")

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")
DETECTION_MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "detection_model.pkl")
SPREAD_MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "spread_regressor.pkl")

EVAL_DIR = os.path.join(PROJECT_ROOT, "reports", "detection_eval")
os.makedirs(EVAL_DIR, exist_ok=True)

OUT_CSV = os.path.join(EVAL_DIR, "ensemble_rules_comparison_v2.csv")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def build_features(df):
    """
    Feature set must MATCH training exactly.
    Do NOT drop spread_ha — the detection model expects it.
    """
    drop = ["fire_occurred", "fire_cause", "timestamp", "split"]

    X = df.drop(columns=drop, errors="ignore")
    X = X.select_dtypes(include="number")

    return X



def metrics(name, y_true, preds):
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision+recall) else 0
    acc       = (tp + tn) / (tn + fp + fn + tp)

    log.info(f"[{name}] ACC={acc:.4f}, PREC={precision:.4f}, "
             f"RECALL={recall:.4f}, F1={f1:.4f}, "
             f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return {
        "rule": name,
        "accuracy": round(acc,4),
        "precision": round(precision,4),
        "recall": round(recall,4),
        "f1": round(f1,4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    log.info("===== ENSEMBLE DECISION RULES v2 STARTED =====")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    TARGET = "fire_occurred"
    y_true = df[TARGET].values

    X = build_features(df)

    # ----------------------------------------------
    # LOAD MODELS
    # ----------------------------------------------

    det_model = joblib.load(DETECTION_MODEL_FILE)
    spread_model = None
    try:
        spread_model = joblib.load(SPREAD_MODEL_FILE)
    except:
        log.warning("Spread model file missing. Spread signals disabled.")

    # ----------------------------------------------
    # MODEL PROBABILITIES
    # ----------------------------------------------

    det_probs = det_model.predict_proba(X)[:, 1]

    spread_pred = np.zeros(len(df))
    if spread_model is not None:
        try:
            spread_pred = spread_model.predict(X)
        except:
            log.warning("Spread model prediction failed. Skipping spread risk.")

    # ----------------------------------------------
    # AUXILIARY SIGNALS
    # ----------------------------------------------

    hotspot_signal = (df.get("hotspot_count", 0) >= 2).astype(int)
    activefire_signal = (df.get("activefire_count", 0) >= 1).astype(int)

    # Weather ignition rule
    wind = df.get("vs", pd.Series(0, index=df.index))
    vpd = df.get("vpd", pd.Series(0, index=df.index))

    weather_risk = ((wind >= 25) & (vpd >= 1.5)).astype(int)

    spread_risk = (spread_pred >= 50).astype(int)

    results = []

    # ----------------------------------------------
    # STRATEGY 1 — STRICT CONFIRMATION
    # ----------------------------------------------

    strict_preds = (det_probs >= 0.25).astype(int)
    results.append(metrics("STRICT_CONFIRMATION", y_true, strict_preds))

    # ----------------------------------------------
    # STRATEGY 2 — BALANCED RESPONSE
    # ----------------------------------------------

    balanced_preds = (
        ((det_probs >= 0.15) & ((hotspot_signal + activefire_signal) >= 1))
    ).astype(int)

    results.append(metrics("BALANCED_RESPONSE", y_true, balanced_preds))

    # ----------------------------------------------
    # STRATEGY 3 — EARLY WARNING (FULL ENSEMBLE)
    # ----------------------------------------------

    early_warning_preds = (
        (det_probs >= 0.10) |
        (hotspot_signal >= 1) |
        (activefire_signal >= 1) |
        (weather_risk == 1) |
        (spread_risk == 1)
    ).astype(int)

    results.append(metrics("EARLY_WARNING", y_true, early_warning_preds))

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)

    log.info(f"Saved ensemble summary -> {OUT_CSV}")
    log.info("===== ENSEMBLE DECISION RULES v2 COMPLETE =====")


if __name__ == "__main__":
    main()
