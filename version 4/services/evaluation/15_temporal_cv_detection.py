# services/evaluation/15_temporal_cv_detection.py

"""
STEP 15 — TEMPORAL CROSS-VALIDATION FOR DETECTION

Train on past years, test on a held-out year.
Evaluates seasonal / year-to-year robustness.

File: services/evaluation/15_temporal_cv_detection.py

What it does

Parses timestamp → year

For each year (except the earliest), it:

trains on all previous years

tests on that year

computes ROC-AUC & PR-AUC

Saves:

reports/detection_eval/temporal_cv_detection_by_year.csv
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

# logger setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from services.training.logger import get_logger

log = get_logger("STEP_15_TEMPORAL_CV_DET")

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "detection_model.pkl")

OUT_CSV = os.path.join(PROJECT_ROOT, "reports", "detection_eval", "temporal_cv_detection_by_year.csv")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)


def build_features(df: pd.DataFrame, target_col: str):
    drop_cols = [target_col, "timestamp", "split"]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include="number")
    return X


def main():
    log.info("===== TEMPORAL CV (DETECTION) STARTED =====")

    df = pd.read_csv(DATA_FILE, low_memory=False)
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column not found; temporal CV requires timestamps.")

    # Parse year
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year

    # we only use rows where target is defined
    df = df.dropna(subset=["fire_occurred"])
    df["fire_occurred"] = df["fire_occurred"].astype(int)

    model = joblib.load(MODEL_FILE)

    years = sorted(df["year"].unique())
    log.info(f"Years in dataset: {years}")

    rows = []

    # leave-one-year-out style
    for y in years[1:]:  # skip very first year (no past data to train on)
        train_df = df[df["year"] < y]
        test_df  = df[df["year"] == y]

        if train_df.empty or test_df.empty:
            log.warning(f"Skipping year {y}: train or test set empty.")
            continue

        X_train = build_features(train_df, "fire_occurred")
        y_train = train_df["fire_occurred"].values

        X_test  = build_features(test_df, "fire_occurred")
        y_test  = test_df["fire_occurred"].values

        # Copy of model so we don't mutate original
        from copy import deepcopy
        model_copy = deepcopy(model)

        log.info(f"Training on years < {y}, size={len(y_train)}, testing on year {y}, size={len(y_test)}")

        # Fit base estimator again for this fold
        model_copy.fit(X_train, y_train)
        y_prob = model_copy.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_prob)
        pr  = average_precision_score(y_test, y_prob)

        rows.append({
            "test_year": y,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "roc_auc": roc,
            "pr_auc": pr
        })

        log.info(f"Year {y}: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}")

    if not rows:
        log.warning("No temporal folds evaluated.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    log.info(f"Saved temporal CV results -> {OUT_CSV}")
    log.info("===== TEMPORAL CV (DETECTION) COMPLETE =====")


if __name__ == "__main__":
    main()
