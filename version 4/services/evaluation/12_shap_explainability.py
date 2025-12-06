"""
STEP 12 — SHAP EXPLAINABILITY (XGBOOST NATIVE, STABLE ON WINDOWS)

We use XGBoost's built-in SHAP algorithm via:
    booster.predict(dmatrix, pred_contribs=True)

Outputs:
-------------------------------------------------
reports/shap/
 - shap_feature_importance.csv   (global)
 - shap_summary_bar.png          (global bar plot)
 - shap_top_records.csv          (local top-200 events)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# -------------------------------------------------
# LOGGING IMPORT (path fix)
# -------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
ROOT_SERVICES = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.append(ROOT_SERVICES)

from training.logger import get_logger

log = get_logger("STEP_12_SHAP")

# -------------------------------------------------
# PATHS
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

DATA_FILE  = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "detection_model.pkl")

SHAP_DIR = os.path.join(PROJECT_ROOT, "reports", "shap")
os.makedirs(SHAP_DIR, exist_ok=True)


# -------------------------------------------------
# HELPER: GET TRAINED XGB CLASSIFIER
# -------------------------------------------------
def extract_trained_xgb(calibrated_model):
    """
    Our detection model is a CalibratedClassifierCV around XGBClassifier.
    This extracts the trained XGBClassifier from it.
    """
    if hasattr(calibrated_model, "calibrated_classifiers_"):
        return calibrated_model.calibrated_classifiers_[0].estimator
    return calibrated_model


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    log.info("===== SHAP EXPLAINABILITY STARTED =====")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(DATA_FILE, low_memory=False)
    log.info(f"Full dataset shape: {df.shape}")

    # Load model
    calibrated = joblib.load(MODEL_FILE)
    xgb_clf = extract_trained_xgb(calibrated)
    booster = xgb_clf.get_booster()

    feature_names = booster.feature_names
    log.info(f"Booster feature count: {len(feature_names)}")

    # Build X using exactly the features the model was trained on
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        log.warning(f"Some model features not found in data columns: {missing_cols}")

    valid_features = [c for c in feature_names if c in df.columns]
    X = df[valid_features].copy()

    log.info(f"X shape after aligning with booster features: {X.shape}")

    # -----------------------------
    # SAMPLE FOR GLOBAL EXPLAINABILITY
    # -----------------------------
    sample_size = min(1000, len(X))
    X_sample = X.sample(sample_size, random_state=42)
    log.info(f"Sample size for SHAP (global): {X_sample.shape[0]} rows")

    # Create DMatrix with the *same* feature names/order
    dmatrix_sample = xgb.DMatrix(X_sample, feature_names=valid_features)

    # -----------------------------
    # GLOBAL SHAP VALUES (XGBOOST NATIVE)
    # -----------------------------
    # pred_contribs=True returns SHAP values (one extra column for bias term)
    contribs = booster.predict(dmatrix_sample, pred_contribs=True)
    contribs = np.array(contribs)

    # Last column is bias term; drop it for feature-wise importance
    shap_values = contribs[:, :-1]

    log.info(f"SHAP matrix shape (global): {shap_values.shape}")

    # Mean |SHAP| per feature
    mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({
            "feature": valid_features,
            "mean_abs_shap": mean_abs
        })
        .sort_values("mean_abs_shap", ascending=False)
    )

    imp_path = os.path.join(SHAP_DIR, "shap_feature_importance.csv")
    importance_df.to_csv(imp_path, index=False)
    log.info(f"Saved global importance -> {imp_path}")

    # -----------------------------
    # GLOBAL BAR PLOT
    # -----------------------------
    top_k = min(20, len(valid_features))
    top_imp = importance_df.head(top_k).iloc[::-1]  # reverse for nicer barh

    plt.figure(figsize=(10, 6))
    plt.barh(top_imp["feature"], top_imp["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global Feature Importance (XGBoost SHAP)")
    plt.tight_layout()

    plot_path = os.path.join(SHAP_DIR, "shap_summary_bar.png")
    plt.savefig(plot_path, dpi=140)
    plt.close()
    log.info(f"Saved global SHAP bar plot -> {plot_path}")

    # -----------------------------
    # LOCAL EXPLANATIONS FOR TOP-RISK EVENTS
    # -----------------------------
    # Use calibrated model probas (same as detection pipeline)
    probs = calibrated.predict_proba(X)[:, 1]
    df["ml_fire_prob"] = probs

    top_n = 200
    top_df = df.sort_values("ml_fire_prob", ascending=False).head(top_n).copy()
    X_local = top_df[valid_features]

    dmatrix_local = xgb.DMatrix(X_local, feature_names=valid_features)
    contribs_local = booster.predict(dmatrix_local, pred_contribs=True)
    contribs_local = np.array(contribs_local)[:, :-1]  # drop bias

    # For each record, find top 3 features by |SHAP|
    abs_local = np.abs(contribs_local)
    top_feature_strings = []
    for row_vals in abs_local:
        idx = np.argsort(row_vals)[-3:]  # 3 strongest drivers
        feat_list = [valid_features[i] for i in idx]
        top_feature_strings.append(", ".join(feat_list))

    top_df["top_features"] = top_feature_strings

    # Choose a compact export set (you can add more columns later)
    export_cols = [c for c in ["lat", "lon", "ml_fire_prob", "top_features"] if c in top_df.columns]

    local_path = os.path.join(SHAP_DIR, "shap_top_records.csv")
    top_df[export_cols].to_csv(local_path, index=False)
    log.info(f"Saved local top-record explanations -> {local_path}")

    log.info("===== SHAP EXPLAINABILITY COMPLETE =====")


if __name__ == "__main__":
    main()
