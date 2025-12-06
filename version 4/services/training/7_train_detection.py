"""
STEP 7 — FIRE DETECTION MODEL TRAINING (PRODUCTION GRADE)

Algorithm:
✅ XGBoost w/ imbalance handling

Evaluation:
✅ Recall
✅ ROC-AUC
✅ PR-AUC
✅ Confusion Matrix
✅ Cross-Validation

✅ Threshold tuning
✅ Calibration
"""

import os
import joblib
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

from logger import get_logger

log = get_logger("STEP_7_DETECTION")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE  = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "detection_model.pkl")

THRESHOLD = 0.25


def main():

    log.info("===== FIRE DETECTION TRAINING STARTED =====")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    TARGET = "fire_occurred"

    X = df.drop(columns=[TARGET, "fire_cause", "timestamp", "split"], errors="ignore")
    X = X.select_dtypes(include="number")

    y = df[TARGET]

    log.info(f"Feature matrix shape: {X.shape}")

    # ---------- CLASS BALANCE WEIGHT ----------
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_weight = n_neg / max(n_pos, 1)

    print("Scale positive weight:", round(scale_weight, 2))

    # ---------- STRATIFIED SPLIT ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.25,
        random_state=42
    )

    # ---------- BASE MODEL ----------
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.90,
        scale_pos_weight=scale_weight,
        tree_method="hist",
        random_state=42
    )

    # ---------- CALIBRATION ----------
    model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv=3
    )

    model.fit(X_train, y_train)
    log.info("Training + Calibration completed.")

    # ---------- CROSS-VALIDATION ----------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_scores = cross_val_score(
        base_model,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    print("\nCV ROC-AUC:")
    print("Mean:", roc_scores.mean())
    print("Std :", roc_scores.std())

    # ---------- EVALUATION ----------
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print("\n=== FINAL THRESHOLD:", THRESHOLD, " ===")
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, y_prob)
    pr  = average_precision_score(y_test, y_prob)

    print("ROC-AUC:", roc)
    print("PR-AUC :", pr)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------- SAVE MODEL ----------
    joblib.dump(model, MODEL_FILE)

    print("\nSaved model →", MODEL_FILE)
    log.info("===== FIRE DETECTION TRAINING COMPLETE =====")


if __name__ == "__main__":
    main()
