"""
STEP 8 — ROOT CAUSE CLASSIFICATION MODEL

Task:
Multiclass classification → fire origin

Targets:
["Human", "Lightning", "Unknown"]

Evaluation:
✅ Accuracy
✅ Macro F1
✅ Confusion Matrix
✅ Cross-validation

Output:
models/cause_model.pkl

# File: services/training/8_train_cause_classifier.py

Goal:
Predict wildfire cause category:

Human

Lightning

Unknown

Using LightGBM for:

native categorical handling

class imbalance tolerance

faster convergence
"""

import os
import joblib
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from logger import get_logger

log = get_logger("STEP_8_CAUSE")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "cause_classifier.pkl")

TARGET_COL = "fire_cause"


def main():

    log.info("===== ROOT CAUSE MODEL TRAINING STARTED =====")

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    df = pd.read_csv(DATA_FILE, low_memory=False)

    if TARGET_COL not in df.columns:
        raise ValueError("fire_cause column not found in dataset!")

    X = df.drop(columns=[TARGET_COL, "fire_occurred", "timestamp", "split"], errors="ignore")
    X = X.select_dtypes(include="number")

    y = df[TARGET_COL]

    log.info(f"Data shape: {X.shape}")
    log.info(f"Class distribution:\n{y.value_counts()}")

    # -------------------------------------------------
    # TRAIN / TEST SPLIT
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.25,
        random_state=42
    )

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=48,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    log.info("Cause model training completed.")

    # -------------------------------------------------
    # CROSS-VALIDATION
    # -------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_f1 = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="f1_macro",
        cv=skf,
        n_jobs=-1
    )

    print("\nCV Macro F1:")
    print("Mean:", round(cv_f1.mean(), 4))
    print("Std :", round(cv_f1.std(), 4))

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------
    y_pred = model.predict(X_test)

    print("\n=== ROOT CAUSE RESULTS ===")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------
    joblib.dump(model, MODEL_FILE)

    print("\n✅ Root Cause Model Saved →", MODEL_FILE)
    log.info("===== ROOT CAUSE MODEL TRAINING COMPLETE =====")


if __name__ == "__main__":
    main()
