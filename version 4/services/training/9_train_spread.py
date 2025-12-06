"""
STEP 9 — FIRE SPREAD PREDICTION MODEL (REGRESSION)

Objective:
----------------------------------
Predict wildfire spread severity (hectares burned)
ONLY on confirmed wildfire events (fire_occurred = 1).

Target:
----------------------------------
spread_ha  (continuous regression target)

Model:
----------------------------------
✅ LightGBM Regressor
✅ Log-transformed target to manage heavy skew
✅ Cross-validated RMSE
✅ Test-set evaluation: RMSE, MAE, R²

Outputs:
----------------------------------
- models/spread_regressor.pkl
- Console metrics report
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

from logger import get_logger

# -------------------------------------------------------
# LOGGER
# -------------------------------------------------------
log = get_logger("STEP_9_SPREAD")

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE  = os.path.join(PROJECT_ROOT, "data", "features", "fire_training_master_clean.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "spread_regressor.pkl")

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():

    log.info("===== FIRE SPREAD REGRESSION TRAINING STARTED =====")

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------
    df = pd.read_csv(DATA_FILE)

    # ---------------------------------------------------
    # FILTER = ONLY REAL FIRE EVENTS
    # ---------------------------------------------------
    df_fire = df[df["fire_occurred"] == 1].copy()
    log.info(f"Fire-only dataset size: {len(df_fire)} records")

    if len(df_fire) < 200:
        raise RuntimeError("Too few fire records for stable regression training.")

    # ---------------------------------------------------
    # FEATURE MATRIX
    # ---------------------------------------------------
    DROP_COLS = [
        "fire_occurred",
        "fire_cause",
        "timestamp",
        "split"
    ]

    X = df_fire.drop(columns=DROP_COLS, errors="ignore")
    X = X.select_dtypes(include=np.number)

    y = df_fire["spread_ha"].clip(lower=0)

    log.info(f"Feature count: {X.shape[1]}")

    # ---------------------------------------------------
    # LOG TRANSFORM (spread is extremely skewed)
    # ---------------------------------------------------
    y_log = np.log1p(y)

    # ---------------------------------------------------
    # TRAIN / TEST SPLIT
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log,
        test_size=0.20,
        random_state=42
    )

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------
    model = LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=40,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1
    )

    # ---------------------------------------------------
    # CROSS VALIDATION
    # ---------------------------------------------------
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rmse = -cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )

    print("\nCV RMSE (log-space):")
    print("Mean:", round(cv_rmse.mean(), 4))
    print("Std :", round(cv_rmse.std(), 4))

    log.info(f"CV RMSE Mean={cv_rmse.mean():.4f} Std={cv_rmse.std():.4f}")

    # ---------------------------------------------------
    # FINAL TRAIN
    # ---------------------------------------------------
    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # TEST EVALUATION
    # ---------------------------------------------------
    preds_log = model.predict(X_test)

    y_true = np.expm1(y_test)
    y_pred = np.expm1(preds_log)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print("\n=== FIRE SPREAD REGRESSION RESULTS ===")
    print(f"RMSE (ha): {round(rmse, 2)}")
    print(f"MAE  (ha): {round(mae, 2)}")
    print(f"R²        : {round(r2, 4)}")

    log.info(f"Test RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

    # ---------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------
    joblib.dump(model, MODEL_FILE)

    print("\n✅ Spread Model Saved →", MODEL_FILE)
    log.info("===== FIRE SPREAD MODEL TRAINING COMPLETE =====")


# -------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------
if __name__ == "__main__":
    main()
