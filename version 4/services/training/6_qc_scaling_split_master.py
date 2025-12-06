"""
STEP 6 — QUALITY CONTROL, CLEANING, SCALING & FINAL MASTER CREATION

PLAIN ENGLISH GOAL
--------------------------------------------------------
We now prepare the final training dataset for ML.

This script does the following:

1) ✅ Quality Control masking
   - Removes unreliable satellite readings caused by:
       • Cloud cover      (solar radiation = 0)
       • Smoke / haze     (visibility < 0.5)
       • Industrial flares (high HFI values)
       • Sensor noise     (fire detected but area = 0)
       • Physically invalid fuel/weather readings

2) ✅ Gap filling
   - Missing numbers filled with column medians

3) ✅ Outlier reporting
   - Uses IQR method
   - Counts only (NO DELETE)

4) ✅ Feature scaling
   - Converts all numeric features into standardized Z-scores

5) ✅ Leakage prevention
   - If there was NO fire → fire spread forced to 0

6) ✅ Train/Test split tag
   - 80% → train
   - 20% → test

7) ✅ Dataset size control
   - Samples final dataset to 100,000 rows
   - Preserves class balance of `fire_occurred`

FINAL OUTPUT
--------------------------------------------------------
Clean ML-ready dataset:

data/features/fire_training_master_clean.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from logger import get_logger

log = get_logger("STEP_6_QC")


# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "features")

RAW_MASTER_FILE = os.path.join(FEATURE_DIR, "fire_training_master_raw.csv")
FINAL_MASTER_FILE = os.path.join(FEATURE_DIR, "fire_training_master_clean.csv")

TARGET_ROWS = 100_000


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    
    print("\n============================")
    print("STEP 6 — FINAL DATA CLEANING")
    print("============================")

    # -----------------------------------------------
    # LOAD RAW MASTER DATA
    # -----------------------------------------------

    df = pd.read_csv(RAW_MASTER_FILE)
    print("Rows loaded:", len(df))

    log.info(f"Raw rows loaded: {len(df)}")
    # -----------------------------------------------
    # STEP 1 — QUALITY CONTROL MASKING
    # -----------------------------------------------

    print("\nApplying QC masking rules...")

    keep_mask = np.ones(len(df), dtype=bool)

    # Mask: CLOUD BLOCK
    if "srad" in df.columns:
        keep_mask &= df["srad"] != 0

    # Mask: SMOKE / HAZE
    if "vs" in df.columns:
        keep_mask &= df["vs"] >= 0.5

    # Mask: INDUSTRIAL FLARES / GLITCH
    if "hfi" in df.columns:
        keep_mask &= df["hfi"] <= 5000

    # Mask: FIRE DETECTED BUT NO AREA = NOISE
    if "estarea" in df.columns:
        keep_mask &= ~(
            (df["estarea"] == 0) &
            (df["fire_occurred"] == 1)
        )

    # Mask: PHYSICAL CONSTRAINTS
    for col in ["fm100", "erc", "fwi"]:
        if col in df.columns:
            keep_mask &= (df[col] > 0)

    before = len(df)
    df = df.loc[keep_mask].copy()
    after = len(df)

    print("QC rows removed:", before - after)
    print("Rows remaining:", after)
    log.info(f"QC rows removed: {before - after}")

    # -----------------------------------------------
    # STEP 2 — GAP FILL (MEDIAN IMPUTATION)
    # -----------------------------------------------

    print("\nApplying gap fill (numerical medians)...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    TARGET_COLS = ["fire_occurred", "spread_ha"]
    numeric_features = [col for col in numeric_cols if col not in TARGET_COLS]

    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())

    df["fire_cause"] = df["fire_cause"].fillna("Unknown")

    print("Gap fill complete.")

    log.info("Gap fill applied to numeric features.")
    # -----------------------------------------------
    # STEP 3 — OUTLIER REPORTING (IQR)
    # -----------------------------------------------

    print("\nOutlier analysis using IQR:")

    for col in numeric_features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr

        count = ((df[col] < low) | (df[col] > high)).sum()
        pct = round(count / len(df) * 100, 3)

        print(f"{col:15s} → {count:,} outliers ({pct}%)")
    log.info("Outlier analysis complete.")


    # -----------------------------------------------
    # STEP 4 — FEATURE SCALING
    # -----------------------------------------------

    print("\nScaling numeric features (Z-score)...")

    for col in numeric_features:
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - df[col].mean()) / std

    print("Scaling complete.")
    log.info("Feature scaling applied to numeric features.")

    # -----------------------------------------------
    # STEP 5 — LEAKAGE PREVENTION
    # -----------------------------------------------

    print("\nApplying leakage protection...")

    if "spread_ha" in df.columns:
        df.loc[df["fire_occurred"] == 0, "spread_ha"] = 0

    log.info("Leakage prevention applied to spread_ha.")
    # -----------------------------------------------
    # STEP 6 — TRAIN / TEST SPLITTING
    # -----------------------------------------------

    print("\nCreating train / test split...")

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=df["fire_occurred"]
        if df["fire_occurred"].nunique() > 1
        else None
    )

    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    log.info(f"Train rows: {len(train_idx)}, Test rows: {len(test_idx)}")
     # ---------------------------------------------
     # PROTECT RARE RECORDS FROM BEING LOST
     # ---------------------------------------------

    # Always retain rare cause-labeled rows
    rare_cause = df[df["fire_cause"].isin(["Human","Lightning"])]

    # Always retain large fires
    rare_spread = df[df["spread_ha"] > 0]

    # Combine protected rows
    protected = pd.concat([rare_cause, rare_spread]).drop_duplicates()
    print("Protected rare rows:", len(protected))
    df = pd.concat([df, protected]).drop_duplicates()
    log.info(f"Protected rare rows from sampling: {len(protected)}")
    # -----------------------------------------------
    # STEP 7 — DATASET SAMPLING
    # -----------------------------------------------

    """print("\nSampling to final size:", TARGET_ROWS)

    if len(df) > TARGET_ROWS:

        ratio = TARGET_ROWS / len(df)

        df = (
            df.groupby("fire_occurred", group_keys=False)
              .apply(
                  lambda g: g.sample(
                      max(1, int(len(g) * ratio)),
                      random_state=42
                  )
              )
        )

    print("Final dataset size:", len(df))"""

    print("\nSampling dataset to", TARGET_ROWS, "rows...")

    # Remove protected rows from pool
    remaining = df.drop(protected.index, errors="ignore")

    # How many rows still needed after keeping protected
    remaining_needed = max(0, TARGET_ROWS - len(protected))
    if remaining_needed > 0:

        sampled = (
            remaining
            .groupby("fire_occurred", group_keys=False)
            .apply(
                lambda x: x.sample(
                    max(1, int(len(x) * remaining_needed / len(remaining))),
                    random_state=42
                )
            )
        )

        df = pd.concat([sampled, protected])

    else:
        df = protected.copy()

    
    log.info(f"Final dataset size after sampling: {len(df)}")
    # -----------------------------------------------
    # SAVE FINAL MASTER DATASET
    # -----------------------------------------------

    df.to_csv(FINAL_MASTER_FILE, index=False)

    print("\n✅ FINAL MASTER DATASET CREATED")
    print("Saved to:", FINAL_MASTER_FILE)

    log.info("Final master dataset saved successfully.")
# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
