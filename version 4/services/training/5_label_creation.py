"""
STEP 5 — LABEL CREATION + TIMESTAMP FEATURES + IMBALANCE CHECK

This step:
-----------------------------------------------------
1. Loads merged data from Step 4
2. Creates ML TIMESTAMP FEATURES
      - month
      - dayofyear
      - week
3. Creates 3 ML TARGET LABELS:

   A) fire_occurred  (DETECTION TARGET)
      From wildfire.csv -> "Yes/No" or "1/0"
      Converted to:
         1 = Fire
         0 = No Fire

   B) fire_cause (CLASSIFICATION TARGET)
      From fire-occurence.csv -> "Human" / "Lightning"
      Final labels:
         Human / Lightning / Unknown

   C) spread_ha (REGRESSION TARGET)
      From fire-occurence.csv -> acres → hectares

4. Checks dataset CLASS IMBALANCE
5. Warns if class is too extreme for RandomForest

OUTPUT:
-----------------------------------------------------
data/features/fire_training_master_raw.csv
"""

import os
import pandas as pd
from collections import Counter
from logger import get_logger

log = get_logger("STEP_5_LABELS")

# -----------------------------------------------------
# PATHS
# -----------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INTERIM_DIR  = os.path.join(PROJECT_ROOT, "data", "interim")
FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "features")
os.makedirs(FEATURE_DIR, exist_ok=True)


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------

def main():

    print("\nSTEP 5 — CREATING LABELS + TIME FEATURES")

    # -------------------------------------------------
    # LOAD MERGED DATA
    # -------------------------------------------------

    src_file = os.path.join(INTERIM_DIR, "step4_with_elevation.pkl")
    df = pd.read_pickle(src_file)
    log.info(f"Loaded Step-4 dataset: {len(df)} rows")

    # -------------------------------------------------
    # TIMESTAMP FEATURE ENGINEERING
    # -------------------------------------------------
    print("\nCreating timestamp features...")

    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], errors="coerce")

        df["month"]     = dt.dt.month
        df["dayofyear"] = dt.dt.dayofyear
        df["week"]      = dt.dt.isocalendar().week.astype("Int64")

        log.info("Time features added: month, dayofyear, week")

    else:
        print("⚠ No timestamp column found. Skipping time features.")
        log.warning("Timestamp column missing – no time features created")

    # -------------------------------------------------
    # LABEL 1 — FIRE OCCURRED
    # -------------------------------------------------

    wildfire_series = df.get("Wildfire", df.get("wildfire_label", "No"))

    # Convert Yes/No/1/0/True → binary numeric label
    df["fire_occurred"] = (
        wildfire_series
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "1", "true"])
            .astype(int)
    )

    log.info(f"fire_occurred distribution:\n{df['fire_occurred'].value_counts()}")

    # -------------------------------------------------
    # LABEL 2 — FIRE CAUSE
    # -------------------------------------------------

    cause_series = df.get("HumanOrLightning", df.get("cause_main", "Unknown"))

    df["fire_cause"] = (
        cause_series
           .astype(str)
           .str.strip()
           .str.title()
           .fillna("Unknown")
    )

    # Only allow true categories
    df.loc[~df["fire_cause"].isin(["Human", "Lightning"]), "fire_cause"] = "Unknown"

    log.info(f"fire_cause distribution:\n{df['fire_cause'].value_counts()}")

    # -------------------------------------------------
    # LABEL 3 — FIRE SPREAD (hectares)
    # -------------------------------------------------

    acres = df.get("EstTotalAcres", df.get("area_acres", 0))
    acres_numeric = pd.to_numeric(acres, errors="coerce").fillna(0)

    df["spread_ha"] = (acres_numeric * 0.404686).clip(lower=0)

    log.info("spread_ha created")

    # -------------------------------------------------
    # CLASS IMBALANCE CHECK
    # -------------------------------------------------

    print("\nChecking class imbalance...")

    dist = Counter(df["fire_occurred"])
    total = sum(dist.values())

    ratio = round(dist.get(1,0) / max(dist.get(0,1),1), 6)

    print("Class counts:", dist)
    print("Positive/Negative ratio:", ratio)

    if ratio < 0.02:
        print("\n🚨 EXTREME CLASS IMBALANCE DETECTED!")
        print("RandomForest alone may UNDERPERFORM.")
        print("Recommended algorithms:")
        print("  ✅ XGBoost")
        print("  ✅ LightGBM")
        print("  ✅ CatBoost")
        print("  ✅ BalancedRandomForest")
        log.warning("Severe imbalance detected – advanced imbalance algorithms recommended.")
    else:
        print("\n✅ Class distribution acceptable for RandomForest")

    # -------------------------------------------------
    # SAVE RAW MASTER DATASET
    # -------------------------------------------------

    out_csv = os.path.join(FEATURE_DIR, "fire_training_master_raw.csv")
    df.to_csv(out_csv, index=False)

    log.info("Raw master dataset saved.")
    print("\n✅ STEP 5 COMPLETE")
    print("Saved →", out_csv)
    print("Total records:", len(df))


# -----------------------------------------------------

if __name__ == "__main__":
    main()
