"""
2_timestamp_normalization.py

GOAL (Simple Explanation):

Different wildfire datasets store dates and times in different formats such as:
    "2025-05-25 19:13:00"
    "07/18/2000 07:00:00 PM"
    "2025-05-25"

This script does FOUR important things:

1) Fixes messy column names (extra spaces, strange characters, upper/lower case).
2) Parses all timestamps safely even when formats are mixed.
3) Converts every timestamp into ONE single standard format:
       YYYY-MM-DD HH:MM:SS   (ISO standard)
4) Adds a universal "date" column used later for temporal joins.

Output files:
All processed datasets are saved into data/interim/ for the next pipeline steps.
"""

import os
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")

# -------------------------------------------------------------------
# SMART DATETIME PARSER
# -------------------------------------------------------------------

def smart_parse_datetime(value):
    """
    Safely parse mixed-format timestamps.

    Tries known formats first (fast),
    then falls back to Pandas' parser if needed (safe).

    This prevents warnings and avoids losing valid dates.
    """
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    formats = [
        "%Y-%m-%d %H:%M:%S",      # 2025-05-25 19:13:00
        "%m/%d/%Y %I:%M:%S %p",  # 07/18/2000 07:00:00 PM
        "%Y-%m-%d",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    # Final fallback: slow but safe
    return pd.to_datetime(value, errors="coerce")

# -------------------------------------------------------------------
# COLUMN CLEANER
# -------------------------------------------------------------------

def clean_columns(df):
    """
    Fix common CSV column problems:
    - Extra spaces
    - Invisible BOM characters
    - Random upper/lower casing
    """
    df.columns = (
        df.columns.astype(str)
                  .str.strip()
                  .str.replace("\ufeff", "", regex=False)
                  .str.lower()
    )
    return df

# -------------------------------------------------------------------
# MAIN PIPELINE STEP
# -------------------------------------------------------------------

def main():

    print("\nSTEP 2 – NORMALIZING ALL TIMESTAMPS")

    # --------------------------------------------------------
    # Load Step-1 datasets
    # --------------------------------------------------------

    wildfire   = pd.read_pickle(os.path.join(INTERIM_DIR, "step1_wildfire.pkl"))
    hotspots   = pd.read_pickle(os.path.join(INTERIM_DIR, "step1_hotspots.pkl"))
    activefire = pd.read_pickle(os.path.join(INTERIM_DIR, "step1_activefires.pkl"))
    fire_occ   = pd.read_pickle(os.path.join(INTERIM_DIR, "step1_fire_occ.pkl"))

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    wildfire   = clean_columns(wildfire)
    hotspots   = clean_columns(hotspots)
    activefire = clean_columns(activefire)
    fire_occ   = clean_columns(fire_occ)

    # --------------------------------------------------------
    # 1) WILDFIRE BASE DATASET
    # --------------------------------------------------------

    wildfire = wildfire.rename(
        columns={
            "latitude":  "lat",
            "longitude": "lon",
            "datetime":  "timestamp",
            "wildfire":  "wildfire_label",
        }
    )

    wildfire["timestamp"] = wildfire["timestamp"].apply(smart_parse_datetime)
    wildfire["timestamp"] = wildfire["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    wildfire["date"] = pd.to_datetime(wildfire["timestamp"]).dt.date

    wildfire.to_pickle(os.path.join(INTERIM_DIR, "step2_wildfire_time.pkl"))

    print("✅ Wildfire timestamps normalized.")

    # --------------------------------------------------------
    # 2) HOTSPOTS DATASET
    # --------------------------------------------------------

    hotspots = hotspots.rename(columns={"rep_date": "rep_timestamp"})

    hotspots["rep_timestamp"] = hotspots["rep_timestamp"].apply(smart_parse_datetime)
    hotspots["rep_timestamp"] = hotspots["rep_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    hotspots["date"] = pd.to_datetime(hotspots["rep_timestamp"]).dt.date

    hotspots.to_pickle(os.path.join(INTERIM_DIR, "step2_hotspots_time.pkl"))

    print("✅ Hotspot timestamps normalized.")

    # --------------------------------------------------------
    # 3) ACTIVE FIRES DATASET
    # --------------------------------------------------------

    # Auto-detect start-date column safely
    possible_cols = [
        "startdate",
        "start_date",
        "starttime",
        "start_time",
        "start datetime",
    ]

    start_col = None
    for col in possible_cols:
        if col in activefire.columns:
            start_col = col
            break

    if start_col is None:
        raise ValueError(
            f"❌ Cannot find a date column in ACTIVE FIRES dataset.\n"
            f"Columns found: {list(activefire.columns)}"
        )

    print(f"✅ Using active fires date column: '{start_col}'")

    activefire["start_timestamp"] = activefire[start_col].apply(smart_parse_datetime)
    activefire["start_timestamp"] = activefire["start_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    activefire["date"] = pd.to_datetime(activefire["start_timestamp"]).dt.date

    activefire.to_pickle(os.path.join(INTERIM_DIR, "step2_activefires_time.pkl"))

    print("✅ Active fire timestamps normalized.")

    # --------------------------------------------------------
    # 4) FIRE OCCURRENCE DATASET  (CAUSE + SPREAD)
    # --------------------------------------------------------

    fire_occ = fire_occ.rename(
        columns={
            "lat_dd":           "lat",
            "long_dd":          "lon",
            "ign_datetime":    "ign_timestamp",
            "esttotalacres":   "area_acres",
            "humanorlightning":"cause_main",
        }
    )

    fire_occ["ign_timestamp"] = fire_occ["ign_timestamp"].apply(smart_parse_datetime)
    fire_occ["ign_timestamp"] = fire_occ["ign_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    fire_occ["date"] = pd.to_datetime(fire_occ["ign_timestamp"]).dt.date

    fire_occ.to_pickle(os.path.join(INTERIM_DIR, "step2_fire_occ_time.pkl"))

    print("✅ Fire occurrence timestamps normalized.")

    print("\n🎉 STEP 2 COMPLETED SUCCESSFULLY!")
    print("✅ All timestamps standardized to format: YYYY-MM-DD HH:MM:SS")
    print("✅ 'date' columns generated for temporal joining")

# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
