"""
1_data_loading_and_eda.py

Simple idea:
- Read all raw CSV / Excel files from data/static.
- Print shapes, columns, and missing values (basic EDA).
- Save cleaned copies as .pkl into data/interim so later scripts
  don't need to re-read CSVs again.
Goal:
Load each raw file, show basic info (rows, columns, missing values), 
and save them as simple .pkl (pickle) for the next steps.
"""

import os
import pandas as pd

# Find project root: go two folders up from this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Folders where data lives
STATIC_DIR = os.path.join(PROJECT_ROOT, "data", "static")
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
os.makedirs(INTERIM_DIR, exist_ok=True)


def quick_eda(df: pd.DataFrame, name: str) -> None:
    """Print basic information so a human can get a feel for the data."""
    print("\n" + "=" * 80)
    print(f"EDA for: {name}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Column names:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values per column (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))


def main():
    # 1) Wildfire base dataset (weather + indices + wildfire label)
    wildfire = pd.read_csv(os.path.join(STATIC_DIR, "Wildfire_Dataset.csv"))
    quick_eda(wildfire, "Wildfire_Dataset.csv")
    wildfire.to_pickle(os.path.join(INTERIM_DIR, "step1_wildfire.pkl"))

    # 2) Hotspots (satellite detected fire behavior)
    hotspots = pd.read_csv(os.path.join(STATIC_DIR, "hotspots.csv"))
    quick_eda(hotspots, "hotspots.csv")
    hotspots.to_pickle(os.path.join(INTERIM_DIR, "step1_hotspots.pkl"))

    # 3) Active fires (incident size + response)
    activefires = pd.read_csv(os.path.join(STATIC_DIR, "activefires.csv"))
    quick_eda(activefires, "activefires.csv")
    activefires.to_pickle(os.path.join(INTERIM_DIR, "step1_activefires.pkl"))

    # 4) Fire occurrence (cause + spread labels)
    fire_occ = pd.read_csv(os.path.join(STATIC_DIR, "fire-occurence.csv"))
    quick_eda(fire_occ, "fire-occurence.csv")
    fire_occ.to_pickle(os.path.join(INTERIM_DIR, "step1_fire_occ.pkl"))

    # 5) NFDB stats (optional, not used in joins yet)
    nfdb_stats = pd.read_excel(os.path.join(STATIC_DIR, "nfdb_stats.xlsx"))
    quick_eda(nfdb_stats, "nfdb_stats.xlsx")
    nfdb_stats.to_pickle(os.path.join(INTERIM_DIR, "step1_nfdb_stats.pkl"))

    print("\n✅ Step 1 done: basic EDA + saved interim .pkl files.")


if __name__ == "__main__":
    main()
