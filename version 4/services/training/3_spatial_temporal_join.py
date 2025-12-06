"""
3_spatial_temporal_join.py

Simple idea:
- Create a 'grid_id' by rounding lat/lon (like putting the map into square boxes).
- Use (grid_id, date) to join:
    - wildfire base table
    - hotspots (satellite)
    - active fires (hectares)
    - fire occurrence (cause + spread)
- This is our spatial + temporal join:
    - spatial ≈ same grid square (within a few km)
    - temporal = same calendar 
    
Goal:

Create a grid cell ID using rounded lat/lon.
Use grid_id + date to join all datasets.
This is an easy way to say: “same place (within a few km) on the same day”.
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")


def add_grid_id(df: pd.DataFrame, lat_col: str, lon_col: str, decimals: int = 2):
    """
    Turn (lat, lon) into a simple grid cell.
    Example:
      lat=45.1234, lon=-75.9876  ->  lat_round=45.12, lon_round=-75.99
      grid_id = "45.12_-75.99"

    This means: points in the same cell are close to each other (~1–5 km).
    """
    df["lat_round"] = df[lat_col].round(decimals)
    df["lon_round"] = df[lon_col].round(decimals)
    df["grid_id"] = df["lat_round"].astype(str) + "_" + df["lon_round"].astype(str)
    return df


def main():
    # Load time-normalized datasets from step 2
    wildfire = pd.read_pickle(os.path.join(INTERIM_DIR, "step2_wildfire_time.pkl"))
    hotspots = pd.read_pickle(os.path.join(INTERIM_DIR, "step2_hotspots_time.pkl"))
    activefires = pd.read_pickle(os.path.join(INTERIM_DIR, "step2_activefires_time.pkl"))
    fire_occ = pd.read_pickle(os.path.join(INTERIM_DIR, "step2_fire_occ_time.pkl"))

    # Add grid_id to each dataset
    wildfire = add_grid_id(wildfire, "lat", "lon", decimals=2)
    hotspots = add_grid_id(hotspots, "lat", "lon", decimals=2)
    activefires = add_grid_id(activefires, "lat", "lon", decimals=2)
    fire_occ = add_grid_id(fire_occ, "lat", "lon", decimals=2)

    # Aggregate hotspots per grid per day (average intensity and counts)
    hotspots_agg = (
        hotspots.groupby(["grid_id", "date"])
        .agg(
            hotspot_count=("hfi", "count"),
            mean_hfi=("hfi", "mean"),
            mean_ros=("ros", "mean"),
            total_estarea=("estarea", "sum"),
        )
        .reset_index()
    )

    # Aggregate active fires per grid per day
    activefires_agg = (
        activefires.groupby(["grid_id", "date"])
        .agg(
            activefire_count=("hectares", "count"),
            total_hectares=("hectares", "sum"),
        )
        .reset_index()
    )

    # Fire occurrence: we only keep the key columns for labels
    fire_occ_labels = fire_occ[["grid_id", "date", "cause_main", "area_acres"]].copy()

    # Base table = wildfire (weather + indices)
    master = wildfire.copy()

    # Join hotspots on (grid_id, date)
    master = master.merge(hotspots_agg, on=["grid_id", "date"], how="left")

    # Join active fires
    master = master.merge(activefires_agg, on=["grid_id", "date"], how="left")

    # Join fire occurrence labels
    master = master.merge(fire_occ_labels, on=["grid_id", "date"], how="left")

    # Save joined table as interim
    master.to_pickle(os.path.join(INTERIM_DIR, "step3_spatial_temporal_join.pkl"))

    print("\n✅ Step 3 done: spatial + temporal join created.")
    print("Joined table shape:", master.shape)


if __name__ == "__main__":
    main()
