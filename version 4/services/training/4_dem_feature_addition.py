"""
4_dem_feature_addition.py

Goal (plain English):

- For each wildfire data point (latitude/longitude),
  extract elevation from the SRTM DEM raster (.tif).
- Add a clean 'elevation' column to the dataset.
- Fill missing values safely.

This script is written in a robust,
future-proof way that avoids pandas warnings
and safely handles DEM coverage gaps.
"""

import os
import numpy as np
import pandas as pd
import rasterio

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STATIC_DIR = os.path.join(PROJECT_ROOT, "data", "static")
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")

SRTM_PATH = os.path.join(STATIC_DIR, "srtm_03_08.tif")

# --------------------------------------------------
# DEM SAMPLING FUNCTION
# --------------------------------------------------

def sample_elevation(latitudes, longitudes, raster_path):
    """
    Sample elevation (meters) for each lat/lon point
    from a DEM raster.

    If a coordinate is outside the raster or nodata,
    returns NaN.
    """
    elevations = []

    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        nodata = src.nodata

        for lat, lon in zip(latitudes, longitudes):

            try:
                # Convert lon/lat ➜ raster row/col
                row, col = src.index(lon, lat)

                value = raster[row, col]

                # Handle nodata pixels
                if nodata is not None and value == nodata:
                    elevations.append(np.nan)
                else:
                    elevations.append(float(value))

            except Exception:
                # Any coordinate outside raster
                elevations.append(np.nan)

    return np.array(elevations)

# --------------------------------------------------
# MAIN PIPELINE STEP
# --------------------------------------------------

def main():

    print("\nSTEP 4 – ADDING DEM ELEVATION FEATURE")

    # --------------------------------------------------
    # LOAD JOINED DATASET FROM STEP 3
    # --------------------------------------------------

    master_path = os.path.join(INTERIM_DIR, "step3_spatial_temporal_join.pkl")
    master = pd.read_pickle(master_path)

    print("Records to sample:", len(master))

    # --------------------------------------------------
    # SAMPLE DEM
    # --------------------------------------------------

    master["elevation"] = sample_elevation(
        master["lat"].values,
        master["lon"].values,
        SRTM_PATH,
    )

    # --------------------------------------------------
    # SAFE GAP FILL
    # --------------------------------------------------

    valid_count = master["elevation"].notna().sum()

    print("Valid elevation samples:", valid_count)

    if valid_count > 0:
        median_value = master["elevation"].median()
        master["elevation"] = master["elevation"].fillna(median_value)
        print(f"Missing elevations filled with median = {round(median_value,2)}")
    else:
        # Rare edge case: if no valid DEM points match
        print("⚠️ No valid DEM matches found — setting elevation to 0")
        master["elevation"] = 0.0

    # --------------------------------------------------
    # SAVE INTERIM FILE
    # --------------------------------------------------

    out_path = os.path.join(INTERIM_DIR, "step4_with_elevation.pkl")
    master.to_pickle(out_path)

    print("\n✅ STEP 4 COMPLETED SUCCESSFULLY")
    print("Examples of extracted elevations:")
    print(master["elevation"].head().tolist())

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
"""
4_dem_feature_addition.py

GOAL (plain English):
---------------------
Originally, this step was meant to sample elevation from a DEM file
(srtm_03_08.tif). However, that DEM tile does NOT overlap the geographic
extent of our wildfire records (Canada/US), so all sampled values would be
NaN.

Instead of breaking the pipeline, we:

- Keep this step in the pipeline for clarity.
- Add an 'elevation' column as a placeholder (set to 0.0).
- Clearly document this limitation in code and in the report.

This keeps the pipeline structure clean (6 steps) and makes it easy to
swap in a correct DEM later without changing other steps.

INPUT:
    data/interim/step3_spatial_temporal_join.pkl

OUTPUT:
    data/interim/step4_with_elevation.pkl
"""