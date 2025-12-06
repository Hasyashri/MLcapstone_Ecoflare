# =============================================================
# File: services/features/dynamic_feature_weighting.py
# Purpose: Assign dynamic weights to each data source
# Idea:
#   - Higher coverage → higher weight
#   - More missing values → lower weight
#   - Later: can include model performance feedback
# =============================================================

import pandas as pd
from typing import Dict

from services.management.logging import get_logger

logger = get_logger("DynamicFeatureWeighting")


def compute_source_weight(
    coverage_fraction: float,
    missing_fraction: float,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """
    Compute a simple dynamic weight for one source.

    Formula:
        weight = alpha * coverage_fraction + beta * (1 - missing_fraction)

    Where:
      - coverage_fraction ∈ [0, 1]  (from QC masking)
      - missing_fraction  ∈ [0, 1]  (fraction of NaNs)
      - alpha, beta tune how much you care about each part.

    Returns:
      weight ∈ [0, 1] (clipped).
    """
    coverage = max(0.0, min(1.0, coverage_fraction))
    missing = max(0.0, min(1.0, missing_fraction))

    weight = alpha * coverage + beta * (1.0 - missing)
    weight = max(0.0, min(1.0, weight))

    return weight


def summarize_missing_fraction(df: pd.DataFrame) -> float:
    """
    Compute overall missing fraction for a DataFrame.

    Returns:
      missing_fraction = (# NaN values) / (# all values)
    """
    if df is None or df.empty:
        return 1.0

    total = df.size
    missing = df.isna().sum().sum()
    if total == 0:
        return 1.0

    return missing / total


def assign_weights_to_sources(
    satellite_df: pd.DataFrame | None,
    iot_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
    terrain_df: pd.DataFrame | None,
    vegetation_df: pd.DataFrame | None = None, # Added vegetation_df
) -> Dict[str, float]:
    """
    Compute one dynamic weight per source: satellite, iot, weather, terrain, vegetation.

    Inputs:
      - satellite_df: should include 'coverage_fraction_source' if available.
      - iot_df, weather_df, terrain_df, vegetation_df: any feature tables.

    Output:
      Dictionary: {"satellite": w_sat, "iot": w_iot, "weather": w_wx, "terrain": w_dem, "vegetation": w_veg}
    """
    weights: Dict[str, float] = {}

    # Satellite: use average coverage_fraction_source + missing fraction
    if satellite_df is not None and not satellite_df.empty:
        if "coverage_fraction_source" in satellite_df.columns:
            cov_sat = float(satellite_df["coverage_fraction_source"].mean())
        else:
            cov_sat = 1.0  # assume full coverage if not present
        miss_sat = summarize_missing_fraction(satellite_df)
        #weights["satellite"] = compute_source_weight(cov_sat, miss_sat)
        weights["satellite"] = float(compute_source_weight(cov_sat, miss_sat))
    else:
        weights["satellite"] = 0.0

    # IoT: we don't have coverage_fraction yet → assume coverage=1, use missingness
    if iot_df is not None and not iot_df.empty:
        cov_iot = 1.0
        miss_iot = summarize_missing_fraction(iot_df)
        # weights["iot"] = compute_source_weight(cov_iot, miss_iot)
        weights["iot"] = float(compute_source_weight(cov_iot, miss_iot))
    else:
        weights["iot"] = 0.0

    # Weather: same idea as IoT
    if weather_df is not None and not weather_df.empty:
        cov_wx = 1.0
        miss_wx = summarize_missing_fraction(weather_df)
        # weights["weather"] = compute_source_weight(cov_wx, miss_wx)
        weights["weather"] = float(compute_source_weight(cov_wx, miss_wx))
    else:
        weights["weather"] = 0.0

    # Terrain: usually tiny, no NaNs → high weight if exists
    if terrain_df is not None and not terrain_df.empty:
        cov_dem = 1.0
        miss_dem = summarize_missing_fraction(terrain_df)
        # weights["terrain"] = compute_source_weight(cov_dem, miss_dem)
        weights["terrain"] = float(compute_source_weight(cov_dem, miss_dem))
    else:
        weights["terrain"] = 0.0

    # Vegetation: new source
    if vegetation_df is not None and not vegetation_df.empty:
        cov_veg = 1.0
        miss_veg = summarize_missing_fraction(vegetation_df)
        weights["vegetation"] = float(compute_source_weight(cov_veg, miss_veg))
    else:
        weights["vegetation"] = 0.0

    logger.info(f"Dynamic source weights: {weights}")
    return weights


# Simple test / evaluation
if __name__ == "__main__":
    print("🚀 DYNAMIC FEATURE WEIGHTING TEST")

    # Tiny example: fake shapes just to exercise code
    sat_example = pd.DataFrame({"coverage_fraction_source": [0.8, 0.9]})
    iot_example = pd.DataFrame({"pm2_5": [5.0, None, 7.0]})
    wx_example = pd.DataFrame({"temperature_2m": [10, 11, 12]})
    dem_example = pd.DataFrame({"elevation_m": [100]})
    veg_example = pd.DataFrame({"ndvi": [0.2, 0.5, 0.8]})

    w = assign_weights_to_sources(
        satellite_df=sat_example,
        iot_df=iot_example,
        weather_df=wx_example,
        terrain_df=dem_example,
        vegetation_df=veg_example
    )

    print("✅ Weights:", w)
    print("🎉 DYNAMIC FEATURE WEIGHTING TEST COMPLETE")
