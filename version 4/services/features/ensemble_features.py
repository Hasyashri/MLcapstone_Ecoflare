# =============================================================
# ENSEMBLE FEATURES - 1 ROW PER LOCATION/TIME WITH WEIGHTS
# Combines: Satellite + IoT + Weather + Terrain + Vegetation
# =============================================================

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import importlib
import sys

from services.management.logging import get_logger

LOGGER = get_logger("EnsembleFeatures")

# =============================================================
# CLEAN RELOAD OF FEATURE BUILDER (Fixes stale imports)
# =============================================================

for key in list(sys.modules.keys()):
    if key.startswith("services.features.feature_builder"):
        del sys.modules[key]

try:
    import services.features.feature_builder as fb
    importlib.reload(fb)
except Exception as e:
    LOGGER.error(f"❌ Failed to import feature_builder: {e}")
    raise

# Core ensemble components
from services.features.dynamic_feature_weighting import assign_weights_to_sources
from services.features.uncertainty_propagation import propagate_ensemble_uncertainty
from services.features.spatial_smoothing import smooth_fire_points

# QC + Gapfill
from services.ingestion.qc_masking import qc_numeric_outliers
from services.ingestion.gap_fill import fill_time_series

# =============================================================
# GLOBAL CONSTANTS
# =============================================================

DATA_DIR = Path("data")
FIRMS_DIR = DATA_DIR / "firms"

CITY_TO_AGENCY = {
    "Toronto": "on", "Vancouver": "bc", "Calgary": "ab",
    "Ottawa": "on", "Montreal": "qc", "Winnipeg": "mb"
}

# Baselines used for fire probability
PRODUCTION_BASELINES = {
    'hotspot_count_area': 20.0,
    'mean_pm2_5_area': 50.0,
    'mean_dryness_index_area': 0.8,
    'mean_elevation_area': 500.0,
    'vegetation_stress': 0.6
}


# =============================================================
# FIRE TRUTH
# =============================================================

def load_active_fires_latest(pattern: str = "CWFIS_Ontario_*.csv") -> pd.DataFrame | None:
    files = sorted(FIRMS_DIR.glob(pattern), reverse=True)
    if not files:
        return None
    return pd.read_csv(files[0])


def build_area_fire_truth(active_fires: pd.DataFrame, target_agency: str) -> pd.DataFrame:
    """FIRE TRUTH: Load → QC → Spatial Smoothing → Aggregate."""
    if active_fires is None or active_fires.empty:
        return pd.DataFrame([{
            "area_id": target_agency,
            "fire_count": 0,
            "qc_confidence_score": 0.8
        }])

    af = active_fires.copy()

    agency_col = next((col for col in ['agency', 'AGENCY', 'src_agency'] if col in af.columns), None)
    if agency_col is None:
        return pd.DataFrame([{
            "area_id": target_agency,
            "fire_count": 0,
            "qc_confidence_score": 0.8
        }])

    af["area_id"] = af[agency_col].astype(str).str.lower().str.strip()
    target_agency = target_agency.lower().strip()

    af = smooth_fire_points(af)
    af = af[af["area_id"] == target_agency]

    if af.empty:
        return pd.DataFrame([{
            "area_id": target_agency,
            "fire_count": 0,
            "qc_confidence_score": 0.8
        }])

    area_fire = af.groupby("area_id", as_index=False).agg(
        fire_count=("firename", "count"),
        qc_confidence_score=("qc_confidence_score", "mean")
    ).fillna({"fire_count": 0, "qc_confidence_score": 0.8})

    return area_fire


# =============================================================
# GENERIC SAFE AGGREGATION
# =============================================================

def aggregate_by_area(df: pd.DataFrame, target_agency: str, feature_cols: Dict[str, str]) -> pd.DataFrame:
    """
    Generic aggregation: 1 row per area/time
    Fallbacks added:
      - Add timestamp if missing
      - QC outliers only on available columns
      - Safe gap filling
    """

    if df is None or df.empty:
        return pd.DataFrame([{
            "area_id": target_agency,
            "qc_confidence_score": 0.8
        }])

    df = df.copy()

    # Fallback timestamp
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.utcnow()

    # Run QC on only available columns
    numeric_columns = [c for c in feature_cols.values() if c in df.columns]
    df_qc, _ = qc_numeric_outliers(df, numeric_columns, zscore_max=3.0)

    # Safe gap fill
    df_clean = fill_time_series(df_qc, "timestamp", numeric_columns, "both")

    df_clean["area_id"] = target_agency
    latest = df_clean.tail(1)

    agg_data = {"area_id": target_agency, "qc_confidence_score": 0.85}

    for area_col, src_col in feature_cols.items():
        if src_col in latest.columns:
            agg_data[area_col] = latest[src_col].iloc[0]
        else:
            agg_data[area_col] = 0.0  # Fallback

    return pd.DataFrame([agg_data])


# =============================================================
# MAIN ENSEMBLE BUILDER
# =============================================================

def build_production_ensemble_features(city_name: str = "Toronto") -> Tuple[pd.DataFrame, List[Dict]]:
    """MAIN PIPELINE: 5 sources → 1 row with weights + uncertainty."""
    print("🏭 BUILDING ENSEMBLE: 5→1 ROW...")

    target_agency = CITY_TO_AGENCY.get(city_name, "on")

    # -------------------------
    # BUILD RAW FEATURES
    # -------------------------
    sat = fb.build_satellite_features()
    iot = fb.build_iot_features(city_name)
    wx = fb.build_weather_features(city_name)
    dem = fb.build_terrain_features(city_name)
    veg = fb.build_vegetation_features()  # FIXED: now guaranteed to exist

    # Fire truth
    fires = load_active_fires_latest()
    area_fire = build_area_fire_truth(fires, target_agency)

    # -------------------------
    # AGGREGATION (CORRECTED)
    # -------------------------
    sat_area = aggregate_by_area(sat, target_agency, {"hotspot_count_area": "latitude"})
    iot_area = aggregate_by_area(iot, target_agency, {"mean_pm2_5_area": "pm2_5"})
    wx_area = aggregate_by_area(wx, target_agency, {"mean_dryness_index_area": "dryness_index"})
    dem_area = aggregate_by_area(dem, target_agency, {"mean_elevation_area": "elevation_m"})

    # FIXED vegetation column mapping
    veg_area = aggregate_by_area(veg, target_agency, {"vegetation_stress": "veg_health"})

    # -------------------------
    # DYNAMIC WEIGHTS
    # -------------------------
    weights = assign_weights_to_sources(sat, iot, wx, dem, veg)

    # -------------------------
    # MERGE ALL SOURCES
    # -------------------------
    ensemble = area_fire.copy()
    for source_df in [sat_area, iot_area, wx_area, dem_area, veg_area]:
        ensemble = ensemble.merge(source_df, on="area_id", how="left")

    # Replace NaN with safe defaults
    ensemble = ensemble.fillna(0.0)

    # -------------------------
    # UNCERTAINTY
    # -------------------------
    sources = {
        "satellite": sat, "iot": iot, "weather": wx,
        "terrain": dem, "vegetation": veg
    }
    ensemble["ensemble_uncertainty"] = propagate_ensemble_uncertainty(sources, weights).iloc[0]

    # -------------------------
    # NORMALIZATION
    # -------------------------
    for col in [
        'hotspot_count_area',
        'mean_pm2_5_area',
        'mean_dryness_index_area',
        'mean_elevation_area',
        'vegetation_stress'
    ]:
        baseline = PRODUCTION_BASELINES.get(col, 1.0)
        ensemble[f"{col}_norm"] = ensemble.get(col, 0) / baseline

    # -------------------------
    # FIRE PROBABILITY
    # -------------------------
    ensemble["fire_probability"] = sum(
        weights.get(source, 0.2) *
        ensemble.get(f"{col}_norm", pd.Series([0.0]))
        for source, col in [
            ("satellite", "hotspot_count_area"),
            ("iot", "mean_pm2_5_area"),
            ("weather", "mean_dryness_index_area"),
            ("terrain", "mean_elevation_area"),
            ("vegetation", "vegetation_stress")
        ]
    )

    # -------------------------
    # RISK LABEL
    # -------------------------
    ensemble["risk_level"] = pd.cut(
        ensemble["fire_probability"],
        bins=[-0.1, 0.3, 0.6, 0.8, 1.5],
        labels=["Low", "Medium", "High", "Critical"]
    )

    # -------------------------
    # ALERTS
    # -------------------------
    alerts = []
    if ensemble["fire_probability"].iloc[0] > 0.7:
        alerts.append({
            "area_id": target_agency,
            "risk_level": str(ensemble["risk_level"].iloc[0]),
            "probability": float(ensemble["fire_probability"].iloc[0]),
            "uncertainty": float(ensemble["ensemble_uncertainty"].iloc[0])
        })

    return ensemble, alerts


# =============================================================
# MAIN FOR DEBUGGING
# =============================================================

if __name__ == "__main__":
    features, alerts = build_production_ensemble_features("Toronto")
    print("\n✅ ENSEMBLE (1 ROW):")
    print(features.round(3))
    print(f"\n🚨 ALERTS: {alerts}")
    print("🎉 5→1 ROW ENSEMBLE COMPLETE!")


FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)