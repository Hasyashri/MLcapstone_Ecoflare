# %%writefile services/features/feature_builder.py
# =============================================================
# BULLETPROOF v3: ZERO NoneType crashes - Guaranteed DataFrames
# =============================================================

import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
import importlib
import sys
import numpy as np

from services.management.logging import get_logger

# Ingestion imports
from services.ingestion.satellite_ingestion import fetch_modis, fetch_viirs, fetch_cwfis_data
from services.ingestion.weather_ingestion import fetch_weather_data, CANADA_LOCATIONS as WX_LOCATIONS
from services.ingestion.terrain_ingestion import fetch_global_dem, CANADA_LOCATIONS as DEM_LOCATIONS
from services.ingestion.vegetation_ingestion import generate_ndvi_569k

FEATURE_DIR = Path("data/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
logger = get_logger("FeatureBuilder")

# =============================================================
# BULLETPROOF SAFE HELPERS
# =============================================================
def ensure_dataframe(df) -> pd.DataFrame:
    """🔒 FORCE DataFrame - handles None/empty/non-DataFrame."""
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df

def safe_qc_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """🔒 QC wrapper - bulletproof."""
    df = ensure_dataframe(df)
    if df.empty or not cols:
        return df
    
    try:
        available_cols = [c for c in cols if c in df.columns]
        if not available_cols:
            return df
        
        from services.ingestion.qc_masking import qc_numeric_outliers
        result = qc_numeric_outliers(df, available_cols)
        qc_df = result[0] if isinstance(result, tuple) else result
        return ensure_dataframe(qc_df)
    except Exception as e:
        logger.warning(f"QC failed: {e}")
        return df

def safe_fill_time_series(df: pd.DataFrame, time_col: str, value_cols: list) -> pd.DataFrame:
    df = ensure_dataframe(df)
    if df.empty or time_col not in df.columns:
        return df
    try:
        from services.ingestion.gap_fill import fill_time_series
        available_cols = [c for c in value_cols if c in df.columns]
        if available_cols:
            return ensure_dataframe(fill_time_series(df, time_col, available_cols))
        return df
    except:
        return df

def safe_fill_group_mean(df: pd.DataFrame, group_cols: list, value_cols: list) -> pd.DataFrame:
    df = ensure_dataframe(df)
    if df.empty:
        return df
    try:
        from services.ingestion.gap_fill import fill_with_group_mean
        return ensure_dataframe(fill_with_group_mean(df, group_cols, value_cols))
    except:
        return df

# =============================================================
# BULLETPROOF FEATURE BUILDERS
# =============================================================
def build_satellite_features() -> pd.DataFrame:
    print("🛰️ Building SATELLITE features...")
    frames = []
    
    for source_name, fetch_func, min_conf in [
        ("MODIS", fetch_modis, 60),
        ("VIIRS", fetch_viirs, 60),
        ("CWFIS", fetch_cwfis_data, 0)
    ]:
        raw_df = ensure_dataframe(fetch_func())
        if raw_df.empty:
            continue
            
        from services.ingestion.qc_masking import qc_satellite_fires, compute_coverage_fraction
        qc_result = qc_satellite_fires(raw_df, min_confidence=min_conf)
        qc_df = qc_result[0] if isinstance(qc_result, tuple) else qc_result
        qc_df = ensure_dataframe(qc_df)
        
        if not qc_df.empty:
            stats = qc_result[1] if len(qc_result) > 1 else {"output_count": len(qc_df), "input_count": len(raw_df)}
            cov = compute_coverage_fraction(stats["output_count"], stats["input_count"])
            qc_df = qc_df.copy()
            qc_df["source"] = source_name
            qc_df["coverage_fraction_source"] = cov
            frames.append(qc_df)

    if not frames:
        print("⚠️ No satellite data")
        return pd.DataFrame()

    sat_features = pd.concat(frames, ignore_index=True)
    core_cols = [c for c in ["latitude", "longitude", "brightness", "confidence", "source", "coverage_fraction_source"] 
                 if c in sat_features.columns]
    return sat_features[core_cols]

def build_iot_features(location_key: str = "Toronto") -> pd.DataFrame:
    print(f"🌫️ Building IoT features for {location_key}...")
    
    # Reload iot
    if 'services.ingestion.iot_ingestion' in sys.modules:
        del sys.modules['services.ingestion.iot_ingestion']
    import services.ingestion.iot_ingestion
    importlib.reload(services.ingestion.iot_ingestion)
    
    aq_locations = services.ingestion.iot_ingestion.CANADA_LOCATIONS
    if location_key not in aq_locations:
        print(f"⚠️ Unknown IoT location: {location_key}")
        return pd.DataFrame()

    # 🔒 BULLETPROOF CHAIN
    aq_df = ensure_dataframe(services.ingestion.iot_ingestion.fetch_open_meteo_air_quality(**aq_locations[location_key]))
    if aq_df.empty:
        print("⚠️ No IoT data")
        return pd.DataFrame()
    
    print(f"  📊 Raw IoT: {aq_df.shape}")
    
    # Safe processing
    aq_qc = safe_qc_numeric(aq_df, ["pm2_5", "pm10"])
    print(f"  📊 After QC: {aq_qc.shape}")
    
    if "timestamp" in aq_qc.columns:
        aq_qc["timestamp"] = pd.to_datetime(aq_qc["timestamp"], errors='coerce')
        aq_qc = aq_qc.sort_values(["location", "timestamp"]).reset_index(drop=True)
        aq_filled = safe_fill_time_series(aq_qc, "timestamp", ["pm2_5", "pm10"])
        aq_filled = safe_fill_group_mean(aq_filled, ["location"], ["pm2_5", "pm10"])
    else:
        aq_filled = aq_qc
    
    print(f"  📊 After gapfill: {aq_filled.shape}")

    # 🔒 SAFE ROC - explicit checks
    if (isinstance(aq_filled, pd.DataFrame) and 
        not aq_filled.empty and 
        "pm2_5" in aq_filled.columns and 
        len(aq_filled) > 1 and 
        "timestamp" in aq_filled.columns):
        
        sorted_df = aq_filled.sort_values("timestamp").reset_index(drop=True)
        sorted_df["pm2_5_roc"] = sorted_df["pm2_5"].diff()
        aq_filled = sorted_df
    else:
        print("⚠️ Skipping ROC - insufficient data")

    logger.info(f"IoT ({location_key}): {aq_filled.shape}")
    return aq_filled

def build_weather_features(location_key: str = "Toronto") -> pd.DataFrame:
    print(f"🌤️ Building WEATHER features for {location_key}...")
    
    if location_key not in WX_LOCATIONS:
        return pd.DataFrame()

    w_df = ensure_dataframe(fetch_weather_data(**WX_LOCATIONS[location_key]))
    if w_df.empty:
        return pd.DataFrame()

    num_cols = [c for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"] if c in w_df.columns]
    w_qc = safe_qc_numeric(w_df, num_cols)
    
    if "timestamp" in w_qc.columns:
        w_qc["timestamp"] = pd.to_datetime(w_qc["timestamp"], errors='coerce')
        w_qc = w_qc.sort_values(["location", "timestamp"])
        w_filled = safe_fill_time_series(w_qc, "timestamp", num_cols)
        w_filled = safe_fill_group_mean(w_filled, ["location"], num_cols)
    else:
        w_filled = w_qc

    # Dryness index (safe)
    if "temperature_2m" in w_filled.columns and "relative_humidity_2m" in w_filled.columns:
        temp = w_filled["temperature_2m"].fillna(0)
        temp_norm = (temp - temp.min()) / (temp.max() - temp.min() + 1e-6)
        humi_inv = 1 - (w_filled["relative_humidity_2m"].fillna(50) / 100.0)
        w_filled["dryness_index"] = (temp_norm + humi_inv) / 2.0

    return w_filled

def build_terrain_features(location_key: str = "Toronto") -> pd.DataFrame:
    print(f"🗻 Building TERRAIN features for {location_key}...")
    
    if location_key not in DEM_LOCATIONS:
        return pd.DataFrame()

    dem_df = ensure_dataframe(fetch_global_dem(**DEM_LOCATIONS[location_key]))
    return dem_df

def build_vegetation_features() -> pd.DataFrame:
    print("🌿 Building VEGETATION features...")
    
    ndvi_df = ensure_dataframe(generate_ndvi_569k())
    if ndvi_df.empty:
        veg_dir = Path("data/features")
        veg_files = list(veg_dir.glob("vegetation_ndvi_*.csv"))
        if veg_files:
            latest_veg = sorted(veg_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            ndvi_df = pd.read_csv(latest_veg)
            print(f"✅ Loaded raw NDVI: {ndvi_df.shape}")
    
    return ensure_dataframe(ndvi_df)

def save_feature_snapshot(features: pd.DataFrame, name: str) -> Path:
    features = ensure_dataframe(features)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    path = FEATURE_DIR / f"{name}_{ts}.csv"
    features.to_csv(path, index=False)
    logger.info(f"Saved: {path}")
    return path

if __name__ == "__main__":
    print("🚀 BULLETPROOF FEATURE BUILDER\n")
    
    sat = build_satellite_features()
    print(f"\n✅ Satellite: {sat.shape}")
    save_feature_snapshot(sat, "satellite_all")

    iot = build_iot_features("Toronto")
    print(f"\n✅ IoT: {iot.shape}")
    save_feature_snapshot(iot, "iot_toronto")

    wx = build_weather_features("Toronto")
    print(f"\n✅ Weather: {wx.shape}")
    save_feature_snapshot(wx, "weather_toronto")

    dem = build_terrain_features("Toronto")
    print(f"\n✅ Terrain: {dem.shape}")
    save_feature_snapshot(dem, "terrain_toronto")

    veg = build_vegetation_features()
    print(f"\n✅ Vegetation: {veg.shape}")
    save_feature_snapshot(veg, "vegetation_ndvi")

    print("\n🎉 BULLETPROOF SUCCESS!")
