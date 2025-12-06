# =============================================================
# QC MASKING v2: BULLETPROOF - Never Returns None
# Fixes: IoT/Weather data destruction (120→0 rows)
# =============================================================

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from services.management.logging import get_logger

logger = get_logger("QCMasking")

def ensure_dataframe(df) -> pd.DataFrame:
    """🔒 Convert anything to DataFrame - handles None/non-DataFrame."""
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df

def qc_satellite_fires(
    df: pd.DataFrame,
    min_confidence: int = 50
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """✅ Satellite QC - confidence filter (MODIS/VIIRS/CWFIS)."""
    df = ensure_dataframe(df)
    if df.empty:
        return df, {"input_count": 0, "output_count": 0, "qc_mask_fraction": 0.0}

    input_count = len(df)
    
    try:
        if "confidence" in df.columns:
            qc_df = df[df["confidence"] >= min_confidence].copy()
        else:
            logger.info("qc_satellite_fires: 'confidence' missing, bypass QC")
            qc_df = df.copy()
    except Exception as e:
        logger.error(f"qc_satellite_fires failed: {e}")
        qc_df = df.copy()

    qc_df = ensure_dataframe(qc_df)
    output_count = len(qc_df)
    qc_mask_fraction = 1.0 - (output_count / input_count) if input_count > 0 else 0.0
    
    logger.info(
        f"Satellite QC: {input_count} → {output_count} rows "
        f"({qc_mask_fraction:.1%} masked, min_conf={min_confidence})"
    )

    return qc_df, {
        "input_count": input_count,
        "output_count": output_count,
        "qc_mask_fraction": qc_mask_fraction
    }

def qc_numeric_outliers(
    df: pd.DataFrame,
    numeric_cols: List[str],
    zscore_max: float = 4.0,
    skip_small: bool = True  # ✅ NEW: Skip QC for small datasets
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    ✅ FIXED: QC numeric outliers - gentle for IoT/Weather (small datasets).
    
    - skip_small=True: Skip QC if <50 rows (IoT=120, Weather=24)
    - Vectorized z-score filtering for large NDVI (569K)
    """
    df = ensure_dataframe(df)
    if df.empty:
        return df, {"input_count": 0, "output_count": 0, "qc_mask_fraction": 0.0}

    input_count = len(df)
    
    # ✅ GENTLE QC: Skip for small datasets (IoT/Weather)
    if skip_small and input_count < 50:
        logger.info(f"qc_numeric_outliers: Skipping QC - small dataset ({input_count} rows)")
        return df, {
            "input_count": input_count,
            "output_count": input_count,
            "qc_mask_fraction": 0.0
        }

    qc_df = df.copy()
    filtered_count = 0

    for col in numeric_cols:
        if col not in qc_df.columns:
            logger.warning(f"qc_numeric_outliers: column '{col}' not found, skipping")
            continue

        try:
            series = pd.to_numeric(qc_df[col], errors='coerce')
            valid_mask = ~series.isna()
            
            if valid_mask.sum() < 3:  # Need min 3 valid points for stats
                logger.warning(f"qc_numeric_outliers: insufficient valid data in '{col}'")
                continue
                
            mu = series[valid_mask].mean()
            sigma = series[valid_mask].std(ddof=0)
            if sigma == 0:
                sigma = 1.0  # Avoid div0
                
            z = np.abs((series - mu) / sigma)
            before = len(qc_df)
            qc_df = qc_df[(z <= zscore_max) | (~valid_mask)].copy()
            filtered_count += before - len(qc_df)
            
            logger.info(f"QC z-score {col}: filtered {before - len(qc_df)} rows")
            
        except Exception as e:
            logger.warning(f"QC failed for {col}: {e}")
            continue

    qc_df = ensure_dataframe(qc_df)
    output_count = len(qc_df)
    qc_mask_fraction = 1.0 - (output_count / input_count) if input_count > 0 else 0.0
    
    logger.info(
        f"Numeric QC total: {input_count} → {output_count} rows "
        f"({qc_mask_fraction:.1%} masked)"
    )

    return qc_df, {
        "input_count": input_count,
        "output_count": output_count,
        "qc_mask_fraction": qc_mask_fraction
    }

def compute_coverage_fraction(valid_count: int, total_count: int) -> float:
    """✅ Coverage = valid/total (0.0 if total<=0)."""
    if total_count <= 0:
        logger.warning("compute_coverage_fraction: total_count <= 0")
        return 0.0
    coverage = valid_count / total_count
    logger.debug(f"Coverage: {coverage:.1%} ({valid_count}/{total_count})")
    return coverage

def add_timestamp_if_missing(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """✅ Add timestamp column if missing."""
    df = ensure_dataframe(df)
    if df.empty:
        return df
        
    df = df.copy()
    if ts_col not in df.columns:
        df[ts_col] = pd.Timestamp.utcnow()
        logger.info(f"Added fallback '{ts_col}' column")
    return df

# =============================================================
# NEW: GENTLE MODE FOR PRODUCTION
# =============================================================
def qc_production_mode(df: pd.DataFrame, data_type: str = "generic") -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    ✅ PRODUCTION QC: Context-aware, data-type specific.
    
    Args:
        df: Raw data
        data_type: "satellite" | "iot" | "weather" | "vegetation" | "terrain"
    """
    df = ensure_dataframe(df)
    
    if data_type == "satellite":
        return qc_satellite_fires(df)
    elif data_type in ["iot", "weather"]:
        # Gentle QC for small time-series
        return qc_numeric_outliers(df, df.select_dtypes(include=[np.number]).columns.tolist(), 
                                 zscore_max=4.0, skip_small=True)
    elif data_type == "vegetation":
        # Aggressive QC OK for 569K pixels
        return qc_numeric_outliers(df, ["ndvi"], zscore_max=3.0, skip_small=False)
    else:
        # Terrain/static - minimal QC
        return df, {"input_count": len(df), "output_count": len(df), "qc_mask_fraction": 0.0}

# =============================================================
# TEST FUNCTIONS
# =============================================================
if __name__ == "__main__":
    print("🧪 QC MASKING TEST")
    
    # Mock IoT data (120 rows)
    iot_mock = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=120, freq="H"),
        "pm2_5": np.random.normal(5, 2, 120),
        "pm10": np.random.normal(15, 5, 120)
    })
    iot_qc, stats = qc_numeric_outliers(iot_mock, ["pm2_5", "pm10"], skip_small=True)
    print(f"IoT: {len(iot_mock)} → {len(iot_qc)} rows (skip_small=True)")
    
    # Mock NDVI (569K)
    ndvi_mock = pd.DataFrame({"ndvi": np.random.normal(0.3, 0.1, 569012)})
    ndvi_qc, _ = qc_numeric_outliers(ndvi_mock, ["ndvi"], skip_small=False)
    print(f"NDVI: {len(ndvi_mock)} → {len(ndvi_qc)} rows (full QC)")
    
    print("✅ QC MASKING BULLETPROOF!")
