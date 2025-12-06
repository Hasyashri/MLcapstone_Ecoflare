# =============================================================
# SPATIAL SMOOTHING - Gaussian/IDW for NDVI/fire points
# Small smoothing over sparse satellite/NDVI data
# =============================================================

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from services.management.logging import get_logger

logger = get_logger("SpatialSmoothing")

def smooth_fire_points(fire_df: pd.DataFrame, sigma: float = 1.5, method: str = "gaussian") -> pd.DataFrame:
    """
    Smooth fire_count over time/space (small Gaussian kernel).
    
    Args:
        fire_df: DataFrame with 'fire_count', 'latitude', 'longitude'
        sigma: Gaussian smoothing parameter (1.5 = gentle)
        method: "gaussian" or "idw"
    """
    if fire_df is None or fire_df.empty or 'fire_count' not in fire_df.columns:
        logger.warning("No fire_count column for smoothing")
        return fire_df
    
    df = fire_df.copy()
    
    if method == "gaussian":
        # Sort by time/location for smooth transitions
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        elif all(col in df.columns for col in ['latitude', 'longitude']):
            df = df.sort_values(['latitude', 'longitude'])
        
        # Apply 1D Gaussian filter along sorted axis
        smoothed = gaussian_filter1d(df['fire_count'].fillna(0).values, sigma=sigma)
        df['fire_count_smoothed'] = smoothed
        df['fire_count'] = df['fire_count_smoothed'].fillna(df['fire_count'])
        
        logger.info(f"Gaussian smoothing: fire_count std {df['fire_count'].std():.2f}")
    
    elif method == "idw":
        # Simple rolling mean (proxy for spatial IDW)
        df['fire_count_smoothed'] = df['fire_count'].rolling(5, center=True, min_periods=1).mean()
        df['fire_count'] = df['fire_count_smoothed'].fillna(df['fire_count'])
        logger.info("IDW proxy smoothing applied")
    
    return df.drop(columns=['fire_count_smoothed'], errors='ignore')

def smooth_ndvi_grid(ndvi_df: pd.DataFrame, sigma: float = 2.0) -> pd.DataFrame:
    """
    Spatial Gaussian smoothing for NDVI pixels.
    
    Args:
        ndvi_df: DataFrame with 'latitude', 'longitude', 'ndvi'
        sigma: Smoothing parameter
    """
    if ndvi_df is None or ndvi_df.empty or 'ndvi' not in ndvi_df.columns:
        logger.warning("No NDVI data for smoothing")
        return ndvi_df
    
    df = ndvi_df.sort_values(['latitude', 'longitude']).copy()
    
    # Spatial sorting ensures smooth neighborhood
    smoothed = gaussian_filter1d(df['ndvi'].fillna(0).values, sigma=sigma)
    df['ndvi_smoothed'] = smoothed
    df['ndvi'] = df['ndvi_smoothed'].fillna(df['ndvi'])
    
    logger.info(f"NDVI smoothing: mean={df['ndvi'].mean():.3f}, std={df['ndvi'].std():.3f}")
    return df.drop(columns=['ndvi_smoothed'], errors='ignore')

if __name__ == "__main__":
    print("🗺️ SPATIAL SMOOTHING TEST")
    
    # Mock fire data
    fire_mock = pd.DataFrame({
        'fire_count': [0, 3, 12, 8, 2, 15, 0],
        'latitude': [43.6, 43.61, 43.62, 43.63, 43.64, 43.65, 43.66]
    })
    
    smoothed_fire = smooth_fire_points(fire_mock)
    print("✅ Fire smoothing:")
    print(f"  Before: {fire_mock['fire_count'].values}")
    print(f"  After:  {smoothed_fire['fire_count'].round(1).values}")
    
    print("\n🎉 SPATIAL SMOOTHING READY!")
