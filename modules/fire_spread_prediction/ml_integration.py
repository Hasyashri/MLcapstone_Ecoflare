# modules/fire_spread_prediction/ml_integration.py

"""
Integration layer for MVP 2 – links feature engineering and prediction.

This is what your FastAPI endpoint or Streamlit UI can call directly.
"""

from typing import Dict, Any

from .feature_engineering import build_features
from .fire_spread_prediction_model import predict_spread


def run_spread_prediction_pipeline(
    fire_location: Dict[str, float],
    wind_data: Dict[str, float],
    vegetation_data: Dict[str, Any],
    time_horizon: int = 60,
) -> Dict[str, Any]:
    """
    High-level helper: raw inputs → features → prediction.
    """
    features = build_features(fire_location, wind_data, vegetation_data)
    prediction = predict_spread(features, time_horizon=time_horizon)
    prediction["features"] = {
        "vegetation_type": features.vegetation_type,
        "wind_speed": features.wind_speed,
        "moisture": features.moisture,
        "slope": features.slope,
        "temp": features.temp,
    }
    return prediction
