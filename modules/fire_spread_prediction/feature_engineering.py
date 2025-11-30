# modules/fire_spread_prediction/feature_engineering.py

"""
Feature engineering service for MVP 2 – Fire Spread Prediction.

This corresponds to the `/api/v1/features` service in the assignment.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class SpreadFeatures:
    vegetation_type: str
    wind_speed: float
    moisture: float
    slope: float
    temp: float


def build_features(
    fire_location: Dict[str, float],
    wind_data: Dict[str, float],
    vegetation_data: Dict[str, Any],
) -> SpreadFeatures:
    """
    Convert raw inputs to a clean, typed feature vector.
    """
    return SpreadFeatures(
        vegetation_type=vegetation_data.get("type", "forest"),
        wind_speed=float(wind_data.get("speed", 10.0)),
        moisture=float(vegetation_data.get("moisture", 0.3)),
        slope=float(fire_location.get("slope", 5.0)),
        temp=float(wind_data.get("temp", 30.0)),
    )


def as_dict(features: SpreadFeatures) -> Dict[str, Any]:
    """Helper for API responses."""
    return asdict(features)
