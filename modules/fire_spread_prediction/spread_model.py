# modules/fire_spread_prediction/spread_model.py

"""
Physics-inspired base spread model for MVP 2.

This is the simple "physical" part of the hybrid model.
"""

from .feature_engineering import SpreadFeatures


def physics_spread_rate(features: SpreadFeatures) -> float:
    """
    Very simple placeholder physics-style formula.

    Higher wind + slope + temperature increase spread.
    Higher moisture decreases spread.
    """
    base = 0.5
    wind_factor = 0.05 * features.wind_speed
    slope_factor = 0.02 * features.slope
    moisture_factor = -0.4 * features.moisture
    temp_factor = 0.02 * max(features.temp - 25, 0)

    spread = base + wind_factor + slope_factor + moisture_factor + temp_factor
    return max(spread, 0.0)
