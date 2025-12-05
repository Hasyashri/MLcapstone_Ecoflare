# modules/root_cause_analysis/root_cause_data.py

"""
Data engineering for MVP 3 – Root Cause Analysis.

This module takes detection output + contextual data and produces
a flat numeric feature dictionary ready for the classifier.
"""

from typing import Dict, Any


def build_rootcause_features(detection_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Map detection + external context to numerical features.
    """
    features = {
        "vote_count": float(detection_data.get("vote_count", 0)),
        "temp": float(detection_data.get("weather_temp", 30)),
        "wind": float(detection_data.get("weather_wind", 10)),
        "humidity": float(detection_data.get("weather_humidity", 40)),
        "near_power_lines": float(detection_data.get("near_power_lines", 0)),
        "population_density": float(detection_data.get("population_density", 0)),
        "recent_lightning_strikes": float(
            detection_data.get("recent_lightning_strikes", 0)
        ),
    }
    return features
