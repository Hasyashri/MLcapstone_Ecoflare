# modules/fire_detection/vegetation_data.py

"""
Vegetation / fuel risk helper for MVP 1 – Detection.
"""

from typing import Dict, Any
import random


def get_vegetation_risk(lat: float, lon: float) -> Dict[str, Any]:
    """
    Simulate a vegetation / FWI risk lookup.
    """
    risk_level = random.choice(["LOW", "MEDIUM", "HIGH", "VERY_HIGH"])
    alert = risk_level in ["HIGH", "VERY_HIGH"]

    return {
        "name": "VEGETATION_RISK",
        "status": "ALERT" if alert else "OK",
        "details": {
            "lat": lat,
            "lon": lon,
            "risk": risk_level,
        },
    }
