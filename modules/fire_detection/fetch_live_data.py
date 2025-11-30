# modules/fire_detection/fetch_live_data.py

"""
Live data helpers for MVP 1 – Detection.
For now, these return MOCKED data so the pipeline runs end-to-end.
Replace the TODOs with real API calls later.
"""

from typing import Dict, Any
import random


def get_nasa_firms(region: str, time_window: str) -> Dict[str, Any]:
    """
    Simulate a NASA FIRMS hotspot query.
    """
    anomaly = random.random() < 0.4
    return {
        "name": "NASA_FIRMS",
        "status": "ALERT" if anomaly else "OK",
        "details": {
            "region": region,
            "time_window": time_window,
            "hotspots": random.randint(0, 10),
        },
    }


def get_cwfis(region: str) -> Dict[str, Any]:
    """
    Simulate a CWFIS active fire query.
    """
    active_fire = random.random() < 0.3
    return {
        "name": "CWFIS",
        "status": "ALERT" if active_fire else "OK",
        "details": {
            "region": region,
            "active_fires": random.randint(0, 5),
        },
    }


def get_weather(lat: float, lon: float) -> Dict[str, Any]:
    """
    Simulate a weather service returning fire-relevant conditions.
    """
    temp = random.uniform(15, 40)
    wind = random.uniform(0, 40)
    humidity = random.uniform(10, 80)
    high_risk = temp > 30 and wind > 15 and humidity < 30

    return {
        "name": "WEATHER",
        "status": "ALERT" if high_risk else "OK",
        "details": {
            "lat": lat,
            "lon": lon,
            "temp": temp,
            "wind": wind,
            "humidity": humidity,
        },
    }
