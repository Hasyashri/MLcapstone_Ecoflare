# modules/fire_detection/iot_data.py

"""
IoT data helper for MVP 1 – Detection.
Simulates smoke / flame sensors deployed in the field.
"""

from typing import Dict, Any
import random


def get_iot_sensor(location_id: str) -> Dict[str, Any]:
    """
    Simulate an IoT sensor reading.
    """
    smoke_level = random.uniform(0, 1)
    flame_detected = random.random() < 0.2
    alert = smoke_level > 0.7 or flame_detected

    return {
        "name": "IOT_SENSOR",
        "status": "ALERT" if alert else "OK",
        "details": {
            "location_id": location_id,
            "smoke_level": smoke_level,
            "flame_detected": flame_detected,
        },
    }
