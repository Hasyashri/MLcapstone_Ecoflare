# ===============================================
# File: modules/fire_detection/iot_data.py
# Purpose: Simulate or fetch IoT sensor data
# ===============================================

import random
from datetime import datetime

# 🔥 ADD THIS BLOCK ONLY
from utils.logger import setup_logger
logger = setup_logger("iot_data")
# 🔥 END OF BLOCK


def fetch_iot_sensor_data(sensor_id="SENSOR_001", location="Toronto"):
    """
    Simulate IoT sensor readings for fire detection.
    In real implementation, this would connect to actual IoT devices.
    """
    logger.info(f"Fetching IoT data from {sensor_id} at {location}...")   # added
    
    # Simulated sensor data
    sensor_data = {
        'sensor_id': sensor_id,
        'location': location,
        'timestamp': datetime.now().isoformat(),
        'temperature': round(random.uniform(15, 35), 2),  # Celsius
        'smoke_level': round(random.uniform(0, 100), 2),  # 0-100 scale
        'humidity': round(random.uniform(30, 80), 2),     # Percentage
        'air_quality_index': random.randint(20, 150),
        'flame_detected': random.choice([True, False])
    }
    
    logger.info(f"IoT Data Read — Temp={sensor_data['temperature']}°C, "
                f"Smoke={sensor_data['smoke_level']}, "
                f"Flame={sensor_data['flame_detected']}")  # added
    
    return sensor_data


def analyze_iot_risk(sensor_data):
    """Analyze IoT data for fire risk"""
    risk_score = 0
    
    if sensor_data['temperature'] > 30:
        risk_score += 30
    if sensor_data['smoke_level'] > 50:
        risk_score += 40
    if sensor_data['flame_detected']:
        risk_score += 30
    
    if risk_score >= 70:
        return "HIGH"
    elif risk_score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


if __name__ == "__main__":
    data = fetch_iot_sensor_data()
    risk = analyze_iot_risk(data)
    print(f"🔥 Fire Risk from IoT: {risk}")
    logger.info(f"🔥 Fire Risk from IoT: {risk}")  # added