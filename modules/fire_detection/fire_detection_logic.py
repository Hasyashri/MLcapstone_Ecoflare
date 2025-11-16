# ===============================================
# File: modules/fire_detection/fire_detection_logic.py
# Purpose: Main fire detection logic combining all data sources
# ===============================================

from fetch_live_data import fetch_nasa_firms_data, fetch_cwfis_data, fetch_weather_data
from iot_data import fetch_iot_sensor_data, analyze_iot_risk
from vegetation_data import fetch_vegetation_data, get_vegetation_fire_risk
from datetime import datetime

# 🔥 ADD THIS BLOCK ONLY
from utils.logger import setup_logger
logger = setup_logger("fire_detection_logic")
# 🔥 END BLOCK


def detect_fire(lat=43.65, lon=-79.38, location_name="Toronto"):
    """
    Main fire detection function using voting logic.
    Combines satellite, weather, IoT, and vegetation data.
    """
    logger.info("=" * 70)
    logger.info(f"WILDFIRE DETECTION - {location_name}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info("")

    # Fetch all data sources
    nasa_data = fetch_nasa_firms_data()
    cwfis_data = fetch_cwfis_data()
    weather_data = fetch_weather_data(lat, lon)
    iot_data = fetch_iot_sensor_data(location=location_name)
    veg_data = fetch_vegetation_data(lat, lon)

    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS & VOTING")
    logger.info("=" * 70)

    # Voting system: count how many sources indicate fire
    fire_votes = 0
    total_votes = 0
    evidence = []

    # Vote 1: Satellite hotspots
    total_votes += 1
    if nasa_data is not None and len(nasa_data) > 0:
        fire_votes += 1
        evidence.append(f"Satellite: {len(nasa_data)} hotspots detected")
        logger.info(f"✅ VOTE 1: Satellite shows {len(nasa_data)} hotspots - FIRE")
    else:
        logger.info(f"⭕ VOTE 1: No satellite hotspots - NO FIRE")

    # Vote 2: Official fire reports
    total_votes += 1
    if cwfis_data is not None and len(cwfis_data) > 0:
        fire_votes += 1
        evidence.append(f"Official: {len(cwfis_data)} fires reported")
        logger.info(f"✅ VOTE 2: {len(cwfis_data)} official fires - FIRE")
    else:
        logger.info(f"⭕ VOTE 2: No official fire reports - NO FIRE")

    # Vote 3: Weather conditions
    total_votes += 1
    if weather_data and 'current' in weather_data:
        temp = weather_data['current']['temperature_2m']
        humidity = weather_data['current']['relative_humidity_2m']

        if temp > 30 and humidity < 30:
            fire_votes += 1
            evidence.append(f"Weather: {temp}°C, {humidity}% humidity - High risk")
            logger.info(f"✅ VOTE 3: Weather conditions indicate HIGH fire risk - FIRE")
        else:
            logger.info(f"⭕ VOTE 3: Weather conditions normal - NO FIRE")