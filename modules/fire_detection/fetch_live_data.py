# ===============================================
# File: modules/fire_detection/fetch_live_data.py
# Purpose: Fetch real-time fire and weather data
# ===============================================

import requests
import pandas as pd
from datetime import datetime
from io import StringIO

# 🔥 ADD THIS BLOCK
from utils.logger import setup_logger
logger = setup_logger("fetch_live_data")
# 🔥 END BLOCK


def fetch_nasa_firms_data():
    try:
        url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Canada_24h.csv"

        logger.info("Fetching NASA FIRMS data...")   # added
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        ontario_fires = df[
            (df['longitude'] >= -95) & (df['longitude'] <= -74) &
            (df['latitude'] >= 41) & (df['latitude'] <= 57)
        ]
        
        logger.info(f"Found {len(ontario_fires)} hotspots")  # added
        return ontario_fires

    except Exception as e:
        logger.error(f"NASA FIRMS failed: {e}")  # added
        return None


def fetch_cwfis_data():
    try:
        url = "https://cwfis.cfs.nrcan.gc.ca/downloads/activefires/activefires.csv"

        logger.info("Fetching CWFIS data...")   # added
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        if 'src_agency' in df.columns:
            ontario_fires = df[df['src_agency'] == 'ON']
        else:
            ontario_fires = df
        
        logger.info(f"Found {len(ontario_fires)} fires")   # added
        return ontario_fires

    except Exception as e:
        logger.error(f"CWFIS failed: {e}")  # added
        return None


def fetch_weather_data(lat=43.65, lon=-79.38):
    try:
        logger.info(f"Fetching weather for ({lat}, {lon})...")   # added

        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
               f"&timezone=America/Toronto&forecast_days=1")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'current' in data:
            logger.info(f"Weather: {data['current']['temperature_2m']}°C")  # added
            return data
        return None

    except Exception as e:
        logger.error(f"Weather failed: {e}")  # added
        return None
