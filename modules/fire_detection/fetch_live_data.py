# ===============================================
# File: modules/fire_detection/fetch_live_data.py
# Purpose: Fetch real-time fire and weather data
# ===============================================

import requests
import pandas as pd
from datetime import datetime
from io import StringIO

def fetch_nasa_firms_data():
    """Fetch NASA FIRMS satellite hotspots for Ontario"""
    try:
        url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Canada_24h.csv"
        print("🛰️ Fetching NASA FIRMS data...")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        ontario_fires = df[
            (df['longitude'] >= -95) & (df['longitude'] <= -74) &
            (df['latitude'] >= 41) & (df['latitude'] <= 57)
        ]
        
        print(f"✅ Found {len(ontario_fires)} hotspots")
        return ontario_fires
    except Exception as e:
        print(f"⚠️ NASA FIRMS failed: {e}")
        return None

def fetch_cwfis_data():
    """Fetch Canadian wildfire data"""
    try:
        url = "https://cwfis.cfs.nrcan.gc.ca/downloads/activefires/activefires.csv"
        print("🔥 Fetching CWFIS data...")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        if 'src_agency' in df.columns:
            ontario_fires = df[df['src_agency'] == 'ON']
        else:
            ontario_fires = df
        
        print(f"✅ Found {len(ontario_fires)} fires")
        return ontario_fires
    except Exception as e:
        print(f"⚠️ CWFIS failed: {e}")
        return None

def fetch_weather_data(lat=43.65, lon=-79.38):
    """Fetch weather data for location"""
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
               f"&timezone=America/Toronto&forecast_days=1")
        
        print(f"🌡️ Fetching weather for ({lat}, {lon})...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'current' in data:
            print(f"✅ Weather: {data['current']['temperature_2m']}°C")
            return data
        return None
    except Exception as e:
        print(f"⚠️ Weather failed: {e}")
        return None

if __name__ == "__main__":
    print("Testing data fetch...")
    fetch_nasa_firms_data()
    fetch_cwfis_data()
    fetch_weather_data()
