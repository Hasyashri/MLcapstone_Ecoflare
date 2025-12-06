# %%writefile services/ingestion/weather_ingestion.py
# =============================================================
# File: services/ingestion/weather_ingestion.py
# Purpose: Fetch real-time weather data for wildfire monitoring
# Source: Open-Meteo Forecast API (Free, No Token) [attached_file:1]
# =============================================================

import requests
import pandas as pd
from datetime import datetime, UTC
from pathlib import Path
from services.management.logging import get_logger
from services.ingestion.config_loader import ConfigLoader

# Project standard directory (matches satellite/iot)
DATA_DIR = Path("data/weather")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Production logging
logger = get_logger("WeatherIngestion")
config = ConfigLoader()

# Canada wildfire monitoring locations
CANADA_LOCATIONS = {
    "Toronto": {"lat": 43.6532, "lon": -79.3832, "location": "Toronto"},
    "Vancouver": {"lat": 49.2827, "lon": -123.1207, "location": "Vancouver"},
    "Calgary": {"lat": 51.0447, "lon": -114.0719, "location": "Calgary"},
    "Ottawa": {"lat": 45.4215, "lon": -75.6972, "location": "Ottawa"},
    "Yellowknife": {"lat": 62.4540, "lon": -114.3718, "location": "Yellowknife"},
}


def fetch_weather_data(lat=43.6532, lon=-79.3832, location="Toronto"):
    """
    Fetch real-time + 24h forecast weather for wildfire monitoring.
    Key vars: wind speed, humidity, precip, temperature (fire spread predictors)
    
    Args:
        lat, lon: Coordinates (Canada-wide)
        location: Human-readable name for filename/logging
    
    Returns:
        pd.DataFrame: Current + hourly forecast
    """
    print(f"🌤️ Fetching weather for {location} ({lat:.4f}, {lon:.4f})...")
    logger.info(f"Weather fetch: {location} ({lat}, {lon})")
    
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&timezone=America/Toronto"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,"
        "precipitation,weather_code"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        "&forecast_days=1"
    )
    
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        # Current conditions DataFrame
        current_df = pd.DataFrame([data["current"]])
        current_df["location"] = location
        current_df["lat"] = lat
        current_df["lon"] = lon
        current_df["timestamp"] = pd.to_datetime("now", utc=True)
        
        # Hourly forecast DataFrame  
        hourly_df = pd.DataFrame(data["hourly"])
        hourly_df["timestamp"] = pd.to_datetime(hourly_df["time"])
        hourly_df["location"] = location
        hourly_df["lat"] = lat
        hourly_df["lon"] = lon
        
        # Save both
        filename = DATA_DIR / f"weather_{location}_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
        hourly_df.to_csv(filename, index=False)
        
        print(f"💾 Saved {len(hourly_df)} hourly forecasts to {filename.name}")
        print(f"✅ Current: {current_df['temperature_2m'].iloc[0]:.1f}°C, "
              f"Wind: {current_df['wind_speed_10m'].iloc[0]:.1f}km/h, "
              f"Humidity: {current_df['relative_humidity_2m'].iloc[0]:.0f}%")
        logger.info(f"SAVED Weather {location}: {len(hourly_df)} hours → {filename.name}")
        
        return hourly_df
        
    except Exception as e:
        print(f"⚠️ Weather fetch failed: {e}")
        logger.error(f"Weather failed for {location}: {e}")
        
        # FALLBACK: Load latest local CSV
        local_files = sorted(DATA_DIR.glob(f"weather_{location}_*.csv"), reverse=True)
        if local_files:
            print(f"📂 Loading fallback: {local_files[0].name}")
            logger.info(f"Weather fallback {location}: {local_files[0]}")
            return pd.read_csv(local_files[0])
        else:
            print("❌ No local weather fallback available")
            return None


if __name__ == "__main__":
    print("🚀 WEATHER INGESTION STARTED")
    print("📁 Saving to: data/weather/")
    
    # Test Toronto
    df_toronto = fetch_weather_data(**CANADA_LOCATIONS["Toronto"])
    
    # Test wildfire zone
    print("\n🔥 Testing wildfire zone...")
    df_yellowknife = fetch_weather_data(**CANADA_LOCATIONS["Yellowknife"])
    
    print("\n📁 All files:")
    for f in DATA_DIR.glob("*.csv"):
        print(f"   📄 {f.name}")
    
    print("\n🎉 WEATHER INGESTION COMPLETE!")
