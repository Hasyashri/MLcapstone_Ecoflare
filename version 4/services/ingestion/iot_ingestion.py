# %%writefile services/ingestion/iot_ingestion.py
# =============================================================
# File: services/ingestion/iot_ingestion.py
# Purpose: Fetch live IoT / air quality sensor data
# Source: Open-Meteo Air Quality API (Free, No Token)
# Canada-wide coverage
# =============================================================

import requests
import pandas as pd
from datetime import datetime, UTC
from io import StringIO
from pathlib import Path
from services.management.logging import get_logger
from services.ingestion.config_loader import ConfigLoader

# Project standard directory
DATA_DIR = Path("data/iot")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Production logging
logger = get_logger("IoTIngestion")

# Config integration (future-proof)
config = ConfigLoader()


def fetch_open_meteo_air_quality(lat=43.6532, lon=-79.3832, location="Toronto"):
    """
    Fetch real-time air quality data using Open-Meteo Air Quality API.
    - NO API KEY required
    - Canada-wide: any lat/lon coordinates
    - PM2.5, PM10, O3, NO2, SO2 + aerosol optical depth (smoke proxy)
    - Fallback: load most recent local file

    Args:
        lat (float): Latitude (-90 to 90)
        lon (float): Longitude (-180 to 180)
        location (str): Human-readable name for logging

    Returns:
        pd.DataFrame: Hourly air quality readings
    """
    print(f"🌫️ Fetching air quality for {location} ({lat:.4f}, {lon:.4f})...")
    logger.info(f"Air quality fetch: {location} ({lat}, {lon})")

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,"
        "ozone,aerosol_optical_depth,dust"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        # Convert hourly readings to DataFrame
        df = pd.DataFrame(data["hourly"])
        df["timestamp"] = pd.to_datetime(df["time"])
        df["location"] = location
        df["lat"] = lat
        df["lon"] = lon

        # Timestamped filename
        filename = DATA_DIR / f"air_quality_{location}_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)

        print(f"💾 Saved {len(df)} hourly readings to {filename.name}")
        print(f"✅ Latest PM2.5: {df['pm2_5'].iloc[-1]:.1f} µg/m³")
        logger.info(f"SAVED AQ {location}: {len(df)} records → {filename.name}")

        return df

    except Exception as e:
        print(f"⚠️ Failed to fetch Open-Meteo AQ: {e}")
        logger.error(f"Air quality failed for {location}: {e}")

        # FALLBACK: Load latest local CSV
        local_files = sorted(DATA_DIR.glob(f"air_quality_{location}_*.csv"), reverse=True)
        if local_files:
            print(f"📂 Loading fallback: {local_files[0].name}")
            logger.info(f"AQ fallback {location}: {local_files[0]}")
            return pd.read_csv(local_files[0])
        else:
            print("❌ No local air quality fallback available")
            return None


# CANADA CITIES - Easy presets
CANADA_LOCATIONS = {
    "Toronto": {"lat": 43.6532, "lon": -79.3832, "location": "Toronto"},
    "Vancouver": {"lat": 49.2827, "lon": -123.1207, "location": "Vancouver"},
    "Calgary": {"lat": 51.0447, "lon": -114.0719, "location": "Calgary"},
    "Ottawa": {"lat": 45.4215, "lon": -75.6972, "location": "Ottawa"},
    "Montreal": {"lat": 45.5017, "lon": -73.5673, "location": "Montreal"},
    "Winnipeg": {"lat": 49.8951, "lon": -97.1384, "location": "Winnipeg"},
    "Halifax": {"lat": 44.6488, "lon": -63.5752, "location": "Halifax"},
    "Yellowknife": {"lat": 62.4540, "lon": -114.3718, "location": "Yellowknife"},  # NWT wildfire zone
}


if __name__ == "__main__":
    print("🚀 IoT AIR QUALITY INGESTION STARTED")
    print("📁 Saving to: data/iot/")

    # Test Toronto (default)
    df_toronto = fetch_open_meteo_air_quality(**CANADA_LOCATIONS["Toronto"])

    # Test wildfire zone (Yellowknife)
    print("\n🌲 Testing wildfire zone...")
    df_nwt = fetch_open_meteo_air_quality(**CANADA_LOCATIONS["Yellowknife"])

    print("\n📁 All files:")
    for f in DATA_DIR.glob("*.csv"):
        print(f"   📄 {f.name}")

    print("\n🎉 IoT INGESTION COMPLETE!")
