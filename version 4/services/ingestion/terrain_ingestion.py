#%%writefile services/ingestion/terrain_ingestion.py
# =============================================================
# File: services/ingestion/terrain_ingestion.py
# Purpose: Fetch DEM elevation data for wildfire terrain analysis
# Sources: Open-Meteo Elevation + NRCan CDEM [config/service_config.yaml]
# =============================================================

import requests
import pandas as pd
import zipfile
import io
from datetime import datetime, UTC
from pathlib import Path
from services.management.logging import get_logger
from services.ingestion.config_loader import ConfigLoader

# Project standard directory (matches firms/iot/weather)
DATA_DIR = Path("data/terrain")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Production logging + config
logger = get_logger("TerrainIngestion")
config = ConfigLoader()

# Config-driven URLs
OPEN_METEO_ELEVATION = config.get('ingestion', 'terrain', 'open_meteo_elevation')
NRCAN_CDEM_ZIP = config.get('ingestion', 'terrain', 'nrcan_cdem_zip')

# Canada locations (matches other modules)
CANADA_LOCATIONS = {
    "Toronto": {"lat": 43.6532, "lon": -79.3832, "location": "Toronto"},
    "Vancouver": {"lat": 49.2827, "lon": -123.1207, "location": "Vancouver"},
    "Calgary": {"lat": 51.0447, "lon": -114.0719, "location": "Calgary"},
    "Ottawa": {"lat": 45.4215, "lon": -75.6972, "location": "Ottawa"},
    "Yellowknife": {"lat": 62.4540, "lon": -114.3718, "location": "Yellowknife"},
}


def fetch_global_dem(lat=45.4215, lon=-75.6972, location="Ottawa"):
    """
    Fetch precise elevation for any coordinate using Open-Meteo DEM API.
    Critical for wildfire slope/terrain analysis.
    
    Args:
        lat, lon: Coordinates (Canada-wide)
        location: Human-readable name for filename
    
    Returns:
        pd.DataFrame: elevation data
    """
    print(f"🗻 Fetching elevation for {location} ({lat:.4f}, {lon:.4f})...")
    logger.info(f"DEM fetch: {location} ({lat}, {lon})")
    
    params = {"latitude": lat, "longitude": lon}
    
    try:
        response = requests.get(OPEN_METEO_ELEVATION, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        elevation = data.get("elevation", [None])[0]
        if elevation is None:
            print("⚠️ No elevation data returned")
            return None
        
        # Pandas DataFrame (project standard)
        df = pd.DataFrame([{
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elevation,
            "timestamp": datetime.now(UTC)
        }])
        
        filename = DATA_DIR / f"dem_{location}_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"💾 Saved elevation {elevation:.1f}m to {filename.name}")
        logger.info(f"SAVED DEM {location}: {elevation}m → {filename.name}")
        return df
        
    except Exception as e:
        print(f"⚠️ Global DEM failed: {e}")
        logger.error(f"DEM failed for {location}: {e}")
        
        # FALLBACK: Load latest local CSV
        local_files = sorted(DATA_DIR.glob(f"dem_{location}_*.csv"), reverse=True)
        if local_files:
            print(f"📂 Fallback DEM: {local_files[0].name}")
            return pd.read_csv(local_files[0])
        return None


def fetch_nrcan_cdem():
    """
    Download NRCan national CDEM (2.5m resolution DEM for all Canada).
    Stable ZIP - extracts GeoTIFFs for GIS analysis.
    """
    print("🇨🇦 Fetching NRCan national CDEM (2.5m DEM)...")
    logger.info("NRCAN CDEM national download started")
    
    try:
        response = requests.get(NRCAN_CDEM_ZIP, timeout=120, stream=True)
        response.raise_for_status()
        
        # Streaming ZIP extraction
        zip_bytes = io.BytesIO(response.content)
        timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M')
        extract_dir = DATA_DIR / f"nrcan_cdem_{timestamp}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_bytes) as z:
            z.extractall(extract_dir)
        
        # Log extracted files
        extracted_files = list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.tiff"))
        print(f"💾 Extracted {len(extracted_files)} CDEM tiles to {extract_dir.name}")
        logger.info(f"SAVED NRCAN CDEM: {len(extracted_files)} tiles → {extract_dir}")
        return extract_dir
        
    except Exception as e:
        print(f"⚠️ NRCan CDEM failed: {e}")
        logger.error(f"CDEM download failed: {e}")
        return None


if __name__ == "__main__":
    print("🚀 TERRAIN INGESTION STARTED")
    print("📁 Saving to: data/terrain/")
    
    # Test elevation APIs
    print("🗻 Point elevations...")
    fetch_global_dem(**CANADA_LOCATIONS["Toronto"])
    fetch_global_dem(**CANADA_LOCATIONS["Yellowknife"])
    
    # National DEM
    print("\n🇨🇦 National CDEM...")
    cdem_dir = fetch_nrcan_cdem()
    
    print("\n📁 All terrain files:")
    for f in DATA_DIR.glob("*.csv"):
        print(f"   📄 {f.name}")
    if cdem_dir:
        print(f"   📁 {cdem_dir.name}/ (CDEM tiles)")
    
    print("\n🎉 TERRAIN INGESTION COMPLETE!")
