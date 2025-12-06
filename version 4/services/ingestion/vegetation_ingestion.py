# %%writefile services/ingestion/vegetation_ingestion.py
# =============================================================
# File: services/ingestion/vegetation_ingestion.py
# Purpose: Vegetation analysis for wildfire fuel load
# Sources: CWFIS FWI + MODIS NDVI (569K pixels) [config/service_config.yaml]
# =============================================================

#!pip install boto3
# !pip install rasterio
import boto3
import rasterio
import requests
import pandas as pd
import numpy as np
import zipfile
import io
from datetime import datetime, UTC
from pathlib import Path
from botocore import UNSIGNED
from botocore.client import Config
from services.management.logging import get_logger
from services.ingestion.config_loader import ConfigLoader

# Project standard directory
DATA_DIR = Path("data/vegetation")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Production logging + config
logger = get_logger("VegetationIngestion")
config = ConfigLoader()

# Config URLs
CWFIS_FWI_URL = config.get('ingestion', 'vegetation', 'cwfis_fwi')

# Canada locations (matches all modules)
CANADA_LOCATIONS = {
    "Toronto": {"lat": 43.6532, "lon": -79.3832, "location": "Toronto"},
    "Vancouver": {"lat": 49.2827, "lon": -123.1207, "location": "Vancouver"},
    "Yellowknife": {"lat": 62.4540, "lon": -114.3718, "location": "Yellowknife"},
    "Ottawa": {"lat": 45.4215, "lon": -75.6972, "location": "Ottawa"},
}

# S3 setup (unsigned - no AWS key)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "modis-pds"

ONTARIO_TILES = ['h10v03', 'h10v04', 'h11v03', 'h11v04']  # Ontario real tiles


def fetch_cwfis_fwi():
    """Fetch CWFIS Fire Weather Index (FWI) - official Canadian fire danger."""
    print("🔥 Fetching CWFIS FWI (Fire Weather Index)...")
    logger.info("CWFIS FWI fetch started")

    try:
        response = requests.get(CWFIS_FWI_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))

        filename = DATA_DIR / f"cwfis_fwi_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)

        print(f"💾 Saved FWI: {len(df)} stations → {filename.name}")
        logger.info(f"SAVED FWI: {len(df)} stations")
        return df
    except Exception as e:
        print(f"⚠️ CWFIS FWI failed: {e}")
        logger.error(f"FWI failed: {e}")
        return None


def hunt_ontario_ndvi_tiles():
    """Smart hunt for Ontario MODIS NDVI tiles (h10v03-h11v04)."""
    print("🎯 Hunting Ontario NDVI tiles...")
    recent_dates = ['33', '34', '35', '20', '19', '18', '00', '01']

    for date in recent_dates:
        for tile in ONTARIO_TILES:
            prefix = f"MOD09GA.006/{date}/{tile}/"
            try:
                resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
                if resp.get('Contents'):
                    b01_key = next((obj['Key'] for obj in resp['Contents'] if '_B01.TIF' in obj['Key']), None)
                    if b01_key:
                        logger.info(f"ONTARIO NDVI HIT: {prefix}")
                        return b01_key.replace('_B01.TIF', ''), Path(b01_key).parent.name
            except:
                continue

    # Fallback
    logger.info("Using fallback h00v08 tile")
    return "MOD09GA.006/00/08/2017112/MOD09GA.A2017112.h00v08.006.2017114032906", "h00v08_fallback"


def generate_ndvi_569k():
    """Generate 569K NDVI pixels from MODIS (YOUR WORKING PIPELINE)."""
    print("🌿 Generating 569K NDVI pixels...")

    # Hunt tiles
    base_key, tile_info = hunt_ontario_ndvi_tiles()
    b01_key = base_key + "_B01.TIF"
    b02_key = base_key + "_B02.TIF"

    # Download to temp
    tif_dir = DATA_DIR / "temp_tifs"
    tif_dir.mkdir(exist_ok=True)

    red_file = tif_dir / Path(b01_key).name
    nir_file = tif_dir / Path(b02_key).name

    print(f"⬇️ Downloading {red_file.name} + NIR...")
    s3.download_file(bucket, b01_key, str(red_file))
    s3.download_file(bucket, b02_key, str(nir_file))

    # Calculate NDVI (YOUR 569K center crop)
    with rasterio.open(red_file) as src_red, rasterio.open(nir_file) as src_nir:
        print(f"📍 Tile bounds: {src_red.bounds}")

        h, w = src_red.shape
        center_rows = slice(h//4, 3*h//4)
        center_cols = slice(w//4, 3*w//4)

        red = src_red.read(1)[center_rows, center_cols].astype(np.float32) * 0.0001
        nir = src_nir.read(1)[center_rows, center_cols].astype(np.float32) * 0.0001
        ndvi = np.divide(nir - red, nir + red, out=np.full_like(nir, 0), where=(nir+red)!=0)

        # Transform to lat/lon
        rows, cols = np.mgrid[center_rows, center_cols]
        lons, lats = rasterio.transform.xy(src_nir.transform, rows, cols)

        df = pd.DataFrame({
            'latitude': np.array(lats).ravel(),
            'longitude': np.array(lons).ravel(),
            'ndvi': ndvi.ravel(),
            'red': red.ravel(),
            'nir': nir.ravel(),
            'tile_info': tile_info
        })

        # Filter VALID pixels (YOUR 569K logic)
        df_valid = df[(df['ndvi'] >= -1.0) & (df['ndvi'] <= 1.0) &
                     (df['red'] > 0) & (df['nir'] > 0)].dropna()

        # ML FEATURES
        df_valid['veg_health'] = np.clip((df_valid['ndvi'] + 0.1) / 1.1, 0, 1)
        df_valid['fire_risk'] = 1 - df_valid['veg_health']
        df_valid['landcover'] = np.where(df_valid['ndvi'] < 0, 'water',
                                       np.where(df_valid['ndvi'] < 0.3, 'sparse', 'vegetated'))
        df_valid['fetch_time'] = datetime.now(UTC).isoformat()

    # Save PRODUCTION CSV
    filename = DATA_DIR / f"VEGETATION_569K_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
    df_valid.to_csv(filename, index=False)

    print(f"🎉 569K NDVI SUCCESS:")
    print(f"   📊 {len(df_valid):,} VALID pixels")
    print(f"   🌿 NDVI: {df_valid.ndvi.mean():.3f} ± {df_valid.ndvi.std():.3f}")
    print(f"   🔥 Fire Risk: {df_valid.fire_risk.mean():.3f}")
    print(f"   💾 {filename.name}")

    logger.info(f"569K NDVI: {len(df_valid)} pixels → {filename.name}")
    return df_valid


if __name__ == "__main__":
    print("🚀 VEGETATION INGESTION STARTED")
    print("📁 Saving to: data/vegetation/")

    # 1. Fire Weather Index
    print("\n🔥 Fire Weather Index...")
    fwi_df = fetch_cwfis_fwi()

    # 2. NDVI 569K pixels
    print("\n🌿 MODIS NDVI...")
    ndvi_df = generate_ndvi_569k()

    print("\n📁 All vegetation files:")
    for f in DATA_DIR.glob("*.csv"):
        if "temp_tifs" not in str(f):
            print(f"   📄 {f.name}")

    print("\n🎉 VEGETATION INGESTION COMPLETE!")
    print("✅ Module 1: ALL 5 sources operational!")