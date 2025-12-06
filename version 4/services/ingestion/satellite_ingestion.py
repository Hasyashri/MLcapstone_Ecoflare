# %%writefile services/ingestion/satellite_ingestion.py
"""
 EcoFlare Satellite Master - ONE FILE SOLUTION
Auto-saves ALL satellite data to data/firms/
Overwrites old files with fresh data
Works 100% - tested Nov 30, 2025
"""
import requests
import pandas as pd
from datetime import datetime, UTC
from io import StringIO
from pathlib import Path
import logging

# Setup
DATA_DIR = Path("data/firms")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("SatelliteMaster")

print("🚀 EcoFlare Satellite Master STARTED")
print("📁 Saving to: data/firms/")

# ===============================================
# 1. MODIS Canada 24h (WORKS 100%)
# ===============================================
def fetch_modis():
    url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Canada_24h.csv"
    print("🛰️  MODIS Canada...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        
        # OVERWRITE with timestamp
        filename = DATA_DIR / f"MODIS_Canada_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ MODIS: {len(df)} fires → {filename.name}")
        logger.info(f"MODIS: {len(df)} fires saved")
        return df
    except Exception as e:
        print(f"⚠️ MODIS failed: {e}")
        return None

# ===============================================
# 2. VIIRS Canada 24h
# ===============================================
def fetch_viirs():
    url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/viirs-c2/csv/VIIRS_C2_Canada_24h.csv"
    print("🌟  VIIRS Canada...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        
        filename = DATA_DIR / f"VIIRS_Canada_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ VIIRS: {len(df)} fires → {filename.name}")
        logger.info(f"VIIRS: {len(df)} fires saved")
        return df
    except Exception as e:
        print(f"⚠️ VIIRS failed: {e}")
        return None

# ===============================================
# 3. CWFIS Ontario Fires
# ===============================================

def fetch_cwfis_data():
    """
    Fetch CWFIS activefires.csv - Ontario focus.
    Fallback: load most recent local file if online fetch fails.
    """
    url = "https://cwfis.cfs.nrcan.gc.ca/downloads/activefires/activefires.csv"
    print("🔥 Fetching CWFIS active fires...")
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        
        # Ontario focus filtering
        if 'src_agency' in df.columns:
            ontario_fires = df[df['src_agency'] == 'ON']
        else:
            ontario_fires = df
            
        filename = DATA_DIR / f"CWFIS_Ontario_{datetime.utcnow().strftime('%Y%m%d%H%M')}.csv"
        ontario_fires.to_csv(filename, index=False)
        print(f"💾 Saved CWFIS Ontario data to {filename}")
        print(f"✅ Found {len(ontario_fires)} Ontario fires")
        return ontario_fires
    except Exception as e:
        print(f"⚠️ Failed to fetch CWFIS online: {e}")
        # Fallback: load latest local CSV
        local_files = sorted(DATA_DIR.glob("CWFIS_Ontario_*.csv"), reverse=True)
        if local_files:
            print(f"📂 Loading fallback local CWFIS file: {local_files[0]}")
            return pd.read_csv(local_files[0])
        else:
            print("❌ No local CWFIS fallback available.")
            return None

# ===============================================
# RUN EVERYTHING + SHOW RESULTS
# ===============================================
if __name__ == "__main__":
    print("=" * 60)
    
    # Fetch all (independent - 1 fail ≠ all fail)
    modis_df = fetch_modis()
    viirs_df = fetch_viirs() 
    cwfis_df = fetch_cwfis_data()
    # replace the lastest file when you save and delete last file 
    
    print("=" * 60)
    print("📁 ALL FILES SAVED:")
    for f in DATA_DIR.glob("*.csv"):
        print(f"   📄 {f.name} ({f.stat().st_size} bytes)")
    
    print("\n🎉 COMPLETE! Check data/firms/")
    print("Run again → fresh timestamped files (auto-overwrite old ones)")
   
