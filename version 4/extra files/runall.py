import subprocess
import sys

# ------------------------------
# 1️⃣ INGESTION MODULES
# ------------------------------
ingestion_modules = [
    "services.ingestion.satellite_ingestion",
    "services.ingestion.iot_ingestion",
    "services.ingestion.weather_ingestion",
    "services.ingestion.terrain_ingestion",
    "services.ingestion.vegetation_ingestion"
]

print("====================================================")
print("🚀 RUNNING INGESTION MODULES")
print("====================================================\n")

for module in ingestion_modules:
    print(f"🛰️ RUNNING {module} ...")
    try:
        subprocess.run([sys.executable, "-m", module], check=True)
        print(f"✅ {module} COMPLETED\n")
    except subprocess.CalledProcessError:
        print(f"⚠️ {module} FAILED, continuing with next...\n")

# ------------------------------
# 2️⃣ FEATURE MODULES
# ------------------------------
feature_modules = [
    "services.features.feature_builder",
    "services.features.dynamic_feature_weighting",
    "services.features.ensemble_features",
    "services.features.uncertainty_propagation",
    "services.features.spatial_smoothing"
]

print("====================================================")
print("🚀 RUNNING FEATURE MODULES")
print("====================================================\n")

for module in feature_modules:
    print(f"🌟 RUNNING {module} ...")
    try:
        subprocess.run([sys.executable, "-m", module], check=True)
        print(f"✅ {module} COMPLETED\n")
    except subprocess.CalledProcessError:
        print(f"⚠️ {module} FAILED, continuing with next...\n")

print("====================================================")
print("🎉 PIPELINE FINISHED: INGESTION + FEATURE MODULES")
print("====================================================")
# %%writefile services/ingestion/weather_ingestion.py
"""
 EcoFlare Weather Ingestion - Fetches weather data from Open-Meteo API
 Auto-saves hourly forecasts to data/weather/
 Falls back to latest local CSV on failure
 Works 100% - tested Nov 30, 2025
"""