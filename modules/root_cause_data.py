# ===============================================
# File: modules/root_cause_analysis/root_cause_data.py
# Purpose: Data ingestion, enrichment, and feature engineering
# ===============================================

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import numpy as np

class RootCauseDataPipeline:
    """Pipeline for processing detected fire spots for RCA"""

    def __init__(self):
        pass

    def create_fire_gdf(self, fire_events):
        """
        fire_events: list of dicts from detection process
        Example:
        [{'lat': 43.65, 'lon': -79.38, 'timestamp': '2025-10-26T12:30:00'}]
        """
        if not fire_events:
            return gpd.GeoDataFrame(columns=["lat", "lon", "timestamp"])
        df = pd.DataFrame(fire_events)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        return gdf

    def engineer_features(self, gdf, weather_data=None, iot_data=None, vegetation_data=None):
        """Enrich fire spots with interpretable features"""
        if gdf.empty:
            return pd.DataFrame(), gdf

        # Base features
        gdf["hour"] = gdf["timestamp"].dt.hour
        gdf["day_of_week"] = gdf["timestamp"].dt.dayofweek

        # Environmental + weather influence
        if weather_data and "current" in weather_data:
            curr = weather_data["current"]
            gdf["temperature"] = curr.get("temperature_2m", np.nan)
            gdf["humidity"] = curr.get("relative_humidity_2m", np.nan)
            gdf["wind_speed"] = curr.get("wind_speed_10m", np.nan)
        else:
            gdf["temperature"] = np.nan
            gdf["humidity"] = np.nan
            gdf["wind_speed"] = np.nan

        # IoT layer
        if iot_data:
            gdf["iot_temp"] = iot_data.get("temperature", np.nan)
            gdf["smoke_level"] = iot_data.get("smoke_level", np.nan)
            gdf["flame_detected"] = int(iot_data.get("flame_detected", False))
        else:
            gdf["iot_temp"] = np.nan
            gdf["smoke_level"] = np.nan
            gdf["flame_detected"] = 0

        # Vegetation type influence
        if vegetation_data:
            veg_risk = vegetation_data.get("fire_risk_areas", [])
            gdf["vegetation_influence"] = len(veg_risk)
        else:
            gdf["vegetation_influence"] = 0

        features = gdf[["hour", "day_of_week", "temperature", "humidity",
                        "wind_speed", "iot_temp", "smoke_level",
                        "flame_detected", "vegetation_influence"]]

        return features, gdf
