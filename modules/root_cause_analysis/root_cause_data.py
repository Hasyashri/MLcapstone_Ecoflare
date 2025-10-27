# root_cause_data.py
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Point

class RootCauseDataPipeline:
    def __init__(self, lightning_path, population_path, infrastructure_path, weather_api, veg_raster):
        self.lightning_path = lightning_path
        self.population_path = population_path
        self.infrastructure_path = infrastructure_path
        self.weather_api = weather_api
        self.veg_raster = veg_raster

    def ingest_realtime_fires(self, detection_data):
        """
        detection_data: list of dicts [{'lat': float, 'lon': float, 'timestamp': str}, ...]
        """
        df = pd.DataFrame(detection_data)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        gdf["timestamp"] = pd.to_datetime(gdf["timestamp"])
        return gdf

    def load_reference_layers(self):
        lightning = gpd.read_file(self.lightning_path)
        population = gpd.read_file(self.population_path)
        infrastructure = gpd.read_file(self.infrastructure_path)
        return lightning, population, infrastructure

    def spatial_temporal_join(self, fire_gdf, lightning, population, infrastructure):
        fire_gdf = fire_gdf.to_crs(lightning.crs)
        nearby_lightning = gpd.sjoin_nearest(fire_gdf, lightning, distance_col='lightning_dist')
        merged_pop = gpd.sjoin_nearest(nearby_lightning, population, distance_col='pop_dist')
        merged_final = gpd.sjoin_nearest(merged_pop, infrastructure, distance_col='infra_dist')
        return merged_final

    def extract_features(self, merged_gdf):
        merged_gdf["lightning_recent"] = (merged_gdf["lightning_time"] > (datetime.utcnow() - timedelta(hours=6))).astype(int)
        merged_gdf["infra_density"] = 1 / (merged_gdf["infra_dist"] + 1)
        merged_gdf["pop_density"] = merged_gdf["pop_value"]
        merged_gdf["hour"] = merged_gdf["timestamp"].dt.hour
        feature_cols = ["lightning_recent", "infra_density", "pop_density", "hour"]
        return merged_gdf[feature_cols], merged_gdf
