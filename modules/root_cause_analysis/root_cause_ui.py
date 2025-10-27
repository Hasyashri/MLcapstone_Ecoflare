# root_cause_ui.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from root_cause_data import RootCauseDataPipeline
from root_cause_classifier import RootCauseClassifier
import pandas as pd
import numpy as np

def run_root_cause_ui(live_fire_data):
    st.title("🔥 EcoFlare: Root Cause Analysis")

    pipeline = RootCauseDataPipeline(
        lightning_path="data/lightning.geojson",
        population_path="data/population.geojson",
        infrastructure_path="data/infrastructure.geojson",
        weather_api="https://api.weatherdata.local",
        veg_raster="data/vegetation.tif"
    )
    classifier = RootCauseClassifier(model_path="models/root_cause_model.pkl")

    fire_gdf = pipeline.ingest_realtime_fires(live_fire_data)
    lightning, pop, infra = pipeline.load_reference_layers()
    merged = pipeline.spatial_temporal_join(fire_gdf, lightning, pop, infra)
    X, enriched = pipeline.extract_features(merged)

    preds, probs, shap_values = classifier.predict(X)
    enriched["pred_cause"] = preds
    enriched["confidence"] = np.max(probs, axis=1)

    st.subheader("Detected Fire Spots and Predicted Causes")
    m = folium.Map(location=[enriched.lat.mean(), enriched.lon.mean()], zoom_start=6)
    for _, row in enriched.iterrows():
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color="red",
            fill=True,
            popup=f"Cause: {row.pred_cause}<br>Confidence: {row.confidence:.2f}"
        ).add_to(m)
    st_folium(m, width=800, height=500)

    st.download_button(
        "Download Root Cause Report",
        enriched.to_csv(index=False).encode("utf-8"),
        file_name="root_cause_predictions.csv",
        mime="text/csv"
    )
