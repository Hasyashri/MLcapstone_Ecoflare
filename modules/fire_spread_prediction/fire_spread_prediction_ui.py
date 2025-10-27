import streamlit as st
from modules.fire_spread_prediction.spread_model import enhanced_predict_fire_spread
from modules.fire_spread_prediction.ml_integration import FireSpreadMLModel
from modules.fire_spread_prediction.feature_engineering import construct_features
from streamlit_folium import st_folium
import folium

# Load ML model - path configurable for future updates
model_path = "ml_models/fire_spread_model.pkl"
preprocessor_path = "ml_models/preprocessor.pkl"
ml_model = FireSpreadMLModel(model_path, preprocessor_path)

def show_fire_spread_prediction(real_time_data):
    """
    Show fire spread prediction panel with real-time data inputs,
    integrating root cause outputs and allowing easy parameter updates.
    """

    st.header("🧯 Fire Spread Prediction Module")

    fire_location = real_time_data['fire_location']
    wind = real_time_data.get('wind_data', {'speed': 10, 'direction': 45})
    vegetation = real_time_data.get('vegetation_data', {'type': 'mixed_forest', 'moisture': 30})
    root_cause = real_time_data.get('root_cause', 'unknown')
    fire_timestamp = real_time_data.get('detection_time')

    # Optional user control for time horizon to predict
    time_horizon_hours = st.slider("Prediction Time Horizon (hours)", 1, 24, 12)

    # Feature engineering
    features = construct_features(fire_location, wind, vegetation, vegetation['moisture'],
                                  root_cause, fire_timestamp)

    # Get prediction from spread model
    prediction = enhanced_predict_fire_spread(
        fire_location, wind, vegetation, vegetation['moisture'],
        root_cause, fire_timestamp,
        ml_model=ml_model, time_horizon_hours=time_horizon_hours
    )

    # Display risk level and spread rate
    st.metric("Fire Spread Rate (km/h)", prediction['spread_rate_kmh'])
    st.metric("Risk Level", prediction['risk_level'])
    st.metric("Dominant Wind Direction (°)", prediction['dominant_direction_deg'])

    # Map Visualization
    m = folium.Map(location=fire_location, zoom_start=8)
    folium.Marker(location=fire_location, tooltip="Fire Origin",
                  icon=folium.Icon(color="red", icon="fire")).add_to(m)

    for pred in prediction['predictions']:
        folium.Circle(
            location=(pred['predicted_head_lat'], pred['predicted_head_lon']),
            radius=pred['estimated_area_ha'] * 10,
            color='orange',
            fill=True,
            fill_opacity=0.4,
            popup=(f"Hour: {pred['hour']}<br>Area: {int(pred['estimated_area_ha'])} ha<br>"
                   f"Confidence: {pred['confidence']:.2f}")
        ).add_to(m)

    st_folium(m, width=700, height=450)

