# modules/root_cause_analysis/root_cause_ui.py

"""
Streamlit UI helpers for MVP 3.
Call `render_root_cause_tab()` from your main dashboard file.
"""

import streamlit as st
from .root_cause_classifier import predict_root_cause


def render_root_cause_tab():
    st.header("MVP 3 – Root Cause Analysis")

    col1, col2 = st.columns(2)
    with col1:
        vote_count = st.slider("Detection vote count", 0, 5, 3)
        temp = st.slider("Temperature (°C)", -10.0, 45.0, 30.0)
        wind = st.slider("Wind (km/h)", 0.0, 60.0, 20.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 40.0)

    with col2:
        power_lines = st.checkbox("Near power lines?")
        pop_density = st.slider("Population density", 0.0, 500.0, 50.0)
        lightning = st.slider("Recent lightning strikes", 0.0, 20.0, 2.0)

    if st.button("Predict Root Cause"):
        detection_data = {
            "vote_count": vote_count,
            "weather_temp": temp,
            "weather_wind": wind,
            "weather_humidity": humidity,
            "near_power_lines": 1.0 if power_lines else 0.0,
            "population_density": pop_density,
            "recent_lightning_strikes": lightning,
        }
        result = predict_root_cause(detection_data)
        st.json(result)
