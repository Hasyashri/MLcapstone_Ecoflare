# ===============================================
# File: dashboard/streamlit_dashboard.py
# Purpose: Interactive Streamlit dashboard for wildfire detection
# FIXED: Correct import paths
# ===============================================

import streamlit as st
import sys
import os
from datetime import datetime
import time
import pandas as pd

# CRITICAL FIX: Add correct paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
fire_detection_dir = os.path.join(parent_dir, 'modules', 'fire_detection')

# Add to Python path
sys.path.insert(0, parent_dir)
sys.path.insert(0, fire_detection_dir)

import logging

# -------------------------
# Logging setup
# -------------------------
logger = logging.getLogger("EcoFlareLogger")
logger.setLevel(logging.INFO)

# File handler - writes logs to app.log
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)

# Optional: console output for debugging
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Log app start
logger.info("EcoFlare Streamlit app started")


# Import directly from module files (not from 'modules' package)
try:
    from fetch_live_data import fetch_nasa_firms_data, fetch_cwfis_data, fetch_weather_data
    from iot_data import fetch_iot_sensor_data, analyze_iot_risk
    from vegetation_data import fetch_vegetation_data, get_vegetation_fire_risk
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error(f"Current directory: {current_dir}")
    st.error(f"Fire detection directory: {fire_detection_dir}")
    st.error(f"Python path: {sys.path[:3]}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🔥 EcoFlare - Ontario Wildfire Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .fire-detected {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 2rem;
        border-radius: 10px;
        border: 3px solid #FF4B4B;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .no-fire {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 2rem;
        border-radius: 10px;
        border: 3px solid #4CAF50;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔥 EcoFlare AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ontario Wildfire Detection & Management System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Location Selection
st.sidebar.header("📍 Location Settings")
location_preset = st.sidebar.selectbox(
    "Choose Location",
    ["Toronto", "Ottawa", "Thunder Bay", "Sault Ste. Marie", "Custom"],
    key="location_select"
)

# Preset coordinates
locations = {
    "Toronto": (43.65, -79.38),
    "Ottawa": (45.42, -75.70),
    "Thunder Bay": (48.38, -89.25),
    "Sault Ste. Marie": (46.49, -84.35)
}

if location_preset == "Custom":
    lat = st.sidebar.number_input("Latitude", value=43.65, format="%.4f", key="lat_input")
    lon = st.sidebar.number_input("Longitude", value=-79.38, format="%.4f", key="lon_input")
else:
    lat, lon = locations[location_preset]
    st.sidebar.info(f"📍 Coordinates: {lat}, {lon}")

# Manual refresh button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data Now", type="primary", key="refresh_btn"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About EcoFlare")
st.sidebar.info("""
**Multi-source wildfire detection system:**
- 🛰️ Satellite hotspots (NASA FIRMS)
- 🔥 Official fire reports (CWFIS)
- 🌡️ Weather conditions (Open-Meteo)
- 📡 IoT sensor data
- 🌲 Vegetation analysis

**Voting Logic:** Fire detected if 2+ sources confirm.
""")

# Main detection function
@st.cache_data(ttl=60, show_spinner=False)
def run_detection(lat, lon, location_name):
    """Run fire detection and return results"""
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'location': location_name,
        'coordinates': (lat, lon),
        'fire_detected': False,
        'fire_votes': 0,
        'total_votes': 4,
        'evidence': [],
        'data_sources': {}
    }
    
    # Fetch data
    nasa_data = fetch_nasa_firms_data()
    results['data_sources']['nasa'] = nasa_data
    
    cwfis_data = fetch_cwfis_data()
    results['data_sources']['cwfis'] = cwfis_data
    
    weather_data = fetch_weather_data(lat, lon)
    results['data_sources']['weather'] = weather_data
    
    iot_data = fetch_iot_sensor_data(location=location_name)
    results['data_sources']['iot'] = iot_data
    
    veg_data = fetch_vegetation_data(lat, lon)
    results['data_sources']['vegetation'] = veg_data
    
    # Voting logic
    fire_votes = 0
    
    # Vote 1: Satellite
    if nasa_data is not None and len(nasa_data) > 0:
        fire_votes += 1
        results['evidence'].append(f"🛰️ Satellite: {len(nasa_data)} hotspots detected")
    
    # Vote 2: Official
    if cwfis_data is not None and len(cwfis_data) > 0:
        fire_votes += 1
        results['evidence'].append(f"🔥 Official: {len(cwfis_data)} active fires reported")
    
    # Vote 3: Weather
    if weather_data and 'current' in weather_data:
        temp = weather_data['current']['temperature_2m']
        humidity = weather_data['current']['relative_humidity_2m']
        wind_speed = weather_data['current'].get('wind_speed_10m', 0)
        
        results['temperature'] = temp
        results['humidity'] = humidity
        results['wind_speed'] = wind_speed
        
        if temp > 30 and humidity < 30:
            fire_votes += 1
            results['evidence'].append(f"🌡️ Weather: Extreme conditions ({temp}°C, {humidity}% humidity)")
    
    # Vote 4: IoT
    iot_risk = analyze_iot_risk(iot_data)
    results['iot_risk'] = iot_risk
    if iot_risk in ['HIGH', 'MEDIUM'] or iot_data.get('flame_detected', False):
        fire_votes += 1
        results['evidence'].append(f"📡 IoT: {iot_risk} risk detected")
    
    # Vegetation
    veg_risk = get_vegetation_fire_risk(veg_data)
    results['vegetation_risk'] = veg_risk
    
    results['fire_votes'] = fire_votes
    results['fire_detected'] = fire_votes >= 2
    
    return results

# Run detection with loading spinner
with st.spinner('🔍 Running multi-source wildfire detection...'):
    results = run_detection(lat, lon, location_preset if location_preset != "Custom" else "Custom Location")

# Display timestamp
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption(f"🕐 Last updated: {results['timestamp']}")

# Main detection result
st.markdown("")
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    if results['fire_detected']:
        st.markdown(f"""
        <div class="fire-detected">
            <h1 style="margin:0; color:#d32f2f;">🔥 FIRE DETECTED!</h1>
            <h2 style="margin:1rem 0; color:#d32f2f;">Confidence: {results['fire_votes']}/{results['total_votes']} sources confirm</h2>
            <p style="font-size: 1.3rem; margin:0; color:#666;">📍 {results['location']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="no-fire">
            <h1 style="margin:0; color:#2e7d32;">✅ NO FIRE DETECTED</h1>
            <h2 style="margin:1rem 0; color:#2e7d32;">Detection Score: {results['fire_votes']}/{results['total_votes']}</h2>
            <p style="font-size: 1.3rem; margin:0; color:#666;">📍 {results['location']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Metrics row
st.subheader("📊 Real-time Detection Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    nasa_count = len(results['data_sources']['nasa']) if results['data_sources']['nasa'] is not None else 0
    st.metric(
        label="🛰️ Satellite Hotspots",
        value=nasa_count,
        delta="Active" if nasa_count > 0 else "None",
        delta_color="inverse" if nasa_count > 0 else "off"
    )

with col2:
    cwfis_count = len(results['data_sources']['cwfis']) if results['data_sources']['cwfis'] is not None else 0
    st.metric(
        label="🔥 Official Fires",
        value=cwfis_count,
        delta="Reported" if cwfis_count > 0 else "None",
        delta_color="inverse" if cwfis_count > 0 else "off"
    )

with col3:
    if 'temperature' in results:
        st.metric(
            label="🌡️ Temperature",
            value=f"{results['temperature']}°C",
            delta=f"{results['humidity']}% humidity"
        )
    else:
        st.metric(label="🌡️ Temperature", value="N/A")

with col4:
    iot_risk = results.get('iot_risk', 'UNKNOWN')
    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
    st.metric(
        label="📡 IoT Sensor Risk",
        value=f"{risk_emoji.get(iot_risk, '⚪')} {iot_risk}"
    )

st.markdown("---")

# Evidence section
st.subheader("🔍 Detection Evidence")
if results['evidence']:
    for evidence in results['evidence']:
        st.success(evidence)
else:
    st.info("✅ No fire indicators detected from any source")

# Vegetation risk
veg_risk = results.get('vegetation_risk', 'UNKNOWN')
veg_color = {"VERY_HIGH": "error", "HIGH": "warning", "MEDIUM": "info", "LOW": "success", "UNKNOWN": "info"}
getattr(st, veg_color.get(veg_risk, "info"))(f"🌲 Vegetation Fire Risk: **{veg_risk}**")

st.markdown("---")

# Detailed data sections
with st.expander("📡 View Detailed Source Data"):
    tab1, tab2, tab3, tab4 = st.tabs(["🛰️ Satellite", "🌡️ Weather", "📡 IoT", "🌲 Vegetation"])
    
    with tab1:
        st.subheader("NASA FIRMS Satellite Hotspots")
        if results['data_sources']['nasa'] is not None and len(results['data_sources']['nasa']) > 0:
            st.dataframe(results['data_sources']['nasa'].head(10), use_container_width=True)
            st.caption(f"Showing top 10 of {len(results['data_sources']['nasa'])} hotspots")
        else:
            st.info("No satellite hotspots detected in Ontario region")
    
    with tab2:
        st.subheader("Current Weather Conditions")
        if results['data_sources']['weather'] and 'current' in results['data_sources']['weather']:
            weather = results['data_sources']['weather']['current']
            st.json(weather)
        else:
            st.info("Weather data unavailable")
    
    with tab3:
        st.subheader("IoT Sensor Readings")
        if results['data_sources']['iot']:
            st.json(results['data_sources']['iot'])
        else:
            st.info("No IoT data available")
    
    with tab4:
        st.subheader("Vegetation & Land Cover Analysis")
        if results['data_sources']['vegetation']:
            st.json(results['data_sources']['vegetation'])
        else:
            st.info("No vegetation data available")

# Footer
st.markdown("---")
st.caption("🔥 EcoFlare AI - Wildfire Detection & Management System | Built with ❤️ for Ontario")
