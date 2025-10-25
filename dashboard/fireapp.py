# ===============================================
# File: dashboard/fireapp.py
# EcoFlare Fire Detection App - Kid-Friendly Version
# Simple language, animations, interactive graphics
# ===============================================

import streamlit as st
import sys
import os
from datetime import datetime
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
fire_detection_dir = os.path.join(parent_dir, 'modules', 'fire_detection')

sys.path.insert(0, parent_dir)
sys.path.insert(0, fire_detection_dir)

# Import modules
from fetch_live_data import fetch_nasa_firms_data, fetch_cwfis_data, fetch_weather_data
from iot_data import fetch_iot_sensor_data, analyze_iot_risk
from vegetation_data import fetch_vegetation_data, get_vegetation_fire_risk

# Page config
st.set_page_config(
    page_title="🔥 FireApp - Easy Fire Detector",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .big-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 2s ease-in-out infinite;
        margin-bottom: 0.5rem;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .fire-alert {
        background: linear-gradient(135deg, #ff4444 0%, #ff6666 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(255, 68, 68, 0.4);
        animation: shake 0.5s infinite;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    .safe-alert {
        background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.4);
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .step-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown('<h1 class="big-title">🔥 FireApp</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your Smart Fire Detective! 🕵️‍♂️</p>', unsafe_allow_html=True)

# Sidebar - Simple controls
st.sidebar.title("🎮 Control Panel")
st.sidebar.markdown("---")

# Simple location picker
st.sidebar.subheader("📍 Where should I check?")
city = st.sidebar.selectbox(
    "Pick a city:",
    ["Toronto 🏙️", "Ottawa 🏛️", "Thunder Bay ⛰️", "Sault Ste. Marie 🌊", "Custom Location 🗺️"],
    key="city_select"
)

# City coordinates
city_coords = {
    "Toronto 🏙️": (43.65, -79.38),
    "Ottawa 🏛️": (45.42, -75.70),
    "Thunder Bay ⛰️": (48.38, -89.25),
    "Sault Ste. Marie 🌊": (46.49, -84.35)
}

if "Custom" in city:
    st.sidebar.info("📝 Type your location:")
    lat = st.sidebar.number_input("Latitude (North-South)", value=43.65, format="%.4f")
    lon = st.sidebar.number_input("Longitude (East-West)", value=-79.38, format="%.4f")
    city_name = "Your Location"
else:
    lat, lon = city_coords[city]
    city_name = city.split()[0]
    st.sidebar.success(f"✅ Checking: {city_name}")

st.sidebar.markdown("---")

# Refresh button
if st.sidebar.button("🔄 Check Again!", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# How it works section
with st.sidebar.expander("🤔 How does this work?", expanded=False):
    st.markdown("""
    **FireApp checks 4 things:**
    
    1. 🛰️ **Space Satellites** - Looking from space
    2. 🔥 **Fire Reports** - Official fire news
    3. 🌡️ **Weather** - Temperature & wind
    4. 📡 **Ground Sensors** - Smoke detectors
    
    **If 2 or more say "FIRE!" → We alert you! 🚨**
    """)

# Main detection function
@st.cache_data(ttl=60, show_spinner=False)
def check_for_fire(lat, lon, location):
    """Check if there's a fire - simple version"""
    
    result = {
        'time': datetime.now().strftime('%I:%M %p'),
        'location': location,
        'fire_found': False,
        'checks_passed': 0,
        'total_checks': 4,
        'clues': [],
        'data': {}
    }
    
    # Get data
    satellites = fetch_nasa_firms_data()
    official = fetch_cwfis_data()
    weather = fetch_weather_data(lat, lon)
    sensors = fetch_iot_sensor_data(location=location)
    plants = fetch_vegetation_data(lat, lon)
    
    result['data'] = {
        'satellites': satellites,
        'official': official,
        'weather': weather,
        'sensors': sensors,
        'plants': plants
    }
    
    # Check 1: Satellites
    if satellites is not None and len(satellites) > 0:
        result['checks_passed'] += 1
        result['clues'].append(f"🛰️ Satellites saw {len(satellites)} hot spots!")
    
    # Check 2: Official reports
    if official is not None and len(official) > 0:
        result['checks_passed'] += 1
        result['clues'].append(f"🔥 {len(official)} fires officially reported!")
    
    # Check 3: Weather
    if weather and 'current' in weather:
        temp = weather['current']['temperature_2m']
        humidity = weather['current']['relative_humidity_2m']
        result['temp'] = temp
        result['humidity'] = humidity
        
        if temp > 30 and humidity < 30:
            result['checks_passed'] += 1
            result['clues'].append(f"🌡️ Very hot & dry! ({temp}°C, {humidity}% wet)")
    
    # Check 4: Sensors
    risk = analyze_iot_risk(sensors)
    result['sensor_risk'] = risk
    if risk in ['HIGH', 'MEDIUM'] or sensors.get('flame_detected'):
        result['checks_passed'] += 1
        result['clues'].append(f"📡 Sensors smell smoke! (Risk: {risk})")
    
    # Vegetation
    result['plant_risk'] = get_vegetation_fire_risk(plants)
    
    # Decision
    result['fire_found'] = result['checks_passed'] >= 2
    
    return result

# Run detection with progress
progress_text = st.empty()
progress_bar = st.progress(0)

steps = [
    "🛰️ Asking satellites...",
    "🔥 Checking fire reports...",
    "🌡️ Reading temperature...",
    "📡 Testing sensors...",
    "✅ Making decision..."
]

for i, step in enumerate(steps):
    progress_text.write(step)
    progress_bar.progress((i + 1) * 20)
    time.sleep(0.3)

result = check_for_fire(lat, lon, city_name)

progress_text.empty()
progress_bar.empty()

# Show time
st.caption(f"⏰ Checked at: {result['time']}")
st.markdown("")

# BIG RESULT
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if result['fire_found']:
        st.markdown(f"""
        <div class="fire-alert">
            <h1 style="color:white; margin:0; font-size:3rem;">🔥🚨 FIRE ALERT! 🚨🔥</h1>
            <h2 style="color:white; margin:1rem 0;">
                Found {result['checks_passed']} out of {result['total_checks']} warning signs!
            </h2>
            <p style="color:white; font-size:1.5rem; margin:0;">
                📍 Location: {result['location']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe-alert">
            <h1 style="color:white; margin:0; font-size:3rem;">✅ ALL CLEAR! ✅</h1>
            <h2 style="color:white; margin:1rem 0;">
                Only {result['checks_passed']} out of {result['total_checks']} warnings
            </h2>
            <p style="color:white; font-size:1.5rem; margin:0;">
                📍 {result['location']} is safe right now!
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Visual score
st.subheader("📊 Fire Detection Score")

# Create gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=result['checks_passed'],
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Warning Signs Found", 'font': {'size': 24}},
    delta={'reference': 2, 'increasing': {'color': "red"}},
    gauge={
        'axis': {'range': [None, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "red" if result['checks_passed'] >= 2 else "green"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 1], 'color': '#90EE90'},
            {'range': [1, 2], 'color': '#FFD700'},
            {'range': [2, 4], 'color': '#FF6B6B'}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 2
        }
    }
))

fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Simple metrics
st.subheader("🔍 What We Found")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sat_count = len(result['data']['satellites']) if result['data']['satellites'] is not None else 0
    st.markdown(f"""
    <div class="metric-box">
        <h2 style="margin:0;">🛰️</h2>
        <h3 style="margin:0.5rem 0;">{sat_count}</h3>
        <p style="margin:0; color:#666;">Hot Spots from Space</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    fire_count = len(result['data']['official']) if result['data']['official'] is not None else 0
    st.markdown(f"""
    <div class="metric-box">
        <h2 style="margin:0;">🔥</h2>
        <h3 style="margin:0.5rem 0;">{fire_count}</h3>
        <p style="margin:0; color:#666;">Official Fire Reports</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    temp = result.get('temp', 'N/A')
    temp_display = f"{temp}°C" if isinstance(temp, (int, float)) else temp
    st.markdown(f"""
    <div class="metric-box">
        <h2 style="margin:0;">🌡️</h2>
        <h3 style="margin:0.5rem 0;">{temp_display}</h3>
        <p style="margin:0; color:#666;">Temperature Now</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    sensor_risk = result.get('sensor_risk', 'UNKNOWN')
    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
    st.markdown(f"""
    <div class="metric-box">
        <h2 style="margin:0;">📡</h2>
        <h3 style="margin:0.5rem 0;">{risk_emoji.get(sensor_risk, '⚪')}</h3>
        <p style="margin:0; color:#666;">Sensor Alert: {sensor_risk}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Clues found
st.subheader("🔎 Detective Clues")

if result['clues']:
    for i, clue in enumerate(result['clues'], 1):
        st.success(f"**Clue #{i}:** {clue}")
else:
    st.info("😊 No fire clues found - everything looks normal!")

# Plant danger level
plant_risk = result.get('plant_risk', 'UNKNOWN')
risk_colors = {
    'VERY_HIGH': '🔴',
    'HIGH': '🟠',
    'MEDIUM': '🟡',
    'LOW': '🟢',
    'UNKNOWN': '⚪'
}
st.info(f"🌲 **Plant Fire Danger:** {risk_colors.get(plant_risk, '⚪')} {plant_risk}")

st.markdown("---")

# How to stay safe
with st.expander("🛡️ How to Stay Safe from Fires", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ DO These Things:
        - 🚨 Listen to fire alerts
        - 🏃 Have an escape plan
        - 📱 Keep phone charged
        - 💧 Know where water is
        - 👨‍👩‍👧 Stay with family
        """)
    
    with col2:
        st.markdown("""
        ### ❌ DON'T Do These:
        - 🔥 Don't play with fire
        - 🌲 Don't go near fires
        - 🚗 Don't block roads
        - 📵 Don't ignore warnings
        - 🏃‍♂️ Don't panic - stay calm!
        """)

# Fun facts
with st.expander("🎓 Cool Fire Facts!", expanded=False):
    st.markdown("""
    - 🌍 Satellites orbit Earth 16 times per day looking for fires!
    - 🔥 A small fire can become huge in just 30 seconds
    - 💨 Wind makes fires spread 3 times faster
    - 🌲 Dry trees burn easier than wet ones
    - 🚁 Firefighting planes can carry 3,000 liters of water!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem; background:#f8f9fa; border-radius:10px;">
    <h3>🔥 FireApp - Making Fire Safety Easy!</h3>
    <p>Built with ❤️ to keep Ontario safe</p>
</div>
""", unsafe_allow_html=True)
    