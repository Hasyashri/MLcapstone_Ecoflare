"""
Wildfire Detection System - Streamlit Dashboard
Save as: wildfire_app.py
Run with: streamlit run wildfire_app.py
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# ============================================
# PAGE CONFIG - MUST BE FIRST
# ============================================
st.set_page_config(
    page_title="Wildfire Detection System",
    page_icon="🔥",
    layout="wide"
)

# ============================================
# SAMPLE DATA
# ============================================
fire_data = [
    {
        'id': 1, 'name': 'Site Alpha', 'lat': 54.2, 'lon': -115.8,
        'fire_detected': True, 'detection_prob': 89, 'cause': 'Lightning',
        'spread_prob': 76, 'spread_ha': 234, 'temp': 32, 'humidity': 12,
        'wind_speed': 28, 'status': 'Critical'
    },
    {
        'id': 2, 'name': 'Site Bravo', 'lat': 53.5, 'lon': -113.2,
        'fire_detected': True, 'detection_prob': 72, 'cause': 'Human',
        'spread_prob': 58, 'spread_ha': 145, 'temp': 28, 'humidity': 18,
        'wind_speed': 22, 'status': 'High'
    },
    {
        'id': 3, 'name': 'Site Charlie', 'lat': 52.8, 'lon': -116.5,
        'fire_detected': True, 'detection_prob': 94, 'cause': 'Lightning',
        'spread_prob': 88, 'spread_ha': 412, 'temp': 35, 'humidity': 8,
        'wind_speed': 35, 'status': 'Critical'
    },
    {
        'id': 4, 'name': 'Site Delta', 'lat': 51.9, 'lon': -114.1,
        'fire_detected': False, 'detection_prob': 38, 'cause': 'Unknown',
        'spread_prob': 22, 'spread_ha': 45, 'temp': 24, 'humidity': 32,
        'wind_speed': 15, 'status': 'Moderate'
    },
    {
        'id': 5, 'name': 'Site Echo', 'lat': 53.1, 'lon': -117.2,
        'fire_detected': False, 'detection_prob': 15, 'cause': 'Unknown',
        'spread_prob': 8, 'spread_ha': 12, 'temp': 20, 'humidity': 45,
        'wind_speed': 10, 'status': 'Low'
    },
    {
        'id': 6, 'name': 'Site Foxtrot', 'lat': 52.3, 'lon': -112.8,
        'fire_detected': True, 'detection_prob': 81, 'cause': 'Human',
        'spread_prob': 67, 'spread_ha': 189, 'temp': 30, 'humidity': 15,
        'wind_speed': 25, 'status': 'High'
    }
]

df = pd.DataFrame(fire_data)

# ============================================
# HEADER
# ============================================
st.markdown("""
    <div style='background: linear-gradient(90deg, #38bdf8, #3b82f6); 
                padding: 2rem; 
                border-radius: 10px; 
                color: white;
                margin-bottom: 2rem;'>
        <h1 style='margin:0; color: white;'>🔥 Wildfire Detection System</h1>
        <p style='margin:0; font-size: 1.1rem;'>Real-time ML-Powered Monitoring | December 5, 2025 - 2:09 PM</p>
    </div>
""", unsafe_allow_html=True)

# ============================================
# ALERT BANNER
# ============================================
critical_count = len(df[df['status'] == 'Critical'])
if critical_count > 0:
    st.error(f"⚠️ CRITICAL ALERT: {critical_count} Active Fire(s) Detected - Immediate Action Required")

# ============================================
# STATISTICS
# ============================================
st.subheader("📊 Overview Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sites Monitored", len(df))
    
with col2:
    fires_detected = len(df[df['fire_detected'] == True])
    st.metric("Fires Detected", fires_detected, delta="Active", delta_color="inverse")
    
with col3:
    avg_spread = df[df['fire_detected'] == True]['spread_prob'].mean()
    st.metric("Avg Spread Risk", f"{avg_spread:.0f}%")
    
with col4:
    total_area = df[df['fire_detected'] == True]['spread_ha'].sum()
    st.metric("Est. Total Area", f"{total_area:.0f} ha")

st.markdown("---")

# ============================================
# MAIN LAYOUT
# ============================================
col_map, col_details = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ Geographic Fire Detection Map")
    
    # Create Folium map
    fire_map = folium.Map(
        location=[53.0, -115.0],
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Color mapping
    color_map = {
        'Critical': 'red',
        'High': 'orange',
        'Moderate': 'lightblue',
        'Low': 'green'
    }
    
    # Add markers
    for idx, row in df.iterrows():
        # Popup content
        popup_html = f"""
        <div style='width: 200px; font-family: Arial;'>
            <h3 style='margin: 0 0 10px 0;'>{row['name']}</h3>
            <p><b>Detection:</b> {row['detection_prob']}%</p>
            <p><b>Cause:</b> {row['cause']}</p>
            <p><b>Spread Risk:</b> {row['spread_prob']}%</p>
            <p><b>Area:</b> {row['spread_ha']} ha</p>
            <p style='background: {"#dc2626" if row["fire_detected"] else "#10b981"}; 
                      color: white; 
                      padding: 5px; 
                      border-radius: 5px; 
                      text-align: center;
                      font-weight: bold;'>
                {'ACTIVE FIRE' if row['fire_detected'] else 'MONITORING'}
            </p>
        </div>
        """
        
        # Create marker
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['name']} - {row['status']}",
            icon=folium.Icon(
                color=color_map.get(row['status'], 'gray'),
                icon='fire' if row['fire_detected'] else 'info-sign',
                prefix='glyphicon'
            )
        ).add_to(fire_map)
    
    # Display map
    map_data = st_folium(fire_map, width=700, height=500)

with col_details:
    st.subheader("📍 Site Selection")
    
    # Site selector
    selected_site = st.selectbox(
        "Choose a site to view details:",
        options=df['name'].tolist(),
        index=0
    )
    
    # Get selected site data
    site = df[df['name'] == selected_site].iloc[0]
    
    # Display site details
    st.markdown(f"### {site['name']}")
    st.caption(f"📍 {site['lat']:.2f}°N, {abs(site['lon']):.2f}°W")
    
    st.markdown("---")
    
    # Fire Detection Box
    st.markdown("#### 🔥 Fire Detection")
    detection_color = "#dc2626" if site['status'] in ['Critical', 'High'] else "#10b981"
    st.markdown(f"""
        <div style='background: #f0f9ff; 
                    border-left: 4px solid {detection_color}; 
                    padding: 15px; 
                    border-radius: 5px;
                    margin-bottom: 15px;'>
            <h2 style='color: {detection_color}; margin: 0;'>{site['detection_prob']}%</h2>
            <p style='margin: 5px 0;'>Status: <b>{'ACTIVE FIRE' if site['fire_detected'] else 'Monitoring'}</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Fire Cause Box
    st.markdown("#### 🔍 Fire Cause")
    cause_icon = "⚡" if site['cause'] == 'Lightning' else "👤" if site['cause'] == 'Human' else "❓"
    st.markdown(f"""
        <div style='background: #fef3c7; 
                    border-left: 4px solid #f59e0b; 
                    padding: 15px; 
                    border-radius: 5px;
                    margin-bottom: 15px;'>
            <h3 style='margin: 0;'>{cause_icon} {site['cause']}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Spread Prediction Box
    st.markdown("#### 📈 Spread Prediction")
    st.markdown(f"""
        <div style='background: #fee2e2; 
                    border-left: 4px solid #ef4444; 
                    padding: 15px; 
                    border-radius: 5px;
                    margin-bottom: 15px;'>
            <h2 style='color: #dc2626; margin: 0;'>{site['spread_prob']}%</h2>
            <p style='margin: 5px 0;'>Est. Area: <b>{site['spread_ha']} hectares</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Weather Conditions
    st.markdown("#### 🌤️ Environmental Conditions")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.metric("🌡️ Temp", f"{site['temp']}°C")
    with w2:
        st.metric("💧 Humidity", f"{site['humidity']}%")
    with w3:
        st.metric("💨 Wind", f"{site['wind_speed']} km/h")

# ============================================
# ACTIVE FIRES LIST
# ============================================
st.markdown("---")
st.subheader("🔥 Active Fire Incidents")

active_fires = df[df['fire_detected'] == True]

if len(active_fires) > 0:
    cols = st.columns(3)
    
    for idx, (_, fire) in enumerate(active_fires.iterrows()):
        with cols[idx % 3]:
            cause_icon = "⚡" if fire['cause'] == 'Lightning' else "👤" if fire['cause'] == 'Human' else "❓"
            border_color = "#dc2626" if fire['status'] == 'Critical' else "#f97316"
            
            st.markdown(f"""
                <div style='background: white; 
                            border: 2px solid {border_color}; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 15px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='margin: 0 0 10px 0;'>{cause_icon} {fire['name']}</h4>
                    <p style='margin: 5px 0;'><b>Detection:</b> <span style='color: #f97316;'>{fire['detection_prob']}%</span></p>
                    <p style='margin: 5px 0;'><b>Spread Risk:</b> <span style='color: #dc2626;'>{fire['spread_prob']}%</span></p>
                    <p style='margin: 5px 0;'><b>Area:</b> {fire['spread_ha']} ha</p>
                    <p style='margin: 10px 0 0 0; 
                              padding: 5px; 
                              background: {border_color}; 
                              color: white; 
                              text-align: center; 
                              border-radius: 5px; 
                              font-weight: bold;'>
                        {fire['status'].upper()}
                    </p>
                </div>
            """, unsafe_allow_html=True)
else:
    st.info("No active fires detected.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("💡 Click on map markers for detailed information | Select sites from dropdown to view analysis")