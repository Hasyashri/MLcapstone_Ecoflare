# ==========================================================
# 🔥 Early Fire Detection Visualization (No official reports)
# Integrates seamlessly with existing FireApp layout
# ==========================================================

import pydeck as pdk
import plotly.express as px
import pandas as pd
from scipy.spatial import ConvexHull

# --- Helper: Approximate fire area in hectares ---
def calculate_fire_area(points_df):
    """Estimate fire area using convex hull of NASA points."""
    if points_df is None or len(points_df) < 3:
        return 0.0
    try:
        hull = ConvexHull(points_df[['latitude', 'longitude']].values)
        area = hull.volume * 100  # rough hectare approximation
        return round(area, 2)
    except Exception:
        return 0.0


# --- Helper: Compute combined risk level ---
def compute_risk_level(weather_data, vegetation_risk, sensor_risk):
    """Combine vegetation, weather, and sensor risks."""
    wind = weather_data['current'].get('wind_speed_10m', 0)
    humidity = weather_data['current'].get('relative_humidity_2m', 50)
    temperature = weather_data['current'].get('temperature_2m', 20)

    sensor_score = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3, "UNKNOWN": 0.4}.get(sensor_risk, 0.4)
    risk_score = (
        (0.4 * vegetation_risk) +
        (0.3 * (wind / 30)) +
        (0.2 * (1 - humidity / 100)) +
        (0.1 * sensor_score)
    )

    if risk_score < 0.3: return "🟢 Low"
    elif risk_score < 0.6: return "🟡 Medium"
    elif risk_score < 0.8: return "🟠 High"
    else: return "🔴 Very High"


# --- Main Visualization Function ---
def visualize_fire_dashboard(result):
    """Visualize detected fires using NASA + IoT data."""
    satellites = result['data'].get('satellites', [])
    weather = result['data'].get('weather', {})
    sensors = result['data'].get('sensors', {})
    plants = result['data'].get('plants', {})

    # Convert to DataFrame
    sat_df = pd.DataFrame(satellites) if satellites else pd.DataFrame()

    if sat_df.empty:
        st.info("✅ No active satellite fire detections nearby.")
        return

    # --- Calculate Metrics ---
    total_points = len(sat_df)
    area_estimate = calculate_fire_area(sat_df)
    vegetation_risk = get_vegetation_fire_risk(plants)
    sensor_risk = result.get('sensor_risk', 'LOW')
    risk_level = compute_risk_level(weather, vegetation_risk, sensor_risk)

    # --- Display Metrics ---
    st.markdown("### 🌍 Fire Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-box">🔥<br><b>Fire Area</b><br>{area_estimate} ha</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box">🛰️<br><b>Satellite Detections</b><br>{total_points}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box">⚠️<br><b>Overall Risk</b><br>{risk_level}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Fire Map ---
    st.subheader("🗺️ Live Fire Detection Map")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-v9",
        initial_view_state=pdk.ViewState(
            latitude=sat_df['latitude'].mean(),
            longitude=sat_df['longitude'].mean(),
            zoom=6,
            pitch=45,
        ),
        layers=[
            pdk.Layer(
                "HeatmapLayer",
                data=sat_df,
                get_position='[longitude, latitude]',
                radius=1000,
                intensity=1,
                threshold=0.1,
                opacity=0.7
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=sat_df,
                get_position='[longitude, latitude]',
                get_fill_color='[255, 80, 0, 160]',
                get_radius=700,
                pickable=True
            )
        ],
        tooltip={
            "html": "<b>🔥 Satellite Fire Point</b><br/>Lat: {latitude}<br/>Lon: {longitude}<br/>",
            "style": {"backgroundColor": "black", "color": "white"}
        }
    ))

    # --- Source Pie (for fun visual breakdown) ---
    st.markdown("### 🔍 Detection Source Breakdown")
    fig = px.pie(
        pd.DataFrame({
            "Source": ["Satellite", "IoT Sensors"],
            "Count": [len(sat_df), len(sensors.get("readings", [])) if sensors else 0]
        }),
        names="Source",
        values="Count",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="🔥 Active Detection Sources"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Environmental Summary ---
    st.markdown("### 🌤️ Environmental Conditions")
    if "current" in weather:
        w = weather["current"]
        st.write(f"""
        - 🌡️ **Temperature:** {w.get('temperature_2m', 'N/A')}°C  
        - 💨 **Wind Speed:** {w.get('wind_speed_10m', 'N/A')} km/h  
        - 💧 **Humidity:** {w.get('relative_humidity_2m', 'N/A')}%  
        - 🌿 **Vegetation Risk:** {vegetation_risk}
        """)
    else:
        st.warning("⚠️ Weather data not available for this location.")

# ==========================================================
# 🔹 To Use:
# After showing fire alert in your main layout, call:
# visualize_fire_dashboard(result)
# ==========================================================
visualize_fire_dashboard(result)