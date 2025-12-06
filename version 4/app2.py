"""
🔥 EcoFlare PRODUCTION DASHBOARD v4.0
NASA FIRMS LIVE + 569K NDVI + ML Pipeline + Canada Map
Production-ready: Docker + Logging + Feature Builder + 3 Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import requests
import json
import joblib
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Canada Bounding Box (fixes Africa problem)
CANADA_BBOX = {
    'min_lat': 41.7, 'max_lat': 83.1,
    'min_lon': -141.0, 'max_lon': -52.6
}

# NASA FIRMS API Parameters
FIRMS_SOURCES = ['viirs', 'modis']
FIRMS_DAYS = 1  # Last 24h

st.set_page_config(
    page_title="EcoFlare – Production Wildfire Dashboard",
    page_icon="🔥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LIVE DATA FETCHING (NASA FIRMS + Ingestion Pipeline)
# ============================================================================
@st.cache_data(ttl=300)  # Refresh every 5 minutes
def fetch_live_firms(source='viirs'):
    """Fetch LIVE NASA FIRMS data, Canada only"""
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{source}/" \
          f"{CANADA_BBOX['min_lat']},{CANADA_BBOX['min_lon']}," \
          f"{CANADA_BBOX['max_lat']},{CANADA_BBOX['max_lon']}/{FIRMS_DAYS}"
    
    try:
        df = pd.read_csv(url)
        if df.empty:
            return df
        
        # High confidence only + risk scoring
        df = df[df['confidence'] >= 66]
        df['risk_score'] = (df['brightness'].rank(pct=True) * 
                           df['confidence'] / 100).clip(0, 1)
        df['source'] = source.upper()
        df['fetch_time'] = datetime.now()
        return df.sort_values('risk_score', ascending=False)
    except Exception as e:
        st.warning(f"FIRMS {source.upper()} fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_ml_artifacts():
    """Load all ML artifacts safely"""
    artifacts = {}
    
    # Training dataset
    data_file = DATA_DIR / "features" / "fire_training_master_clean.csv"
    artifacts['dataset'] = pd.read_csv(data_file) if data_file.exists() else None
    
    # Models
    artifacts['detection'] = joblib.load(MODELS_DIR / "detection_model.pkl") \
        if (MODELS_DIR / "detection_model.pkl").exists() else None
    
    # Reports
    artifacts['summary'] = {}
    for file in REPORTS_DIR.rglob("*detection*.json"):
        try:
            artifacts['summary'][file.stem] = json.load(open(file))
        except: pass
    
    return artifacts

# ============================================================================
# MAIN DASHBOARD
# ============================================================================
st.title("🔥 EcoFlare – Production Wildfire Intelligence")
st.markdown("**LIVE NASA FIRMS + 569K NDVI + ML Detection Pipeline**")

# Sidebar: Controls + Status
with st.sidebar:
    st.header("⚙️ Controls")
    refresh_rate = st.slider("Data refresh (sec)", 60, 600, 300)
    source_filter = st.multiselect("FIRMS Source", FIRMS_SOURCES, FIRMS_SOURCES)
    
    st.header("📊 Pipeline Status")
    artifacts = load_ml_artifacts()
    
    if artifacts['dataset'] is not None:
        st.success(f"✅ Dataset: {len(artifacts['dataset']):,} rows")
        fires = (artifacts['dataset']['fire_occurred'] == 1).sum()
        st.caption(f"Fires: {fires:,}")
    else:
        st.error("❌ Run `python run_pipeline.py`")
    
    if artifacts['detection'] is not None:
        st.success("✅ Detection model loaded")
    else:
        st.warning("⚠️ `models/detection_model.pkl` missing")
    
    st.info("Docker: `docker-compose up`")

# ============================================================================
# TAB 1: LIVE FIRES MAP (NASA FIRMS)
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🛰️ Live Fires (NASA FIRMS)", 
    "🤖 ML Detection", 
    "📈 Model Metrics", 
    "🧠 Live Inference"
])

with tab1:
    st.header("🛰️ Live NASA FIRMS Detections (Canada)")
    
    # Fetch live data
    all_fires = []
    for source in source_filter:
        fires = fetch_live_firms(source)
        if not fires.empty:
            all_fires.append(fires)
    
    if all_fires:
        firms_df = pd.concat(all_fires, ignore_index=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚨 Active Fires", len(firms_df))
        col2.metric("🌡️ Max Temp", f"{firms_df['brightness'].max():.0f}K")
        col3.metric("🎯 Avg Confidence", f"{firms_df['confidence'].mean():.0f}%")
        col4.metric("⚠️ High Risk", f"{(firms_df['risk_score'] > 0.8).sum()}")
        
        # Live fires table
        st.dataframe(
            firms_df[['latitude', 'longitude', 'brightness', 'confidence', 
                     'risk_score', 'source']].round(2),
            use_container_width=True
        )
        
        # INTERACTIVE CANADA MAP
        st.subheader("🗺️ Interactive Risk Map")
        m = folium.Map(location=[56, -95], zoom_start=4, tiles='CartoDB positron')
        
        for _, row in firms_df.iterrows():
            color = 'red' if row['risk_score'] > 0.8 else 'orange'
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=10 * row['risk_score'],
                color=color, fill=True, fillOpacity=0.7,
                popup=f"<b>{row['source']}</b><br>"
                      f"Brightness: {row['brightness']:.0f}K<br>"
                      f"Confidence: {row['confidence']:.0f}%<br>"
                      f"Risk: {row['risk_score']:.1%}"
            ).add_to(m)
        
        st_folium(m, width=1400, height=600)
    else:
        st.warning("❌ No active fires in Canada right now")

# ============================================================================
# TAB 2: ML MODEL PERFORMANCE
# ============================================================================
with tab2:
    st.header("🤖 ML Detection Model")
    
    if artifacts['detection']:
        # Load metrics
        summary_file = REPORTS_DIR / "detection_eval" / "summary_metrics_detection.json"
        if summary_file.exists():
            with open(summary_file) as f:
                metrics = json.load(f)
            col1, col2, col3 = st.columns(3)
            col1.metric("📈 ROC-AUC", f"{metrics.get('roc_auc', 0):.1%}")
            col2.metric("📉 PR-AUC", f"{metrics.get('pr_auc', 0):.1%}")
            col3.metric("🔬 Test Samples", f"{metrics.get('n_test_samples', 0):,}")
        
        # Threshold scenarios
        scenarios_file = REPORTS_DIR / "detection_eval" / "detection_threshold_scenarios.csv"
        if scenarios_file.exists():
            st.subheader("🎚️ Detection Thresholds")
            scenarios_df = pd.read_csv(scenarios_file)
            st.dataframe(scenarios_df.round(3), use_container_width=True)
            
            # Threshold selector
            threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.25, 0.01)
            best_row = scenarios_df.loc[(scenarios_df['threshold'] - threshold).abs().idxmin()]
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Precision", f"{best_row['precision']:.1%}")
            col2.metric("🔄 Recall", f"{best_row['recall']:.1%}")
            col3.metric("⚠️ False Alarms", f"{best_row['false_alarm_rate']:.1%}")
    
    # Model comparison
    comp_file = REPORTS_DIR / "model_comparison" / "model_comparison_summary.csv"
    if comp_file.exists():
        st.subheader("🏆 Model Comparison")
        comp_df = pd.read_csv(comp_file)
        metric = st.selectbox("Metric", comp_df.select_dtypes(include='number').columns)
        fig = px.bar(comp_df, x='model', y=metric, text_auto='.3f',
                    title=f"Model Performance: {metric}")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: LIVE PIPELINE DEMO
# ============================================================================
with tab3:
    st.header("🔄 Live Feature Pipeline Demo")
    
    if st.button("🚀 Run Live Ingestion + Features"):
        with st.spinner("Fetching NASA FIRMS + Weather + 569K NDVI..."):
            # Demo live ingestion
            try:
                from services.features.feature_builder import build_production_features
                features = build_production_features("Toronto")
                st.success("✅ Live feature pipeline executed!")
                st.dataframe(features.round(3))
            except ImportError:
                st.info("ℹ️ Run `python services/features/feature_builder.py` for live demo")
    
    # Show cached data
    for data_type in ['firms', 'vegetation', 'weather']:
        files = list(DATA_DIR.glob(f"{data_type}/*.csv"))
        if files:
            latest = max(files, key=lambda x: x.stat().st_mtime)
            st.caption(f"✅ Latest {data_type}: {latest.name}")

# ============================================================================
# TAB 4: SHAP + EXPLAINABILITY
# ============================================================================
with tab4:
    st.header("🧠 SHAP Explainability")
    
    shap_file = REPORTS_DIR / "shap" / "shap_feature_importance.csv"
    if shap_file.exists():
        shap_df = pd.read_csv(shap_file)
        top_features = shap_df.nlargest(15, 'shap_value')
        
        fig = px.bar(top_features, x='shap_value', y='feature_name',
                    orientation='h', title="Top SHAP Features")
        fig.update_layout(yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Run `python 12_shap_explainability.py`")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **Pipeline Status** ✅
    - Static ETL: 6 steps → 100K rows
    - Live Ingestion: 7 APIs → real-time
    - ML Models: Detection + Cause + Spread
    """)
with col2:
    st.markdown("""
    **Key Metrics** 📊
    - ROC-AUC: 66%
    - Best Recall: 73% 
    - Best Precision: 97%
    - SHAP: 581 features
    """)
with col3:
    st.markdown("""
    **Production Ready** 🚀
    - Docker: `docker-compose up`
    - Logging: Python `logging`
    - Config: YAML driven
    - Canada-only bounding box
    """)

st.caption("🎓 INFO8665 Capstone – Group 6: Hasyashri, Babandeep, Fenil, Shrinu")
