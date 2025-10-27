import streamlit as st
from dashboard.fire_detection_ui import show_fire_detection
from dashboard.root_cause_ui import show_root_cause_analysis
from dashboard.fire_spread_prediction_ui import show_fire_spread_prediction

def main():
    st.title("🔥 EcoFlare Wildfire Management Dashboard")

    # Fire Detection Module
    detection_results = show_fire_detection()

    if detection_results and detection_results.get('fire_detected', False):
        # Root Cause Analysis Module
        root_cause_info = show_root_cause_analysis(detection_results)

        # Combine data for spread prediction
        real_time_data = {
            'fire_location': detection_results['coordinates'],
            'wind_data': detection_results.get('wind_data', {'speed': 10, 'direction': 45}),
            'vegetation_data': detection_results.get('vegetation_data', {'type': 'mixed_forest', 'moisture': 30}),
            'root_cause': root_cause_info.get('root_cause', 'unknown'),
            'detection_time': detection_results.get('timestamp')
        }

        # Fire Spread Prediction Module
        show_fire_spread_prediction(real_time_data)

if __name__ == "__main__":
    main()
