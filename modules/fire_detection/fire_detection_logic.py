# ===============================================
# File: modules/fire_detection/fire_detection_logic.py
# Purpose: Main fire detection logic combining all data sources
# ===============================================

from fetch_live_data import fetch_nasa_firms_data, fetch_cwfis_data, fetch_weather_data
from iot_data import fetch_iot_sensor_data, analyze_iot_risk
from vegetation_data import fetch_vegetation_data, get_vegetation_fire_risk
from datetime import datetime

def detect_fire(lat=43.65, lon=-79.38, location_name="Toronto"):
    """
    Main fire detection function using voting logic.
    Combines satellite, weather, IoT, and vegetation data.
    """
    print("=" * 70)
    print(f"WILDFIRE DETECTION - {location_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Fetch all data sources
    nasa_data = fetch_nasa_firms_data()
    cwfis_data = fetch_cwfis_data()
    weather_data = fetch_weather_data(lat, lon)
    iot_data = fetch_iot_sensor_data(location=location_name)
    veg_data = fetch_vegetation_data(lat, lon)
    
    print()
    print("=" * 70)
    print("ANALYSIS & VOTING")
    print("=" * 70)
    
    # Voting system: count how many sources indicate fire
    fire_votes = 0
    total_votes = 0
    evidence = []
    
    # Vote 1: Satellite hotspots
    total_votes += 1
    if nasa_data is not None and len(nasa_data) > 0:
        fire_votes += 1
        evidence.append(f"Satellite: {len(nasa_data)} hotspots detected")
        print(f"✅ VOTE 1: Satellite shows {len(nasa_data)} hotspots - FIRE")
    else:
        print(f"⭕ VOTE 1: No satellite hotspots - NO FIRE")
    
    # Vote 2: Official fire reports
    total_votes += 1
    if cwfis_data is not None and len(cwfis_data) > 0:
        fire_votes += 1
        evidence.append(f"Official: {len(cwfis_data)} fires reported")
        print(f"✅ VOTE 2: {len(cwfis_data)} official fires - FIRE")
    else:
        print(f"⭕ VOTE 2: No official fire reports - NO FIRE")
    
    # Vote 3: Weather conditions
    total_votes += 1
    if weather_data and 'current' in weather_data:
        temp = weather_data['current']['temperature_2m']
        humidity = weather_data['current']['relative_humidity_2m']
        
        if temp > 30 and humidity < 30:
            fire_votes += 1
            evidence.append(f"Weather: {temp}°C, {humidity}% humidity - High risk")
            print(f"✅ VOTE 3: Extreme weather ({temp}°C, {humidity}%) - FIRE RISK")
        else:
            print(f"⭕ VOTE 3: Weather conditions normal - LOW RISK")
    
    # Vote 4: IoT sensors
    total_votes += 1
    iot_risk = analyze_iot_risk(iot_data)
    if iot_risk in ['HIGH', 'MEDIUM'] or iot_data['flame_detected']:
        fire_votes += 1
        evidence.append(f"IoT: {iot_risk} risk, flame={iot_data['flame_detected']}")
        print(f"✅ VOTE 4: IoT sensors show {iot_risk} risk - FIRE")
    else:
        print(f"⭕ VOTE 4: IoT sensors normal - NO FIRE")
    
    # Vote 5: Vegetation risk
    veg_risk = get_vegetation_fire_risk(veg_data)
    print(f"ℹ️  INFO: Vegetation fire risk is {veg_risk}")
    
    print()
    print("=" * 70)
    print("DETECTION RESULT")
    print("=" * 70)
    
    # Decision: If 2 or more votes say fire, then FIRE DETECTED
    fire_detected = fire_votes >= 2
    
    if fire_detected:
        print(f"🔥 FIRE DETECTED!")
        print(f"   Confidence: {fire_votes}/{total_votes} sources confirm")
        print(f"   Location: {location_name} ({lat}, {lon})")
        print(f"   Vegetation Risk: {veg_risk}")
        print(f"   Evidence:")
        for ev in evidence:
            print(f"     - {ev}")
    else:
        print(f"✅ NO FIRE DETECTED")
        print(f"   Votes: {fire_votes}/{total_votes}")
        print(f"   Location: {location_name}")
        print(f"   Vegetation Risk: {veg_risk}")
    
    print("=" * 70)
    
    return {
        'fire_detected': fire_detected,
        'confidence': f"{fire_votes}/{total_votes}",
        'location': location_name,
        'coordinates': (lat, lon),
        'vegetation_risk': veg_risk,
        'evidence': evidence,
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    result = detect_fire(43.65, -79.38, "Toronto")
    print(f"\n📊 Result: {result}")
