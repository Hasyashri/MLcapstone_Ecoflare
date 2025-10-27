import numpy as np
from datetime import datetime, timedelta

def map_root_cause_to_spread_factor(root_cause: str) -> float:
    mapping = {
        "lightning": 1.1,
        "human": 1.3,
        "equipment": 1.2,
        "unknown": 1.0,
    }
    return mapping.get(root_cause.lower(), 1.0)

def calculate_base_spread_rate(vegetation_type: str, wind_speed: float, moisture: float) -> float:
    base_rates = {
        'grass': 2.5,
        'shrub': 1.8,
        'mixed_forest': 1.2,
        'boreal_forest': 1.5,
        'deciduous_forest': 0.8,
        'agricultural': 2.0
    }
    base_rate = base_rates.get(vegetation_type, 1.2)
    wind_factor = 1 + (wind_speed / 20) ** 1.5
    moisture_factor = max(0.2, 1 - (moisture / 100))
    return base_rate * wind_factor * moisture_factor

def calculate_fire_area(head_distance, flank_distance):
    area_km2 = np.pi * head_distance * flank_distance
    return round(area_km2 * 100, 2)

def calculate_confidence(hour, wind_speed, moisture, root_cause_factor):
    base_confidence = 0.95
    time_decay = 0.03 * hour
    wind_uncertainty = 0.01 * (wind_speed / 10)
    moisture_uncertainty = 0.01 * (moisture / 50)
    confidence = base_confidence - time_decay - wind_uncertainty - moisture_uncertainty - (root_cause_factor - 1) * 0.05
    return max(0.4, min(0.95, confidence))

def assess_risk_level(spread_rate):
    if spread_rate > 3.0:
        return "EXTREME"
    elif spread_rate > 2.0:
        return "HIGH"
    elif spread_rate > 1.0:
        return "MEDIUM"
    else:
        return "LOW"

def enhanced_predict_fire_spread(fire_location, wind_data, vegetation_data, moisture,
                                 root_cause, fire_timestamp,
                                 terrain_features=None, inhibitors=None,
                                 ml_model=None, physics_weight=0.5, ml_weight=0.5,
                                 time_horizon_hours=12):
    base_rate = calculate_base_spread_rate(vegetation_data['type'], wind_data['speed'], moisture)
    root_cause_factor = map_root_cause_to_spread_factor(root_cause)
    physics_rate = base_rate * root_cause_factor

    features = {}  # Implement feature engineering logic that includes root cause etc.

    ml_rate = ml_model.predict(features) if ml_model else physics_rate

    combined_rate = physics_weight * physics_rate + ml_weight * ml_rate

    predictions = []
    lat, lon = fire_location
    wind_direction = wind_data['direction']

    for hour in range(1, time_horizon_hours + 1):
        spread_distance = combined_rate * hour
        head_distance = spread_distance * 3.0
        flank_distance = spread_distance * 0.5

        predictions.append({
            'hour': hour,
            'timestamp': (datetime.utcnow() + timedelta(hours=hour)).isoformat(),
            'predicted_head_lat': lat + (head_distance / 111) * np.cos(np.radians(wind_direction)),
            'predicted_head_lon': lon + (head_distance / 111) * np.sin(np.radians(wind_direction)),
            'estimated_area_ha': calculate_fire_area(head_distance, flank_distance),
            'confidence': calculate_confidence(hour, wind_data['speed'], moisture, root_cause_factor)
        })

    return {
        'origin': fire_location,
        'spread_rate_kmh': round(combined_rate, 2),
        'dominant_direction_deg': wind_direction,
        'risk_level': assess_risk_level(combined_rate),
        'predictions': predictions
    }
