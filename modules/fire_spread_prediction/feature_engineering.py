from datetime import datetime

def construct_features(fire_location, wind_data, vegetation_data, moisture,
                       root_cause, fire_timestamp, terrain_features=None, inhibitors=None):
    features = {
        'latitude': fire_location[0],
        'longitude': fire_location[1],
        'wind_speed': wind_data['speed'],
        'wind_direction': wind_data['direction'],
        'vegetation_type': vegetation_data.get('type', 'mixed_forest'),
        'moisture': moisture,
        'root_cause': root_cause.lower(),
        'time_since_fire_started_hours': (datetime.utcnow() - fire_timestamp).total_seconds() / 3600,
    }
    if terrain_features:
        features.update(terrain_features)
    if inhibitors:
        features.update(inhibitors)
    return features
