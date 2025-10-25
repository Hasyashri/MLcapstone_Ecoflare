# ===============================================
# File: modules/fire_detection/vegetation_data.py
# Purpose: Fetch vegetation data for fire risk assessment
# ===============================================

def fetch_vegetation_data(lat=43.65, lon=-79.38):
    """
    Get vegetation data using geographic fallback for Ontario.
    Returns fire risk based on region.
    """
    print(f"🌲 Analyzing vegetation for ({lat}, {lon})...")
    
    # Southern Ontario (below 46° latitude)
    if lat < 46:
        if lon > -80:  # Eastern Ontario
            return {
                'source': 'geographic_fallback',
                'has_forest': True,
                'has_wetland': False,
                'has_grassland': True,
                'vegetation_types': ['mixed_forest', 'agricultural', 'urban'],
                'fire_risk_areas': ['forest', 'grassland'],
                'region': 'Southern Ontario (East)'
            }
        else:  # Western/Central Ontario
            return {
                'source': 'geographic_fallback',
                'has_forest': True,
                'has_wetland': True,
                'has_grassland': False,
                'vegetation_types': ['deciduous_forest', 'mixed_forest', 'wetland'],
                'fire_risk_areas': ['forest'],
                'region': 'Southern Ontario (West/Central)'
            }
    else:  # Northern Ontario
        return {
            'source': 'geographic_fallback',
            'has_forest': True,
            'has_wetland': True,
            'has_grassland': False,
            'vegetation_types': ['boreal_forest', 'coniferous_forest', 'wetland'],
            'fire_risk_areas': ['boreal_forest'],
            'region': 'Northern Ontario'
        }

def get_vegetation_fire_risk(vegetation_data):
    """Calculate fire risk from vegetation"""
    if not vegetation_data:
        return 'UNKNOWN'
    
    if vegetation_data['has_forest'] and vegetation_data['has_grassland']:
        return 'VERY_HIGH'
    elif vegetation_data['has_forest'] or vegetation_data['has_grassland']:
        return 'HIGH'
    elif vegetation_data['has_wetland']:
        return 'LOW'
    return 'MEDIUM'

if __name__ == "__main__":
    veg_data = fetch_vegetation_data(43.65, -79.38)
    risk = get_vegetation_fire_risk(veg_data)
    print(f"✅ Vegetation risk: {risk}")
    