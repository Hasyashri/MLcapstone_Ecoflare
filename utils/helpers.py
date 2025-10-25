# ===============================================
# File: utils/helpers.py
# Purpose: Helper functions used across modules
# ===============================================

from datetime import datetime

def log_message(message, level="INFO"):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates (simplified)"""
    # Simplified distance calculation
    import math
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # Rough km

def format_coordinates(lat, lon):
    """Format coordinates nicely"""
    return f"{lat:.4f}°N, {abs(lon):.4f}°W"
