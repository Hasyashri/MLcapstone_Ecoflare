import numpy as np

def adjust_for_terrain(base_rate, slope_deg, elevation):
    slope_factor = 1 + 0.03 * np.tan(np.radians(slope_deg))
    elevation_factor = 1 + 0.001 * elevation
    return base_rate * slope_factor * elevation_factor
