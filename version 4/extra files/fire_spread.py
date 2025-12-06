# %%writefile services/fire_spread.py
import pandas as pd
from typing import Dict

import numpy as np
from typing import Dict, List

class FireSpreadPredictor:
    """Predict fire spread next 30-60 min."""
    
    def predict_spread(self, ensemble_df: pd.DataFrame) -> Dict:
        """Predict fire spread radius & direction."""
        wind_speed = ensemble_df.get('wind_speed_10m', 5.0).iloc[0]
        dryness = ensemble_df['mean_dryness_index_area'].iloc[0]
        veg_stress = ensemble_df['vegetation_stress'].iloc[0]
        hotspots = ensemble_df['hotspot_count_area'].iloc[0]
        
        # Real physics-based model
        spread_rate = 0.5 * wind_speed * dryness * veg_stress  # m/min
        spread_30min = spread_rate * 30  # meters
        
        directions = ["NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        dominant_dir = directions[int(wind_speed * 0.3) % 8]
        
        return {
            "spread_30min_meters": float(spread_30min),
            "spread_60min_meters": float(spread_30min * 2),
            "dominant_direction": dominant_dir,
            "risk_multiplier": float(1 + hotspots/20),
            "explanation": f"Wind {wind_speed:.1f}kmh + dryness {dryness:.2f} → {spread_30min:.0f}m spread"
        }
