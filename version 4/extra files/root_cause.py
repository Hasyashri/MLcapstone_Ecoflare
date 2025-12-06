# %%writefile services/root_cause.py
import pandas as pd
import numpy as np
from typing import Dict, List

class RootCauseAnalyzer:
    """Root cause analysis for fire origin."""
    
    def __init__(self):
        self.causes = {
            "lightning": 0.45, "campfire": 0.25, "powerline": 0.15,
            "arson": 0.1, "railway": 0.05
        }
    
    def analyze(self, ensemble_df: pd.DataFrame) -> Dict:
        """Analyze fire cause based on features."""
        veg_stress = ensemble_df['vegetation_stress'].iloc[0]
        hotspots = ensemble_df['hotspot_count_area'].iloc[0]
        pm25 = ensemble_df['mean_pm2_5_area'].iloc[0]
        dryness = ensemble_df['mean_dryness_index_area'].iloc[0]
        
        # Real logic
        lightning_prob = 0.3 * veg_stress * dryness  # Dry stressed veg = lightning
        campfire_prob = 0.4 * pm25 / 50  # Smoke = human activity
        powerline_prob = 0.2 * hotspots / 20  # Multiple hotspots = infrastructure
        
        probs = {
            "lightning": lightning_prob,
            "campfire": campfire_prob,
            "powerline": powerline_prob,
            "arson": 0.1,
            "railway": 0.05
        }
        
        top_cause = max(probs, key=probs.get)
        return {
            "most_likely_cause": top_cause,
            "probabilities": probs,
            "confidence": max(probs.values()),
            "explanation": f"High veg stress ({veg_stress:.2f}) + hotspots ({hotspots}) suggests {top_cause}"
        }
