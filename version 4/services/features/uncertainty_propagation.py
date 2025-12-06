# =============================================================
# UNCERTAINTY PROPAGATION - QC/Coverage → Ensemble Uncertainty
# Simple, production-ready uncertainty scores
# =============================================================

import pandas as pd
import numpy as np
from typing import Dict
from services.management.logging import get_logger

logger = get_logger("UncertaintyPropagation")

def compute_source_uncertainty(df: pd.DataFrame, source_type: str = "generic") -> float:
   """Compute uncertainty for ONE source from QC + coverage."""
   if df is None or df.empty:
       return 1.0  # Max uncertainty for missing data

   # QC uncertainty (1 - confidence score)
   # Ensure qc_score is treated as a Series to safely call .mean() and .fillna()
   qc_score_series = df.get("qc_confidence_score", pd.Series([0.9])) # Default to Series with 0.9 if column missing
   if isinstance(qc_score_series, (int, float)):
       qc_score = qc_score_series
   else:
       qc_score = qc_score_series.mean()
   qc_uncert = 1.0 - qc_score

   # Coverage uncertainty (1 - coverage fraction)
   # Ensure coverage is treated as a Series to safely call .mean() and .fillna()
   coverage_series = df.get("coverage_fraction_source", pd.Series([1.0])) # Default to Series with 1.0 if column missing
   if isinstance(coverage_series, (int, float)):
       coverage = coverage_series
   else:
       coverage = coverage_series.fillna(1.0).mean()
   cov_uncert = 1.0 - coverage

   # Weighted combination (QC more important)
   source_uncert = 0.6 * qc_uncert + 0.4 * cov_uncert

   logger.info(f"{source_type} uncertainty: QC={qc_uncert:.2f}, Cov={cov_uncert:.2f}, Total={source_uncert:.3f}")
   return min(1.0, max(0.0, source_uncert))

def propagate_ensemble_uncertainty(features: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.Series:
   """
   Propagate source uncertainties → FINAL ensemble uncertainty.

   Formula: weighted average of source uncertainties
   """
   source_uncerts = {}

   for source_name, df in features.items():
       if df is not None and not df.empty:
           uncert = compute_source_uncertainty(df, source_name)
           weight = weights.get(source_name, 0.25)
           source_uncerts[source_name] = uncert * weight
           logger.debug(f"{source_name}: {uncert:.3f} * {weight:.3f} = {uncert*weight:.3f}")

   if not source_uncerts:
       logger.warning("No valid sources for uncertainty propagation")
       return pd.Series([0.1])  # Default low uncertainty

   ensemble_uncert = np.mean(list(source_uncerts.values()))
   logger.info(f"Ensemble uncertainty: {ensemble_uncert:.3f}")

   # Return Series matching ensemble shape (usually 1 row)
   return pd.Series([ensemble_uncert])

if __name__ == "__main__":
   print("🧮 UNCERTAINTY PROPAGATION TEST")

   # Mock features
   mock_features = {
       "satellite": pd.DataFrame({"qc_confidence_score": [0.8, 0.9], "coverage_fraction_source": [0.95, 0.92]}),
       "iot": pd.DataFrame({"qc_confidence_score": [0.7], "coverage_fraction_source": [0.88]}),
       "weather": pd.DataFrame({"qc_confidence_score": [0.95]})
   }

   mock_weights = {"satellite": 0.3, "iot": 0.25, "weather": 0.25, "terrain": 0.2}

   uncertainty = propagate_ensemble_uncertainty(mock_features, mock_weights)
   print(f"✅ Ensemble uncertainty: {uncertainty.iloc[0]:.3f}")
   print("🎉 UNCERTAINTY PROPAGATION READY!")
