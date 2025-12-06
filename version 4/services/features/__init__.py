"""
EcoFlare Feature Service - Module 1

This package turns cleaned, real-time data from ingestion
into ML-ready features for fire detection and risk scoring.

Files:
    - feature_builder.py          : build per-source feature tables
    - dynamic_feature_weighting.py: assign weights to each source
    - ensemble_features.py        : combine sources into final feature row
    - uncertainty_propagation.py  : simple uncertainty scores
    - spatial_smoothing.py        : optional smoothing over space
"""
__all__ = [
    "feature_builder",
    "dynamic_feature_weighting",
    "ensemble_features",
    "uncertainty_propagation",
    "spatial_smoothing",
]
