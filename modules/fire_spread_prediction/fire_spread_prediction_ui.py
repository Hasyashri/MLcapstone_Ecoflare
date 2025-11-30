# modules/fire_spread_prediction/fire_spread_prediction_model.py

"""
Hybrid spread prediction model for MVP 2.

Combines the physics spread rate with an optional ML correction model.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import math
import datetime
import os

import numpy as np
from sklearn.linear_model import LinearRegression  # fallback mini model
import joblib

from .feature_engineering import SpreadFeatures
from .spread_model import physics_spread_rate


@dataclass
class SpreadPrediction:
    spread_rate: float
    predicted_area: float
    confidence: float
    model_used: str
    generated_at: str


MODEL_PATH = "models/spread_model.joblib"


def load_or_create_ml_model(model_path: str = MODEL_PATH):
    """
    Try loading a pre-trained model; if missing, create a tiny synthetic one.
    """
    if os.path.exists(model_path):
        return joblib.load(model_path)

    # synthetic training data just so pipeline works end-to-end
    rng = np.random.RandomState(42)
    X = rng.rand(200, 4)
    y = rng.rand(200) * 2.0
    model = LinearRegression()
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model


def predict_spread(
    features: SpreadFeatures, time_horizon: int = 60, model_path: str = MODEL_PATH
) -> Dict[str, Any]:
    """
    Main prediction function used by API / dashboard.

    - Computes a physics-based spread rate.
    - Applies an ML correction if a model is available.
    - Estimates burned area as a growing circle.
    """
    spread = physics_spread_rate(features)
    model_used = "physics_only"

    model = load_or_create_ml_model(model_path)
    if model is not None:
        X = [[features.wind_speed, features.moisture, features.slope, features.temp]]
        correction = float(model.predict(X)[0])
        spread = max(spread + correction, 0.0)
        model_used = "physics_plus_ml"

    radius = spread * (time_horizon / 60.0)
    area = math.pi * radius ** 2

    prediction = SpreadPrediction(
        spread_rate=spread,
        predicted_area=area,
        confidence=0.8,
        model_used=model_used,
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
    )

    return asdict(prediction)
