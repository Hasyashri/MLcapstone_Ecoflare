# modules/root_cause_analysis/root_cause_classification.py

"""
MVP 3 – Root Cause Classification with Random Forest + SHAP.

Predicts the most likely cause of fire:
    ["lightning", "human", "equipment"]
and explains it using SHAP values.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import datetime
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

try:
    import shap
except ImportError:
    shap = None  # SHAP is optional; if missing, explanation will be empty.

from .root_cause_data import build_rootcause_features


CAUSE_LABELS: List[str] = ["lightning", "human", "equipment"]
MODEL_PATH = "models/rootcause_rf.joblib"


@dataclass
class RootCausePrediction:
    predicted_cause: str
    probabilities: Dict[str, float]
    shap_values: Dict[str, float]
    confidence: float
    generated_at: str


def load_or_train_default_model(model_path: str = MODEL_PATH):
    """
    Load a saved RandomForest model or train a small synthetic one.
    This keeps the pipeline self-contained for the assignment.
    """
    if os.path.exists(model_path):
        return joblib.load(model_path)

    rng = np.random.RandomState(42)
    X = rng.rand(300, 7)
    y = rng.randint(0, len(CAUSE_LABELS), size=300)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    clf.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    return clf


def predict_root_cause(
    detection_data: Dict[str, Any], model_path: str = MODEL_PATH
) -> Dict[str, Any]:
    """
    High-level service:
        detection_data → engineered features → RF prediction (+ SHAP).
    """
    features = build_rootcause_features(detection_data)
    feature_names = list(features.keys())
    X = np.array([[features[name] for name in feature_names]])

    model = load_or_train_default_model(model_path)

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = CAUSE_LABELS[pred_idx]
    confidence = float(proba[pred_idx])

    prob_dict = {
        CAUSE_LABELS[i]: float(proba[i]) for i in range(len(CAUSE_LABELS))
    }

    shap_values_dict: Dict[str, float] = {}
    if shap is not None:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)[pred_idx][0]
        shap_values_dict = {
            feature_names[i]: float(shap_vals[i]) for i in range(len(feature_names))
        }

    result = RootCausePrediction(
        predicted_cause=pred_label,
        probabilities=prob_dict,
        shap_values=shap_values_dict,
        confidence=confidence,
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
    )
    return asdict(result)


# Manual quick test
if __name__ == "__main__":
    dummy_detection = {
        "vote_count": 3,
        "weather_temp": 32,
        "weather_wind": 25,
        "weather_humidity": 25,
        "near_power_lines": 1,
        "population_density": 40,
        "recent_lightning_strikes": 5,
    }
    from pprint import pprint

    pprint(predict_root_cause(dummy_detection))
