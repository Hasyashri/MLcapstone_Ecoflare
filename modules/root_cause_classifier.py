# ===============================================
# File: modules/root_cause_analysis/root_cause_classifier.py
# Purpose: ML ignition cause classifier & explainability
# ===============================================

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

CAUSE_LABELS = ["Lightning", "Human Activity", "Equipment Failure", "Spontaneous Combustion"]

class RootCauseClassifier:
    def __init__(self, model_path="models/root_cause_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            )

    def train(self, X, y):
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        shap_values = None
        try:
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(self.model)
            shap_values = self.explainer.shap_values(X)
        except Exception:
            pass
        results = []
        for i in range(len(X)):
            cause = CAUSE_LABELS[int(preds[i]) % len(CAUSE_LABELS)]
            confidence = np.max(probs[i]) if len(probs[i]) else None
            results.append({"predicted_cause": cause, "confidence": round(confidence, 3)})
        return results, shap_values
