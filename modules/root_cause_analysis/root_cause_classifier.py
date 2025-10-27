# root_cause_classifier.py
import joblib
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

class RootCauseClassifier:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = joblib.load(model_path) if model_path else RandomForestClassifier(n_estimators=200)
        self.explainer = None

    def train(self, X_train, y_train):
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X_train, y_train)
        self.model.fit(X_res, y_res)
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        shap_values = self.explainer.shap_values(X)
        return preds, probs, shap_values

    def explain_single(self, shap_values, X, index=0):
        shap.force_plot(self.explainer.expected_value[1], shap_values[1][index], X.iloc[index, :], matplotlib=True)
