# test_root_cause_module.py
import pytest
from root_cause_data import RootCauseDataPipeline
from root_cause_classifier import RootCauseClassifier
import pandas as pd

def test_feature_pipeline():
    pipeline = RootCauseDataPipeline("tests/light.geojson", "tests/pop.geojson", "tests/infra.geojson", "", "")
    dummy_data = [{"lat": 40.0, "lon": -120.0, "timestamp": "2025-10-25T12:00:00"}]
    fire_gdf = pipeline.ingest_realtime_fires(dummy_data)
    assert not fire_gdf.empty

def test_classifier_prediction():
    clf = RootCauseClassifier()
    X = pd.DataFrame({"lightning_recent": [1], "infra_density": [0.5], "pop_density": [100], "hour": [12]})
    clf.model.fit(X, [0])
    preds, probs, shap_values = clf.predict(X)
    assert len(preds) == 1
