import joblib

class FireSpreadMLModel:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, features: dict) -> float:
        import pandas as pd
        df = pd.DataFrame([features])
        X = self.preprocessor.transform(df)
        return float(self.model.predict(X)[0])
