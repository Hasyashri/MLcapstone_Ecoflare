import pandas as pd
import numpy as np

def add_engineered_features(df):
    df["temp_wind_interaction"] = df["temperature"] * df["wind_speed"]
    df["dryness_factor"] = df["temperature"] / (df["humidity"] + 1)
    df["fuel_risk"] = df["vegetation_index"] * df["heat_signature"]

    print("Feature Engineering Complete")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/wildfire_synthetic.csv")
    df = add_engineered_features(df)
    df.to_csv("data/wildfire_synthetic_engineered.csv", index=False)
