import pandas as pd
import os

def combine_realtime_data():
    folder = "modules/fire_detection/"
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    dfs = [pd.read_csv(os.path.join(folder, f)) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    combined.to_csv("modules/fire_detection/test_data/combined_hotspot_data.csv",
                    index=False)

    print("🔥 Real-time data combined and saved to test_data folder!")

if __name__ == "__main__":
    combine_realtime_data()
