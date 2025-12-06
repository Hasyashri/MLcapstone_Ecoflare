import pandas as pd

def root_cause(df):
    correlation = df.corr()["target_fire"].sort_values(ascending=False)
    print("Root Cause Drivers of Wildfire:\n")
    print(correlation)

if __name__ == "__main__":
    df = pd.read_csv("data/wildfire_synthetic.csv")
    root_cause(df)
