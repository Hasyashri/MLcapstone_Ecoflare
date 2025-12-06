import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    print("Data Loaded Successfully")
    return df

def preprocess_data(df):
    X = df.drop("target_fire", axis=1)
    y = df["target_fire"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("Preprocessing Complete")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data("data/wildfire_synthetic.csv")
    preprocess_data(df)
