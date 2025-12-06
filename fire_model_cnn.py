import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/wildfire_synthetic.csv")

    X = df.drop("target_fire", axis=1).values
    y = df["target_fire"].values

    model = build_model(X.shape[1])
    model.fit(X, y, epochs=10, batch_size=4)

    print("Model Training Complete")
