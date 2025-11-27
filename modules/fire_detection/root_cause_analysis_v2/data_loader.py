import pandas as pd

def load_fire_occurrence_data(path):
    """
    Load the fire occurrence dataset for model training.
    """
    df = pd.read_csv(path)
    print("🔥 Fire occurrence dataset loaded:", df.shape)
    return df

def preprocess_data(df):
    """
    Basic preprocessing: handle missing values, encode categories, etc.
    """

    df = df.copy()

    # Fill missing
    df = df.fillna(df.mean(numeric_only=True))

    # Encode categories
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    print("✨ Preprocessing complete:", df.shape)
    return df
