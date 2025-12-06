import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from preprocess import load_data, preprocess_data
from feature_engineering import add_engineered_features
from root_cause_analysis import root_cause
from fire_model_cnn import build_model
import tensorflow as tf

# ----------------------------------------------------
# Dashboard Page Setup
# ----------------------------------------------------
st.set_page_config(page_title="Ecoflare Wildfire Dashboard", layout="wide")

st.title("🔥 Ecoflare Wildfire Analysis Dashboard")
st.write("Dashboard created by **Srinu** for the ML Capstone Ecoflare Project.")

# ----------------------------------------------------
# Sidebar Navigation Menu
# ----------------------------------------------------
menu = st.sidebar.radio(
    "Choose a section:",
    ["Dataset Overview", "EDA Visuals", "Feature Engineering", "Model Training", "Root Cause Analysis", "Results"]
)

# ----------------------------------------------------
# 1. Dataset Overview
# ----------------------------------------------------
if menu == "Dataset Overview":
    st.header("📂 Dataset Overview")

    df = load_data("data/wildfire_synthetic.csv")
    st.write("### Preview of Dataset:")
    st.dataframe(df)

    st.write("### Dataset Info:")
    st.write(df.describe())

    st.write("### Fire Occurrence Distribution:")
    st.bar_chart(df["target_fire"].value_counts())


# ----------------------------------------------------
# 2. EDA Visuals
# ----------------------------------------------------
elif menu == "EDA Visuals":
    st.header("📊 Exploratory Data Analysis")

    df = load_data("data/wildfire_synthetic.csv")

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

    st.write("### Pairplot (Preview)")
    st.info("Full pairplot is heavy, only shown in notebook.")


# ----------------------------------------------------
# 3. Feature Engineering
# ----------------------------------------------------
elif menu == "Feature Engineering":
    st.header("🧩 Feature Engineering Module")

    df = load_data("data/wildfire_synthetic.csv")
    df_fe = add_engineered_features(df)

    st.write("### Engineered Dataset Preview:")
    st.dataframe(df_fe.head())

    df_fe.to_csv("data/wildfire_synthetic_engineered.csv", index=False)
    st.success("Engineered dataset saved as: wildfire_synthetic_engineered.csv")


# ----------------------------------------------------
# 4. Model Training
# ----------------------------------------------------
elif menu == "Model Training":
    st.header("🤖 Train the Wildfire Prediction Model")

    df = load_data("data/wildfire_synthetic.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = build_model(X_train.shape[1])

    st.info("Training model for 5 epochs…")
    history = model.fit(X_train, y_train, epochs=5, batch_size=4, verbose=0)

    st.success("Model Training Completed!")

    st.write("### Training Accuracy Curve:")
    fig, ax = plt.subplots()
    ax.plot(history.history["accuracy"])
    ax.set_title("Training Accuracy Over Epochs")
    st.pyplot(fig)


# ----------------------------------------------------
# 5. Root Cause Analysis
# ----------------------------------------------------
elif menu == "Root Cause Analysis":
    st.header("🔍 Root Cause Analysis of Wildfire Events")

    df = load_data("data/wildfire_synthetic.csv")

    st.write("### Feature Correlation with Fire Occurrence")
    correlation = df.corr()["target_fire"].sort_values(ascending=False)
    st.dataframe(correlation)

    st.write("### Interpretation")
    st.write("""
    **Key Findings:**
    - 🔥 **Heat Signature** and **Smoke Index** are strong drivers of wildfire.
    - 🌬️ Higher **Wind Speed** also increases fire risk.
    - 💧 **Humidity** reduces likelihood of fire.
    - 🌿 Vegetation contributes moderately depending on dryness.
    """)


# ----------------------------------------------------
# 6. Results Summary
# ----------------------------------------------------
elif menu == "Results":
    st.header("📈 Final Model Results Summary")

    try:
        with open("results/metrics.json") as f:
            metrics = json.load(f)

        st.write("### Model Performance Metrics")
        st.json(metrics)

    except FileNotFoundError:
        st.error("❌ metrics.json not found. Please train the model first.")
