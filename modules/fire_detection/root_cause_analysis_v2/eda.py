import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.title("🔥 Correlation Heatmap - Fire Occurrence Data")
    plt.show()

def show_summary(df):
    print("📌 Dataset Summary")
    print(df.describe())
    print(df.info())
