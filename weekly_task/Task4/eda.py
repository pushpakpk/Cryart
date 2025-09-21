import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

def generate_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "mean_temperature": round(df["temperature"].mean(), 2),
        "mean_humidity": round(df["humidity"].mean(), 2),
    }

def plot_line_chart(df):
    plt.figure(figsize=(8, 4))
    plt.plot(df["timestamp"], df["temperature"], marker="o", label="Temperature")
    plt.plot(df["timestamp"], df["humidity"], marker="s", label="Humidity")
    plt.title("Line Chart - Temperature & Humidity")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/line_chart.png")
    plt.close()

def plot_histogram(df):
    plt.figure(figsize=(6, 4))
    df[["temperature", "humidity"]].hist(bins=10, figsize=(8, 4))
    plt.tight_layout()
    plt.savefig("static/histogram.png")
    plt.close()

def plot_heatmap(df):
    plt.figure(figsize=(5, 4))
    sns.heatmap(df[["temperature", "humidity"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("static/heatmap.png")
    plt.close()
