from flask import Flask, render_template
import os
import pandas as pd
from eda import generate_summary, plot_line_chart, plot_histogram, plot_heatmap

app = Flask(__name__)

@app.route("/")
def index():
    # Load or generate dataset
    if os.path.exists("sensor_timeseries.csv"):
        df = pd.read_csv("sensor_timeseries.csv", parse_dates=["timestamp"])
    else:
        import numpy as np
        dates = pd.date_range(start="2025-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "timestamp": dates,
            "temperature": np.random.randint(15, 35, size=10),
            "humidity": np.random.randint(40, 80, size=10)
        })
        df.to_csv("sensor_timeseries.csv", index=False)

    # Summary + charts
    summary = generate_summary(df)
    plot_line_chart(df)
    plot_histogram(df)
    plot_heatmap(df)

    # Pass dataframe head (table), summary dict
    return render_template(
        "index.html",
        tables=df.head().to_html(classes="table table-bordered table-striped", index=False),
        summary=summary
    )

if __name__ == "__main__":
    app.run(debug=True)
