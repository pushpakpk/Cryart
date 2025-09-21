# Task 4 – EDA Dashboard

## Description
This task builds a simple **Flask web dashboard** to perform **Exploratory Data Analysis (EDA)** on a time-series dataset (`sensor_timeseries.csv`).  
The dashboard shows:
- Data preview
- Summary statistics
- Visualizations (Line Chart, Histogram, Heatmap)

If the dataset is missing, a sample file is auto-created.

---

## Files
- `app.py` → Flask app to run the dashboard  
- `eda.py` → EDA functions (summary + plots)  
- `templates/index.html` → Frontend (Bootstrap + Jinja2)  
- `sensor_timeseries.csv` → Dataset (optional, auto-generated)  
- `README.md` → Documentation  

---

## How to Run
1. Install dependencies:
```bash
pip install flask pandas matplotlib seaborn
