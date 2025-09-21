from flask import Flask, render_template_string
from data_cleaning import clean_data
import os

app = Flask(__name__)

# HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Task 3 - Data Cleaning</title>
</head>
<body style="font-family: Arial; margin: 30px;">
    <h2>Cleaned Data Table</h2>
    {{ table|safe }}
</body>
</html>
"""

# Absolute path to CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "messy_data.csv")

@app.route("/")
def index():
    df = clean_data(FILE_PATH)
    return render_template_string(html_template, table=df.to_html(index=False))

if __name__ == "__main__":
    app.run(debug=True)
