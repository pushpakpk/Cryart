# Task 3 – Data Cleaning and Analysis

## Description
This task demonstrates **data cleaning** of a messy CSV file using **pandas** and displays the cleaned table in a **Flask web app**.  
The cleaning steps include:
- Handling missing values
- Converting incorrect types (e.g., Age, Salary)
- Removing extra spaces
- Replacing missing names with "Unknown"
- Auto-creating a sample CSV if it does not exist

---

## Files
- `app.py` – Flask web application to display cleaned data  
- `data_cleaning.py` – Functions to clean the CSV file  
- `messy_data.csv` – Optional sample CSV file (auto-created if missing)  
- `README.md` – This file  

---

## How to Run
1. Install Python 3.x and required packages:
```bash
pip install pandas flask
