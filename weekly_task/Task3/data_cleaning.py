import pandas as pd
import os

def create_sample_csv(file_path: str):
    """Creates a sample messy_data.csv if it doesn't exist"""
    if not os.path.exists(file_path):
        data = """Name, Age , Salary, Department
John ,  25, 50000 , IT
Mary,thirty, 60000,HR
, 40, 70000,Finance
Steve, 29,, IT
Anna, 35, 80000,HR"""
        with open(file_path, "w") as f:
            f.write(data)

def clean_data(file_path: str):
    # Auto-create sample CSV if not present
    create_sample_csv(file_path)

    # Load data
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Strip spaces from string fields
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Department"] = df["Department"].astype(str).str.strip()

    # Convert Age to numeric (errors='coerce' converts invalids to NaN)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Convert Salary to numeric, fill missing with mean
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["Salary"].fillna(df["Salary"].mean(), inplace=True)

    # Replace missing names
    df["Name"].replace("nan", "Unknown", inplace=True)

    return df
