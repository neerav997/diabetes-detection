"""
Generates a synthetic Pima-Indians-style diabetes dataset for TESTING the
pipeline end-to-end without internet access.

IMPORTANT: This is fake data for demo/testing purposes only.
For your real project, replace diabetes.csv with the actual
Pima Indians Diabetes Dataset from Kaggle:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Just download diabetes.csv from that link and drop it into this folder,
overwriting the one this script creates. Everything else (train_model.py,
app.py) works with either file since they use the same column names.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 768  # same size as the real Pima dataset

# Simulate two population clusters: diabetic and non-diabetic,
# with feature distributions roughly resembling the real dataset.
outcome = np.random.binomial(1, 0.35, n)

def sample(mean_0, mean_1, std, low=0, high=None):
    vals = np.where(
        outcome == 1,
        np.random.normal(mean_1, std, n),
        np.random.normal(mean_0, std, n),
    )
    vals = np.clip(vals, low, high)
    return vals

df = pd.DataFrame({
    "Pregnancies": sample(2.5, 4.5, 2.5, 0, 15).round().astype(int),
    "Glucose": sample(110, 145, 25, 60, 200).round().astype(int),
    "BloodPressure": sample(68, 74, 12, 40, 120).round().astype(int),
    "SkinThickness": sample(20, 27, 10, 0, 60).round().astype(int),
    "Insulin": sample(70, 120, 60, 0, 500).round().astype(int),
    "BMI": sample(30, 34, 6, 15, 55).round(1),
    "DiabetesPedigreeFunction": sample(0.4, 0.6, 0.25, 0.05, 2.4).round(3),
    "Age": sample(29, 37, 11, 21, 81).round().astype(int),
    "Outcome": outcome,
})

df.to_csv("diabetes.csv", index=False)
print(f"Created diabetes.csv with {len(df)} rows")
print(df.head())
print("\nClass balance:\n", df["Outcome"].value_counts())
