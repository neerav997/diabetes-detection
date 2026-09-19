"""
Trains a diabetes-prediction model on diabetes.csv and saves:
  - model.pkl   (trained RandomForestClassifier)
  - scaler.pkl  (StandardScaler fit on training data)

Run: python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"

def main():
    df = pd.read_csv("diabetes.csv")
    print(f"Loaded {len(df)} rows")

    # Some entries in the real Pima dataset use 0 as a placeholder for
    # "missing" in columns where 0 isn't physically valid. Replace those
    # 0s with the column median so the model isn't misled.
    zero_invalid_cols = ["Glucose", "BloodPressure", "Insulin", "BMI"]
    for col in zero_invalid_cols:
        if col in df.columns:
            median = df.loc[df[col] != 0, col].median()
            df[col] = df[col].replace(0, median)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    print("\n--- Evaluation on held-out test set ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1 score:  {f1_score(y_test, preds):.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))
    print("\nFull report:\n", classification_report(y_test, preds))

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nSaved model.pkl and scaler.pkl")

if __name__ == "__main__":
    main()
