
import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    print("Machine Learning Model Training")

    if not os.path.exists("data.csv"):
        print("Error: data.csv not found")
        sys.exit(1)

    df = pd.read_csv("data.csv")
    print(f"Dataset shape: {df.shape}")

    X = df[["study_hours"]].values
    y = df["exam_score"].values

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Equation: Exam Score = {slope:.2f} x Study Hours + {intercept:.2f}")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R2 Score: {r2_score(y, predictions):.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y, predictions):.4f}")

    test_hours = np.array([[3.0], [5.0], [7.5], [10.0]])
    test_predictions = model.predict(test_hours)

    for hours, score in zip(test_hours.ravel(), test_predictions):
        print(f"{hours} study hours -> Predicted score: {score:.1f}")

    os.makedirs("output", exist_ok=True)
    joblib.dump(model, "output/model.joblib")

    print("Model saved successfully!")


if __name__ == "__main__":
    main()
