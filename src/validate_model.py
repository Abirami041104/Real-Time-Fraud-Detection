
import pandas as pd
import numpy as np

FILE = "data/processed_transactions.csv"

print("Loading transaction data...")
df = pd.read_csv(FILE)

# Examine the relative amount feature by class
print("\n1. AMOUNT-TO-BALANCE FEATURE")
print(
    df.groupby("isFraud")["amount_to_origin_balance"]
    .describe()
    .round(4)
)

# Examine fraud by transaction type
print("\n2. FRAUD RATE BY TRANSACTION TYPE")
print(
    df.groupby("type")["isFraud"]
    .agg(["count", "sum", "mean"])
    .assign(fraud_percentage=lambda x: x["mean"] * 100)
    .round(4)
)

# Examine the simulated timeline
print("\n3. TRANSACTION TIMELINE")
print("Minimum step:", df["step"].min())
print("Maximum step:", df["step"].max())

print("\nFraud counts by step:")
print(
    df.groupby("step")["isFraud"]
    .agg(["count", "sum"])
    .query("sum > 0")
    .head(10)
)

# Inspect model performance on the saved model
print("\n4. MODEL FEATURE IMPORTANCE")
from joblib import load

model = load("models/fraud_detection_model.pkl")
classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()
importance = classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)

print(importance_df.head(10).to_string(index=False))

print("\nValidation inspection completed.")
