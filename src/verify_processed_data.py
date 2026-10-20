
import pandas as pd
import os

file_path = "data/processed_transactions.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Processed dataset not found. "
        "Run prepare_data.py first."
    )

df = pd.read_csv(file_path, nrows=5)

print("Processed dataset verified!")
print("\nSample rows:")
print(df.to_string(index=False))

print("\nColumns:")
print(df.columns.tolist())

print("\nNumber of columns:", len(df.columns))
