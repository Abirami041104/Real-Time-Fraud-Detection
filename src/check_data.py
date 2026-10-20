
import os
import pandas as pd

file_path = os.path.join(
    "data",
    "PS_20174392719_1491204439457_log.csv"
)

print("Checking dataset path...")

if not os.path.exists(file_path):
    print("ERROR: Dataset not found!")
    print("Expected location:", os.path.abspath(file_path))

else:
    print("Dataset found!")
    print("\nReading first 5 rows...")

    df = pd.read_csv(file_path, nrows=5)

    print(df.to_string(index=False))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nNumber of columns:", len(df.columns))
    print("\nDataset verification successful!")
