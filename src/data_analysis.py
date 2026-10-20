
import os
import pandas as pd

# Dataset path
file_path = os.path.join(
    "data",
    "PS_20174392719_1491204439457_log.csv"
)

# Create reports folder if missing
os.makedirs("reports", exist_ok=True)

print("=" * 60)
print("REAL-TIME FINANCIAL FRAUD DETECTION")
print("DATASET ANALYSIS")
print("=" * 60)

# Load dataset
print("\nLoading full dataset...")
df = pd.read_csv(file_path)

print("\n1. DATASET INFORMATION")
print("-" * 40)
print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())

# Data types and missing values
print("\n2. DATA QUALITY")
print("-" * 40)
print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Fraud distribution
print("\n3. FRAUD DISTRIBUTION")
print("-" * 40)

fraud_counts = df["isFraud"].value_counts().sort_index()
fraud_percentages = (
    df["isFraud"].value_counts(normalize=True).sort_index() * 100
)

print("Transaction counts:")
print(fraud_counts)

print("\nTransaction percentages:")
print(fraud_percentages.round(4))

# Fraud counts by transaction type
print("\n4. FRAUD BY TRANSACTION TYPE")
print("-" * 40)

fraud_by_type = pd.crosstab(
    df["type"],
    df["isFraud"],
    margins=True
)

print(fraud_by_type)

# Amount statistics
print("\n5. TRANSACTION AMOUNTS")
print("-" * 40)
print(df["amount"].describe())

# Save reports
fraud_counts.to_csv("reports/fraud_distribution.csv")
fraud_by_type.to_csv("reports/fraud_by_type.csv")

summary = {
    "total_transactions": len(df),
    "total_columns": len(df.columns),
    "fraud_transactions": int(df["isFraud"].sum()),
    "legitimate_transactions": int((df["isFraud"] == 0).sum()),
    "missing_values": int(df.isnull().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "fraud_percentage": round(
        df["isFraud"].mean() * 100, 4
    )
}

pd.Series(summary).to_csv("reports/dataset_summary.csv")

print("\n6. REPORTS SAVED")
print("-" * 40)
print("reports/dataset_summary.csv")
print("reports/fraud_distribution.csv")
print("reports/fraud_by_type.csv")

print("\nDataset analysis completed successfully!")
