
import os
import pandas as pd

INPUT_FILE = "data/processed_transactions.csv"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("DAY 4: FEATURE ENGINEERING AND FRAUD PATTERN ANALYSIS")
print("=" * 65)

print("\nLoading processed transaction data...")
df = pd.read_csv(INPUT_FILE)

# Feature 1: Difference between sender balance and transaction amount
df["origin_balance_difference"] = (
    df["oldbalanceOrg"] - df["amount"]
)

# Feature 2: Absolute difference, regardless of direction
df["absolute_balance_difference"] = (
    df["origin_balance_difference"].abs()
)

# Feature 3: Relative difference compared with the transaction amount
df["relative_balance_difference"] = (
    df["absolute_balance_difference"]
    / (df["amount"] + 1)
)

print("\n1. NEW FEATURE SUMMARY")
print("-" * 65)

print(
    df[
        [
            "origin_balance_difference",
            "absolute_balance_difference",
            "relative_balance_difference",
        ]
    ].describe().round(4)
)

# Group transactions by the absolute balance difference
df["balance_difference_band"] = pd.cut(
    df["absolute_balance_difference"],
    bins=[-1, 0, 100, 1000, 10000, 100000, float("inf")],
    labels=[
        "0",
        "1-100",
        "101-1000",
        "1001-10000",
        "10001-100000",
        "Above 100000",
    ],
)

print("\n2. FRAUD RATE BY ABSOLUTE BALANCE DIFFERENCE")
print("-" * 65)

difference_summary = df.groupby(
    "balance_difference_band",
    observed=False
)["isFraud"].agg(
    transactions="count",
    fraud_cases="sum",
    fraud_rate="mean",
)

difference_summary["fraud_rate_percent"] = (
    difference_summary["fraud_rate"] * 100
)

print(difference_summary.round(4).to_string())

difference_summary.to_csv(
    os.path.join(OUTPUT_DIR, "day4_balance_difference_analysis.csv")
)

print("\n3. FRAUD RATE BY TRANSACTION TYPE")
print("-" * 65)

type_summary = df.groupby("type").agg(
    transactions=("isFraud", "count"),
    fraud_cases=("isFraud", "sum"),
    average_amount=("amount", "mean"),
    average_origin_balance=("oldbalanceOrg", "mean"),
)

type_summary["fraud_rate_percent"] = (
    type_summary["fraud_cases"]
    / type_summary["transactions"]
    * 100
)

print(type_summary.round(4).to_string())

type_summary.to_csv(
    os.path.join(OUTPUT_DIR, "day4_fraud_by_type_analysis.csv")
)

print("\n4. FRAUD VS LEGITIMATE FEATURE COMPARISON")
print("-" * 65)

feature_comparison = df.groupby("isFraud").agg(
    transaction_count=("amount", "count"),
    average_amount=("amount", "mean"),
    median_amount=("amount", "median"),
    average_origin_balance=("oldbalanceOrg", "mean"),
    average_absolute_balance_difference=(
        "absolute_balance_difference", "mean"
    ),
)

print(feature_comparison.round(4).to_string())

feature_comparison.to_csv(
    os.path.join(OUTPUT_DIR, "day4_feature_comparison.csv")
)

print("\nReports saved:")
print("1. reports/day4_balance_difference_analysis.csv")
print("2. reports/day4_fraud_by_type_analysis.csv")
print("3. reports/day4_feature_comparison.csv")

print("\nDay 4 feature analysis completed successfully!")
