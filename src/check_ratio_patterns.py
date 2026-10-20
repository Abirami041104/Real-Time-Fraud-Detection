
import pandas as pd

FILE = "data/processed_transactions.csv"

print("Loading data...")
df = pd.read_csv(FILE)

print("\n1. FRAUD RATE BY RATIO RANGE")

df["ratio_band"] = pd.cut(
    df["amount_to_origin_balance"],
    bins=[-1, 0.5, 0.9, 1.0, 1.1, 2.0, float("inf")],
    labels=[
        "0 to 0.5",
        "0.5 to 0.9",
        "0.9 to 1.0",
        "1.0 to 1.1",
        "1.1 to 2.0",
        "Above 2.0"
    ]
)

summary = df.groupby(
    "ratio_band", observed=False
)["isFraud"].agg(
    transactions="count",
    fraud_cases="sum",
    fraud_rate="mean"
)

summary["fraud_rate_percent"] = summary["fraud_rate"] * 100

print(summary.round(4).to_string())

print("\n2. FRAUD AND LEGITIMATE RATIO QUANTILES")

print(
    df.groupby("isFraud")["amount_to_origin_balance"]
    .quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
    .unstack()
    .round(4)
    .to_string()
)

print("\n3. FRAUD RATE BY TYPE AND RATIO")

print(
    df.groupby(["type", "ratio_band"], observed=False)["isFraud"]
    .agg(["count", "sum", "mean"])
    .round(4)
    .to_string()
)

print("\nRatio pattern inspection completed.")
