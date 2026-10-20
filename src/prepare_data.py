
import os
import pandas as pd

INPUT_FILE = (
    "data/PS_20174392719_1491204439457_log.csv"
)

OUTPUT_FILE = "data/processed_transactions.csv"

CHUNK_SIZE = 200_000

# Create the output folder if needed
os.makedirs("data", exist_ok=True)

# Features available before the transaction is settled
selected_columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud"
]

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

print("Starting feature engineering...")
print("Reading transactions in chunks.")

total_rows = 0
total_fraud = 0
first_chunk = True

# Process the large CSV without loading it all at once
for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=selected_columns,
    chunksize=CHUNK_SIZE
):
    # Create a useful relative-amount feature
    chunk["amount_to_origin_balance"] = (
        chunk["amount"]
        / (chunk["oldbalanceOrg"] + 1)
    )

    # Check that selected model features are complete
    if chunk.isnull().any().any():
        raise ValueError(
            "Missing values found in the selected data."
        )

    # Write the first chunk, then append subsequent chunks
    chunk.to_csv(
        OUTPUT_FILE,
        index=False,
        mode="w" if first_chunk else "a",
        header=first_chunk
    )

    first_chunk = False
    total_rows += len(chunk)
    total_fraud += int(chunk["isFraud"].sum())

    print(
        f"Processed {total_rows:,} transactions..."
    )

print("\nFeature engineering completed!")
print("Total processed transactions:", f"{total_rows:,}")
print("Fraud transactions:", f"{total_fraud:,}")
print("Output file:", OUTPUT_FILE)

print("\nFeatures created:")
print(selected_columns + ["amount_to_origin_balance"])
