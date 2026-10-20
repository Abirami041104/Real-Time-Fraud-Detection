
import pandas as pd

file_path = "data/PS_20174392719_1491204439457_log.csv"

total = {0: 0, 1: 0}
matches = {0: 0, 1: 0}

print("Checking fraud patterns...")

for chunk in pd.read_csv(
    file_path,
    usecols=["amount", "oldbalanceOrg", "isFraud"],
    chunksize=200_000
):
    for label in [0, 1]:
        rows = chunk[chunk["isFraud"] == label]

        total[label] += len(rows)

        matches[label] += (
            rows["amount"] == rows["oldbalanceOrg"]
        ).sum()

print("\nAmount equals origin balance:")

for label, name in [(0, "Legitimate"), (1, "Fraud")]:
    percentage = matches[label] / total[label] * 100

    print(
        f"{name}: {matches[label]:,} / "
        f"{total[label]:,} ({percentage:.4f}%)"
    )
