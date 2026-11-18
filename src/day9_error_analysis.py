
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ---------------------------------------
# DAY 9: MODEL ERROR ANALYSIS
# ---------------------------------------

DATA_FILE = "data/processed_transactions.csv"
MODEL_FILE = "models/fraud_detection_model.pkl"

ERROR_FILE = "reports/day9_misclassified_transactions.csv"
SUMMARY_FILE = "reports/day9_error_summary.csv"

RANDOM_STATE = 42
SAMPLE_SIZE = 500_000
THRESHOLD = 0.6

print("=" * 65)
print("DAY 9: FRAUD DETECTION MODEL ERROR ANALYSIS")
print("=" * 65)

# 1. Load dataset
print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Original rows:", len(df))

# 2. Create the same stratified sample
if len(df) > SAMPLE_SIZE:
    df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["isFraud"],
        random_state=RANDOM_STATE
    )

print("Sample rows:", len(df))

# 3. Separate features and target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# 4. Create the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Test rows:", len(X_test))

# 5. Load saved model
print("\nLoading trained model...")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Saved model not found.")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")

# 6. Generate probabilities and predictions
print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= THRESHOLD).astype(int)

# 7. Create detailed prediction results
results = X_test.copy()

results["actual"] = y_test
results["predicted"] = predictions
results["fraud_probability"] = probabilities

# 8. Identify error types
results["error_type"] = "CORRECT"

results.loc[
    (results["actual"] == 0) & (results["predicted"] == 1),
    "error_type"
] = "FALSE_POSITIVE"

results.loc[
    (results["actual"] == 1) & (results["predicted"] == 0),
    "error_type"
] = "FALSE_NEGATIVE"

# 9. Extract misclassified transactions
errors = results[
    results["error_type"] != "CORRECT"
].copy()

# 10. Display confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 65)
print("CONFUSION MATRIX RESULTS")
print("=" * 65)

print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("\nTotal misclassified transactions:", len(errors))

# 11. Analyse errors by transaction type
if len(errors) > 0:

    error_summary = (
        errors.groupby(["error_type", "type"])
        .agg(
            transaction_count=("actual", "count"),
            average_amount=("amount", "mean"),
            average_fraud_probability=("fraud_probability", "mean")
        )
        .reset_index()
    )

else:
    error_summary = pd.DataFrame(
        columns=[
            "error_type",
            "type",
            "transaction_count",
            "average_amount",
            "average_fraud_probability"
        ]
    )

# 12. Save reports
os.makedirs("reports", exist_ok=True)

errors.to_csv(ERROR_FILE, index=False)
error_summary.to_csv(SUMMARY_FILE, index=False)

print("\n" + "=" * 65)
print("ERROR ANALYSIS SUMMARY")
print("=" * 65)

print(error_summary.to_string(index=False))

print("\nMisclassified transactions saved:", ERROR_FILE)
print("Error summary saved:", SUMMARY_FILE)

print("\nDay 9 error analysis completed!")
