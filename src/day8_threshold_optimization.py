
import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

DATA_FILE = "data/processed_transactions.csv"
MODEL_FILE = "models/fraud_detection_model.pkl"
OUTPUT_FILE = "reports/day8_threshold_comparison.csv"

RANDOM_STATE = 42
SAMPLE_SIZE = 500_000

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

print("=" * 65)
print("DAY 8: FRAUD DETECTION THRESHOLD OPTIMIZATION")
print("=" * 65)

# 1. Load dataset
print("\nLoading dataset...")
df = pd.read_csv(DATA_FILE)

print("Original rows:", len(df))

# 2. Create the same representative sample
if len(df) > SAMPLE_SIZE:
    df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["isFraud"],
        random_state=RANDOM_STATE
    )

print("Sample rows:", len(df))
print("Fraud cases:", int(df["isFraud"].sum()))

# 3. Separate features and target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# 4. Create a stratified test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Actual fraud cases in test:", int(y_test.sum()))

# 5. Load the existing model
print("\nLoading saved Random Forest model...")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Saved model not found.")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")

# 6. Generate fraud probabilities once
print("\nGenerating prediction probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

print("Prediction probabilities generated!")

# 7. Calculate threshold-independent metrics
pr_auc = average_precision_score(y_test, probabilities)
roc_auc = roc_auc_score(y_test, probabilities)

print("\nPR-AUC:", round(pr_auc, 6))
print("ROC-AUC:", round(roc_auc, 6))

# 8. Evaluate different thresholds
results = []

for threshold in THRESHOLDS:

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    })

    print("\nThreshold:", threshold)
    print("Precision:", round(precision, 6))
    print("Recall:", round(recall, 6))
    print("F1-score:", round(f1, 6))
    print("False positives:", fp)
    print("False negatives:", fn)

# 9. Save threshold comparison
os.makedirs("reports", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("FINAL THRESHOLD COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nResults saved:", OUTPUT_FILE)
print("\nDay 8 threshold analysis completed!")
