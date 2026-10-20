
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

DATA_FILE = "data/processed_transactions.csv"
MODEL_FILE = "models/fraud_detection_model.pkl"
REPORT_FILE = "reports/model_metrics.txt"

RANDOM_STATE = 42
SAMPLE_SIZE = 500_000

print("=" * 60)
print("FRAUD DETECTION - MODEL TRAINING")
print("=" * 60)

# 1. Validate the processed dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "Processed data not found. Run prepare_data.py first."
    )

# 2. Read a manageable, representative sample
print("\nLoading dataset...")
df = pd.read_csv(DATA_FILE)

print("Original rows:", f"{len(df):,}")

if df["isFraud"].nunique() != 2:
    raise ValueError("Expected both fraud and legitimate rows.")

# Stratified sampling preserves the fraud proportion
if len(df) > SAMPLE_SIZE:
    df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["isFraud"],
        random_state=RANDOM_STATE
    )

print("Sample rows:", f"{len(df):,}")
print("Fraud cases:", int(df["isFraud"].sum()))
print("Fraud percentage:", round(df["isFraud"].mean() * 100, 4))

# 3. Separate features and target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# 4. Split before training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTraining rows:", f"{len(X_train):,}")
print("Testing rows:", f"{len(X_test):,}")

# 5. Encode the transaction type
categorical_features = ["type"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "transaction_type",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# 6. Build the model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

# 7. Train
print("\nTraining Random Forest...")
model.fit(X_train, y_train)
print("Training completed!")

# 8. Evaluate on untouched test data
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
pr_auc = average_precision_score(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nCLASSIFICATION REPORT")
print(report)

print("CONFUSION MATRIX")
print("Rows = actual; columns = predicted")
print(cm)

print("\nPR-AUC / Average Precision:", round(pr_auc, 6))
print("ROC-AUC:", round(roc_auc, 6))

# 9. Save model and evaluation report
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, MODEL_FILE)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("Random Forest Fraud Detection Model\n\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))
    file.write(f"\n\nAverage Precision: {pr_auc:.6f}")
    file.write(f"\nROC-AUC: {roc_auc:.6f}")

print("\nModel saved:", MODEL_FILE)
print("Metrics saved:", REPORT_FILE)
print("\nMODEL TRAINING PIPELINE COMPLETED!")
