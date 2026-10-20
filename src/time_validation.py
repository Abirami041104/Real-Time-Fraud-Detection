
import pandas as pd
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

FILE = "data/processed_transactions.csv"

print("Loading processed data...")
df = pd.read_csv(FILE)

# Sort transactions chronologically
df = df.sort_values("step").reset_index(drop=True)

# Use earlier 80% for training and later 20% for testing
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]
X_train = train_df[features]
y_train = train_df["isFraud"]

X_test = test_df[features]
y_test = test_df["isFraud"]

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

if y_train.nunique() < 2 or y_test.nunique() < 2:
    print("ERROR: Training or testing data contains only one class.")
    raise SystemExit(1)

preprocessor = ColumnTransformer(
    transformers=[
        ("transaction_type", OneHotEncoder(handle_unknown="ignore"),
         ["type"])
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("PR-AUC:", round(average_precision_score(y_test, probabilities), 4))

print("\nTime-based validation completed.")
