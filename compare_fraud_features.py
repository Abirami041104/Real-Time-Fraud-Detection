
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

DATA_FILE = "data/processed_transactions.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Use the same time-based split for this experiment
cutoff = df["step"].quantile(0.8)

train = df[df["step"] <= cutoff]
test = df[df["step"] > cutoff]

# Exclude balance-related features
features = ["step", "type", "amount"]

X_train = train[features]
y_train = train["isFraud"]

X_test = test[features]
y_test = test["isFraud"]

preprocessor = ColumnTransformer(
    [
        (
            "transaction_type",
            OneHotEncoder(handle_unknown="ignore"),
            ["type"]
        )
    ],
    remainder="passthrough"
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("Training comparison model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[
    :, list(model.classes_).index(1)
]

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, predictions))

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)

print(
    "PR-AUC:",
    round(average_precision_score(y_test, probabilities), 6)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 6)
)
