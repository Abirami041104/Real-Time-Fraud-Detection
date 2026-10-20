
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

print("Loading dataset...")
df = pd.read_csv("data/processed_transactions.csv")

cutoff = df["step"].quantile(0.8)

train = df[df["step"] <= cutoff]
test = df[df["step"] > cutoff]

# Keep original balances; remove the engineered ratio
features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]

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

print("Training model without the engineered ratio...")
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
    "\nPR-AUC:",
    round(average_precision_score(y_test, probabilities), 6)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 6)
)
