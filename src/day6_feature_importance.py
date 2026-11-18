
import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_FILE = "data/processed_transactions.csv"
OUTPUT_FILE = "reports/day6_feature_ablation.csv"

print("=" * 65)
print("DAY 6: INDIVIDUAL FEATURE CONTRIBUTION ANALYSIS")
print("=" * 65)

print("\nLoading transaction data...")
df = pd.read_csv(DATA_FILE)

# Use a chronological split
cutoff = df["step"].quantile(0.8)

train = df[df["step"] <= cutoff]
test = df[df["step"] > cutoff]

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Time cutoff:", cutoff)

# Three feature combinations
experiments = {
    "All_features": [
        "step", "type", "amount",
        "oldbalanceOrg", "oldbalanceDest",
        "amount_to_origin_balance"
    ],
    "Without_ratio": [
        "step", "type", "amount",
        "oldbalanceOrg", "oldbalanceDest"
    ],
    "Without_balance_features": [
        "step", "type", "amount"
    ]
}

results = []

for experiment_name, features in experiments.items():

    print("\nRunning:", experiment_name)

    X_train = train[features]
    y_train = train["isFraud"]

    X_test = test[features]
    y_test = test["isFraud"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "transaction_type",
                OneHotEncoder(handle_unknown="ignore"),
                ["type"]
            )
        ],
        remainder="passthrough"
    )

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
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "experiment": experiment_name,
        "feature_count": len(features),
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc
    })

    print("PR-AUC:", round(pr_auc, 6))
    print("ROC-AUC:", round(roc_auc, 6))

# Save comparison
os.makedirs("reports", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nFINAL FEATURE COMPARISON")
print(results_df.to_string(index=False))

print("\nReport saved:", OUTPUT_FILE)
print("\nDay 6 analysis completed!")
