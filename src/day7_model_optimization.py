import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_FILE = "data/processed_transactions.csv"
OUTPUT_FILE = "reports/day7_model_optimization.csv"

RANDOM_STATE = 42
SAMPLE_SIZE = 500_000

print("=" * 65)
print("DAY 7: RANDOM FOREST MODEL OPTIMIZATION")
print("=" * 65)

# 1. Load dataset
print("\nLoading dataset...")
df = pd.read_csv(DATA_FILE)

print("Original rows:", len(df))

# 2. Create a representative stratified sample
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

# 4. Create one common train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Define optimization experiments
experiments = {
    "Baseline": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 2
    },
    "More_Trees": {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_leaf": 2
    },
    "Deeper_Trees": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 2
    },
    "Larger_Min_Leaf": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 5
    }
}

results = []

# 6. Train and evaluate each configuration
for experiment_name, parameters in experiments.items():

    print("\n" + "-" * 55)
    print("Experiment:", experiment_name)
    print("Parameters:", parameters)

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
                    **parameters,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    probabilities = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "experiment": experiment_name,
        **parameters,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "training_time_seconds": round(training_time, 2)
    })

    print("PR-AUC:", round(pr_auc, 6))
    print("ROC-AUC:", round(roc_auc, 6))
    print("Training time:", round(training_time, 2), "seconds")

# 7. Save results
os.makedirs("reports", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("FINAL MODEL OPTIMIZATION COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nResults saved:", OUTPUT_FILE)
print("\nDay 7 optimization completed!")