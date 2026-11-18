
import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("DAY 11 - SHAP MODEL EXPLAINABILITY")
print("=" * 60)

# 1. File paths
MODEL_FILE = "models/fraud_detection_model.pkl"
DATA_FILE = "data/processed_transactions.csv"
OUTPUT_DIR = "reports/day11_shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load trained pipeline
print("\nLoading trained model...")
model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")

# 3. Load a manageable sample
print("\nLoading transaction sample...")

df = pd.read_csv(DATA_FILE)

# Keep only the model input features
feature_columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_origin_balance"
]

# Sample transactions for analysis
sample_df = df.sample(
    n=1000,
    random_state=42
).copy()

X_sample = sample_df[feature_columns]

print("Sample size:", len(X_sample))
print("Fraud cases:", int(sample_df["isFraud"].sum()))

# 4. Apply the existing preprocessing
print("\nTransforming features...")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X_sample)

feature_names = preprocessor.get_feature_names_out()

X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X_sample.index
)

print("Transformed feature count:", len(feature_names))

# 5. Create SHAP TreeExplainer
print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(
    X_transformed_df,
    check_additivity=False
)

# Handle SHAP output format for binary classification
if isinstance(shap_values, list):
    fraud_shap_values = shap_values[1]
elif shap_values.ndim == 3:
    fraud_shap_values = shap_values[:, :, 1]
else:
    fraud_shap_values = shap_values

print("SHAP calculation completed!")

# 6. Global feature importance
print("\nGenerating global feature importance...")

importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": abs(fraud_shap_values).mean(axis=0)
})

importance = importance.sort_values(
    by="mean_absolute_shap",
    ascending=False
)

importance.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"),
    index=False
)

print("\nTOP FEATURES")
print(importance.head(10).to_string(index=False))

# 7. SHAP summary plot
print("\nCreating SHAP summary plot...")

shap.summary_plot(
    fraud_shap_values,
    X_transformed_df,
    show=False,
    max_display=15
)

plt.title("SHAP Summary - Fraud Prediction")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_summary.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 8. SHAP bar plot
print("Creating SHAP importance bar plot...")

shap.summary_plot(
    fraud_shap_values,
    X_transformed_df,
    plot_type="bar",
    show=False,
    max_display=15
)

plt.title("Global Feature Importance - SHAP")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_bar.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 9. Explain one fraud transaction
fraud_indices = sample_df.index[
    sample_df["isFraud"] == 1
]

if len(fraud_indices) > 0:

    fraud_index = fraud_indices[0]
    position = X_sample.index.get_loc(fraud_index)

    print("\nExplaining one fraud transaction...")
    print(X_sample.loc[fraud_index])

    # Extract a scalar expected value for the fraud class
    import numpy as np

    expected_values = np.asarray(
        explainer.expected_value
    ).reshape(-1)

    fraud_base_value = float(expected_values[1])

    explanation = shap.Explanation(
        values=fraud_shap_values[position],
        base_values=fraud_base_value,
        data=X_transformed_df.iloc[position].values,
        feature_names=feature_names
    )

    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False
    )

    plt.title("SHAP Explanation - Fraud Transaction")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "fraud_waterfall.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

else:
    print("No fraud transaction found in sample.")

# 10. Save sample predictions
predictions = model.predict_proba(X_sample)[:, 1]

prediction_df = X_sample.copy()
prediction_df["actual_isFraud"] = sample_df["isFraud"].values
prediction_df["fraud_probability"] = predictions

prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "sample_predictions.csv"),
    index=False
)

print("\n" + "=" * 60)
print("DAY 11 SHAP ANALYSIS COMPLETED")
print("Output folder:", OUTPUT_DIR)
print("=" * 60)
