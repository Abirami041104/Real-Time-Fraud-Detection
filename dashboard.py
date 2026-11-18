
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------
# 1. Dashboard configuration
# -------------------------------
st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="ðŸ›¡ï¸",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "data" / "live_predictions.csv"

st.title("ðŸ›¡ï¸ Real-Time Financial Fraud Detection")
st.caption("Live transaction monitoring powered by Machine Learning and Kafka")

# -------------------------------
# 2. Load live predictions
# -------------------------------
if not CSV_FILE.exists():
    st.warning("No predictions found yet.")
    st.info(
        "Start your Kafka consumer and producer first. "
        "The dashboard will display predictions saved to "
        "data/live_predictions.csv."
    )
    st.stop()

try:
    df = pd.read_csv(CSV_FILE)
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not read predictions file: {error}")
    st.stop()

required_columns = [
    "timestamp",
    "transaction_id",
    "type",
    "amount",
    "prediction",
    "fraud_probability",
    "status",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns in CSV: {', '.join(missing_columns)}")
    st.stop()

if df.empty:
    st.info("Waiting for transaction predictions...")
    st.stop()

# -------------------------------
# 3. Prepare data
# -------------------------------
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
df["fraud_probability"] = pd.to_numeric(
    df["fraud_probability"], errors="coerce"
).fillna(0)

df = df.dropna(subset=["amount", "prediction"]).copy()
df["prediction"] = df["prediction"].astype(int)
df["fraud_probability_pct"] = df["fraud_probability"] * 100

# -------------------------------
# 4. Sidebar filters
# -------------------------------
st.sidebar.header("ðŸ”Ž Filters")

transaction_types = sorted(df["type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Transaction type",
    options=transaction_types,
    default=transaction_types,
)

statuses = sorted(df["status"].dropna().unique().tolist())
selected_statuses = st.sidebar.multiselect(
    "Prediction status",
    options=statuses,
    default=statuses,
)

filtered_df = df[
    df["type"].isin(selected_types)
    & df["status"].isin(selected_statuses)
].copy()

st.sidebar.caption(f"Showing {len(filtered_df)} of {len(df)} transactions")

# -------------------------------
# 5. KPI cards
# -------------------------------
total_transactions = len(filtered_df)
fraud_count = int((filtered_df["prediction"] == 1).sum())
normal_count = total_transactions - fraud_count
fraud_rate = (
    fraud_count / total_transactions * 100
    if total_transactions > 0 else 0
)
total_amount = filtered_df["amount"].sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Transactions", f"{total_transactions:,}")
col2.metric("Fraud Alerts", f"{fraud_count:,}")
col3.metric("Not Flagged", f"{normal_count:,}")
col4.metric("Predicted Fraud Rate", f"{fraud_rate:.2f}%")
col5.metric("Total Amount", f"â‚¹{total_amount:,.2f}")

st.divider()

# -------------------------------
# 6. Charts
# -------------------------------
st.subheader("ðŸ“Š Transaction Analytics")

chart1, chart2 = st.columns(2)

with chart1:
    status_counts = (
        filtered_df["status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Count")
    )

    if not status_counts.empty:
        fig = px.pie(
            status_counts,
            names="Status",
            values="Count",
            title="Fraud Alerts vs Not Flagged",
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

with chart2:
    type_counts = (
        filtered_df.groupby(["type", "status"])
        .size()
        .reset_index(name="Count")
    )

    if not type_counts.empty:
        fig = px.bar(
            type_counts,
            x="type",
            y="Count",
            color="status",
            barmode="group",
            title="Transactions by Type and Prediction",
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 7. Fraud alerts
# -------------------------------
st.subheader("ðŸš¨ Fraud Alerts")

fraud_df = filtered_df[
    filtered_df["prediction"] == 1
].sort_values("fraud_probability", ascending=False)

if fraud_df.empty:
    st.success("No predicted fraud transactions in the current filter.")
else:
    alert_columns = [
        "timestamp",
        "transaction_id",
        "type",
        "amount",
        "fraud_probability_pct",
        "status",
    ]
    alerts = fraud_df[alert_columns].rename(
        columns={"fraud_probability_pct": "Fraud Probability (%)"}
    )
    st.dataframe(
        alerts,
        use_container_width=True,
        hide_index=True,
        column_config={
            "amount": st.column_config.NumberColumn(
                "Amount (â‚¹)", format="â‚¹%.2f"
            ),
            "Fraud Probability (%)": st.column_config.NumberColumn(
                "Fraud Probability (%)", format="%.2f%%"
            ),
        },
    )

# -------------------------------
# 8. Transaction history
# -------------------------------
st.subheader("ðŸ“‹ Transaction History")

history = filtered_df.sort_values("timestamp", ascending=False).copy()
history["fraud_probability_pct"] = (
    history["fraud_probability"] * 100
)

display_columns = [
    "timestamp",
    "transaction_id",
    "type",
    "amount",
    "prediction",
    "fraud_probability_pct",
    "status",
]

history = history[display_columns].rename(
    columns={
        "fraud_probability_pct": "Fraud Probability (%)",
        "prediction": "Predicted Class",
    }
)

st.dataframe(
    history,
    use_container_width=True,
    hide_index=True,
    column_config={
        "amount": st.column_config.NumberColumn(
            "Amount (â‚¹)", format="â‚¹%.2f"
        ),
        "Fraud Probability (%)": st.column_config.NumberColumn(
            "Fraud Probability (%)", format="%.2f%%"
        ),
        "Predicted Class": st.column_config.NumberColumn(
            "Predicted Class", format="%d"
        ),
    },
)

st.caption(
    "Note: Predictions are generated by your trained model. "
    "A fraud alert is a model prediction, not proof of fraud."
)


# -------------------------------
# 9. SHAP MODEL EXPLAINABILITY
# -------------------------------

st.divider()
st.header("Model Explainability")
st.caption("Understanding model predictions using SHAP analysis.")

SHAP_DIR = BASE_DIR / "reports" / "day11_shap"

feature_file = SHAP_DIR / "feature_importance.csv"
summary_file = SHAP_DIR / "shap_summary.png"
bar_file = SHAP_DIR / "shap_bar.png"
waterfall_file = SHAP_DIR / "fraud_waterfall.png"
predictions_file = SHAP_DIR / "sample_predictions.csv"

# Global Feature Importance
st.subheader("1. Global Feature Importance")

if feature_file.exists():
    shap_df = pd.read_csv(feature_file)

    feature_names = {
        "remainder__amount_to_origin_balance": "Amount / Origin Balance",
        "remainder__amount": "Transaction Amount",
        "transaction_type__type_PAYMENT": "Payment Type",
        "remainder__oldbalanceOrg": "Old Origin Balance",
        "transaction_type__type_CASH_IN": "Cash In Type",
        "remainder__oldbalanceDest": "Old Destination Balance",
        "transaction_type__type_CASH_OUT": "Cash Out Type",
        "transaction_type__type_TRANSFER": "Transfer Type",
        "remainder__step": "Transaction Step",
        "transaction_type__type_DEBIT": "Debit Type",
    }

    shap_df["Feature"] = shap_df["feature"].map(feature_names).fillna(shap_df["feature"])

    shap_df = shap_df.sort_values("mean_absolute_shap", ascending=True)

    fig = px.bar(
        shap_df,
        x="mean_absolute_shap",
        y="Feature",
        orientation="h",
        title="SHAP Feature Importance",
        labels={
            "mean_absolute_shap": "Mean Absolute SHAP Value",
            "Feature": "Feature",
        },
        text="mean_absolute_shap",
    )

    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance Details")

    st.dataframe(
        shap_df[["Feature", "mean_absolute_shap"]].sort_values(
            "mean_absolute_shap", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "mean_absolute_shap": st.column_config.NumberColumn(
                "Mean Absolute SHAP",
                format="%.6f",
            )
        },
    )
else:
    st.warning("Feature importance CSV not found.")

# SHAP Summary Plot
st.divider()
st.subheader("2. SHAP Summary Plot")

if summary_file.exists():
    st.image(str(summary_file), caption="Feature Impact Distribution", use_container_width=True)
else:
    st.warning("SHAP summary image not found.")

# SHAP Bar Plot
st.divider()
st.subheader("3. SHAP Importance Bar Plot")

if bar_file.exists():
    st.image(str(bar_file), caption="Overall Feature Importance", use_container_width=True)
else:
    st.warning("SHAP bar image not found.")

# Individual Fraud Explanation
st.divider()
st.subheader("4. Individual Fraud Transaction Explanation")

if waterfall_file.exists():
    st.write(
        "This waterfall plot explains one fraud prediction "
        "from the Day 11 analysis."
    )
    st.image(str(waterfall_file), caption="SHAP Waterfall Explanation", use_container_width=True)
else:
    st.warning("Fraud waterfall image not found.")

# Sample Predictions
st.divider()
st.subheader("5. SHAP Sample Predictions")

if predictions_file.exists():
    predictions_df = pd.read_csv(predictions_file)
    st.dataframe(predictions_df, use_container_width=True, hide_index=True)
else:
    st.info("Sample predictions file not found.")

st.caption(
    "These are saved Day 11 SHAP results, not newly calculated "
    "explanations for each live transaction."
)

# Refresh the dashboard periodically.
st.html(
    '<meta http-equiv="refresh" content="10">'
)

