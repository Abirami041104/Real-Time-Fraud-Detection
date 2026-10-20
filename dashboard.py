
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------
# 1. Dashboard configuration
# -------------------------------
st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "data" / "live_predictions.csv"

st.title("🛡️ Real-Time Financial Fraud Detection")
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
st.sidebar.header("🔎 Filters")

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
col5.metric("Total Amount", f"₹{total_amount:,.2f}")

st.divider()

# -------------------------------
# 6. Charts
# -------------------------------
st.subheader("📊 Transaction Analytics")

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
st.subheader("🚨 Fraud Alerts")

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
                "Amount (₹)", format="₹%.2f"
            ),
            "Fraud Probability (%)": st.column_config.NumberColumn(
                "Fraud Probability (%)", format="%.2f%%"
            ),
        },
    )

# -------------------------------
# 8. Transaction history
# -------------------------------
st.subheader("📋 Transaction History")

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
            "Amount (₹)", format="₹%.2f"
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

# Refresh the dashboard periodically.
st.html(
    '<meta http-equiv="refresh" content="10">'
)
