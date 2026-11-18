
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection | Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 700;
    color: #2563eb;
}
.subtitle {
    color: #64748b;
    font-size: 16px;
}
.metric-card {
    background-color: #f1f5f9;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}
.metric-label {
    font-size: 15px;
    color: #475569;
}
.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Fraud Detection Performance Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Day 10 | Model Evaluation and Threshold Analysis</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "reports" / "day8_threshold_comparison.csv"

if not DATA_FILE.exists():
    st.error(f"Required data file not found: {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)

required_columns = [
    "threshold", "precision", "recall", "f1_score",
    "true_positives", "false_positives",
    "false_negatives", "true_negatives"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

if df.empty:
    st.error("The threshold comparison file is empty.")
    st.stop()

# --------------------------------------------------
# THRESHOLD SELECTION
# --------------------------------------------------

st.sidebar.header("Dashboard Controls")

thresholds = sorted(df["threshold"].unique())

selected_threshold = st.sidebar.selectbox(
    "Select Classification Threshold",
    thresholds,
    index=len(thresholds) - 1
)

selected_row = df[df["threshold"] == selected_threshold].iloc[0]

best_row = df.loc[df["f1_score"].idxmax()]

st.sidebar.info(
    f"Best observed F1-score threshold: {best_row['threshold']}"
)

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

metrics = [
    (col1, "Precision", selected_row["precision"], True),
    (col2, "Recall", selected_row["recall"], True),
    (col3, "F1-score", selected_row["f1_score"], False),
    (col4, "False Positives", selected_row["false_positives"], False)
]

for column, label, value, percentage in metrics:
    display_value = (
        f"{value * 100:.2f}%"
        if percentage
        else f"{value:,.0f}" if label == "False Positives"
        else f"{value:.4f}"
    )

    column.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display_value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.caption(
    f"Metrics shown above are for the selected threshold: {selected_threshold}"
)

st.divider()

# --------------------------------------------------
# PERFORMANCE METRICS CHART
# --------------------------------------------------

st.subheader("1. Performance Across Thresholds")

performance_df = df.melt(
    id_vars="threshold",
    value_vars=["precision", "recall", "f1_score"],
    var_name="Metric",
    value_name="Score"
)

performance_df["Metric"] = performance_df["Metric"].replace({
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score"
})


fig_performance = px.line(
    performance_df,
    x="threshold",
    y="Score",
    color="Metric",
    markers=True,
    title="Precision, Recall and F1-score Across Thresholds",
    hover_data={"Score": ":.4f", "threshold": True}
)

fig_performance.update_traces(
    line=dict(width=3),
    marker=dict(size=10),
    mode="lines+markers"
)

fig_performance.update_layout(
    yaxis=dict(
        title="Performance Score",
        range=[0.70, 1.02],
        tickformat=".0%"
    ),
    xaxis_title="Classification Threshold",
    hovermode="x unified",
    legend_title="Performance Metrics",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_performance, use_container_width=True)

st.info(
    "Interpretation: Precision increases as the threshold rises, "
    "while recall remains at 100% in this test sample. "
    "The F1-score combines precision and recall."
)

st.divider()

# --------------------------------------------------
# CLASSIFICATION ERRORS
# --------------------------------------------------

st.subheader("2. Classification Error Analysis")

error_df = df.melt(
    id_vars="threshold",
    value_vars=["false_positives", "false_negatives"],
    var_name="Error Type",
    value_name="Number of Transactions"
)

error_df["Error Type"] = error_df["Error Type"].replace({
    "false_positives": "False Positives",
    "false_negatives": "False Negatives"
})

fig_errors = px.bar(
    error_df,
    x="threshold",
    y="Number of Transactions",
    color="Error Type",
    barmode="group",
    text="Number of Transactions",
    title="False Positives vs False Negatives"
)

fig_errors.update_traces(textposition="outside")

fig_errors.update_layout(
    xaxis_title="Classification Threshold",
    yaxis_title="Number of Transactions"
)

st.plotly_chart(fig_errors, use_container_width=True)

st.info(
    "Interpretation: False positives are legitimate transactions "
    "incorrectly flagged as fraud. False negatives are fraudulent "
    "transactions missed by the model."
)

st.divider()

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("3. Confusion Matrix")

tn = int(selected_row["true_negatives"])
fp = int(selected_row["false_positives"])
fn = int(selected_row["false_negatives"])
tp = int(selected_row["true_positives"])

matrix = [
    [tn, fp],
    [fn, tp]
]

fig_matrix = px.imshow(
    matrix,
    text_auto=True,
    labels={
        "x": "Predicted Class",
        "y": "Actual Class",
        "color": "Transactions"
    },
    x=["Legitimate", "Fraud"],
    y=["Legitimate", "Fraud"],
    color_continuous_scale="Blues",
    title=f"Confusion Matrix at Threshold {selected_threshold}"
)

st.plotly_chart(fig_matrix, use_container_width=True)

st.markdown("""
**How to read the matrix:**

- True Negative (TN): Legitimate transaction correctly identified.
- False Positive (FP): Legitimate transaction incorrectly flagged.
- False Negative (FN): Fraudulent transaction missed.
- True Positive (TP): Fraudulent transaction correctly identified.
""")

st.divider()

# --------------------------------------------------
# THRESHOLD COMPARISON TABLE
# --------------------------------------------------

st.subheader("4. Detailed Threshold Comparison")

display_df = df.copy()

display_df["precision"] = (display_df["precision"] * 100).round(2)
display_df["recall"] = (display_df["recall"] * 100).round(2)
display_df["f1_score"] = display_df["f1_score"].round(4)

display_df = display_df.rename(columns={
    "threshold": "Threshold",
    "precision": "Precision (%)",
    "recall": "Recall (%)",
    "f1_score": "F1-score",
    "true_positives": "True Positives",
    "false_positives": "False Positives",
    "false_negatives": "False Negatives",
    "true_negatives": "True Negatives"
})

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------
# AUTOMATIC INSIGHTS
# --------------------------------------------------

st.subheader("5. Automated Model Insights")

st.write(
    f"**Best observed threshold:** {best_row['threshold']}"
)

st.write(
    f"**Highest observed F1-score:** {best_row['f1_score']:.4f}"
)

st.write(
    f"**Precision at best threshold:** {best_row['precision'] * 100:.2f}%"
)

st.write(
    f"**Recall at best threshold:** {best_row['recall'] * 100:.2f}%"
)

st.write(
    f"**False positives at best threshold:** "
    f"{int(best_row['false_positives'])}"
)

st.write(
    f"**False negatives at best threshold:** "
    f"{int(best_row['false_negatives'])}"
)

st.warning(
    "Important: The best observed threshold is based on F1-score "
    "among the tested values. It is not automatically the best "
    "production threshold. Real deployment requires chronological "
    "validation and consideration of false-alert and missed-fraud costs."
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Real-Time Financial Fraud Detection Pipeline | "
    "Day 10 Performance Analysis | Python + Streamlit + Plotly"
)