
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Fraud Detection Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# 2. CUSTOM STYLING
# ==========================================================

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #3B82F6;
}

.subtitle {
    color: #94A3B8;
    font-size: 17px;
}

.metric-card {
    background-color: #F1F5F9;
    padding: 20px 10px;
    border-radius: 12px;
    text-align: center;
    min-height: 115px;
}

.metric-label {
    font-size: 15px;
    color: #475569;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #0F172A;
}

.insight-card {
    background-color: #E0F2FE;
    color: #0C4A6E;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. DASHBOARD TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">Fraud Detection Performance Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Day 10 | Model Evaluation and Threshold Analysis</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# 4. LOAD DATA
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "reports" / "day8_threshold_comparison.csv"

if not DATA_FILE.exists():
    st.error(f"Threshold comparison file not found: {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)

required_columns = [
    "threshold",
    "precision",
    "recall",
    "f1_score",
    "true_positives",
    "false_positives",
    "false_negatives",
    "true_negatives"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns in CSV: {missing_columns}")
    st.stop()

if df.empty:
    st.error("The threshold comparison CSV is empty.")
    st.stop()

df = df.sort_values("threshold").reset_index(drop=True)

# ==========================================================
# 5. SIDEBAR CONTROLS
# ==========================================================

st.sidebar.title("Dashboard Controls")

thresholds = df["threshold"].tolist()

best_index = df["f1_score"].idxmax()
best_row = df.loc[best_index]

selected_threshold = st.sidebar.selectbox(
    "Select Classification Threshold",
    thresholds,
    index=len(thresholds) - 1
)

selected_row = df[
    df["threshold"] == selected_threshold
].iloc[0]

st.sidebar.success(
    f"Best observed F1-score threshold: {best_row['threshold']}"
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Use the dropdown to compare model performance "
    "at different classification thresholds."
)

# ==========================================================
# 6. KEY PERFORMANCE INDICATORS
# ==========================================================

st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

metrics = [
    (col1, "Precision", f"{selected_row['precision'] * 100:.2f}%"),
    (col2, "Recall", f"{selected_row['recall'] * 100:.2f}%"),
    (col3, "F1-score", f"{selected_row['f1_score']:.4f}"),
    (col4, "False Positives", f"{int(selected_row['false_positives']):,}")
]

for column, label, value in metrics:
    column.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.caption(
    f"Performance values shown for threshold {selected_threshold}."
)

st.divider()

# ==========================================================
# 7. PERFORMANCE COMPARISON CHART
# ==========================================================

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
    hover_data={
        "Score": ":.4f",
        "threshold": True
    }
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

st.plotly_chart(
    fig_performance,
    use_container_width=True
)

st.info(
    "Interpretation: Precision improves as the threshold increases. "
    "Recall remains at 100% for all tested thresholds in this sample. "
    "F1-score combines precision and recall."
)

st.divider()

# ==========================================================
# 8. CLASSIFICATION ERROR ANALYSIS
# ==========================================================

st.subheader("2. Classification Error Analysis")

fig_errors = go.Figure()

# False Positives
fig_errors.add_trace(
    go.Bar(
        x=df["threshold"],
        y=df["false_positives"],
        name="False Positives",
        text=df["false_positives"],
        textposition="outside",
        hovertemplate=(
            "Threshold: %{x}<br>"
            "False Positives: %{y}<extra></extra>"
        )
    )
)

# False Negatives
fig_errors.add_trace(
    go.Bar(
        x=df["threshold"],
        y=df["false_negatives"],
        name="False Negatives",
        hovertemplate=(
            "Threshold: %{x}<br>"
            "False Negatives: %{y}<extra></extra>"
        )
    )
)

fig_errors.update_layout(
    title="False Positives vs False Negatives",
    xaxis_title="Classification Threshold",
    yaxis_title="Number of Incorrect Predictions",
    barmode="group",
    template="plotly_white",
    height=450,
    yaxis=dict(
        rangemode="tozero",
        dtick=5
    ),
    legend_title="Error Category",
    hovermode="x unified"
)

st.plotly_chart(
    fig_errors,
    use_container_width=True
)

st.markdown("""
**Understanding classification errors:**

- **False Positive:** A legitimate transaction incorrectly flagged as fraud.
- **False Negative:** A fraudulent transaction incorrectly classified as legitimate.
""")

st.info(
    "Observation: False positives decreased from 36 to 1 "
    "across the tested thresholds. False negatives remained zero."
)

st.divider()

# ==========================================================
# 9. CONFUSION MATRIX
# ==========================================================

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
    text_auto=",",
    labels={
        "x": "Predicted Class",
        "y": "Actual Class",
        "color": "Number of Transactions"
    },
    x=["Legitimate", "Fraud"],
    y=["Legitimate", "Fraud"],
    color_continuous_scale="Blues",
    title=f"Confusion Matrix - Threshold {selected_threshold}"
)

fig_matrix.update_layout(
    template="plotly_white",
    height=450
)

st.plotly_chart(
    fig_matrix,
    use_container_width=True
)

# Confusion matrix explanation

cm1, cm2 = st.columns(2)

with cm1:
    st.success(f"True Negatives: {tn:,}")
    st.error(f"False Positives: {fp:,}")

with cm2:
    st.error(f"False Negatives: {fn:,}")
    st.success(f"True Positives: {tp:,}")

st.divider()

# ==========================================================
# 10. DETAILED THRESHOLD COMPARISON
# ==========================================================

st.subheader("4. Detailed Threshold Comparison")

display_df = df.copy()

display_df["precision"] = (
    display_df["precision"] * 100
).round(2)

display_df["recall"] = (
    display_df["recall"] * 100
).round(2)

display_df["f1_score"] = (
    display_df["f1_score"]
).round(4)

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

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ==========================================================
# 11. AUTOMATED MODEL INSIGHTS
# ==========================================================

st.subheader("5. Automated Model Insights")

st.markdown(
    f"""
    <div class="insight-card">
        <b>Best observed threshold:</b> {best_row['threshold']}<br>
        <b>Highest observed F1-score:</b> {best_row['f1_score']:.4f}<br>
        <b>Precision:</b> {best_row['precision'] * 100:.2f}%<br>
        <b>Recall:</b> {best_row['recall'] * 100:.2f}%<br>
        <b>False Positives:</b> {int(best_row['false_positives'])}<br>
        <b>False Negatives:</b> {int(best_row['false_negatives'])}
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "The threshold with the highest observed F1-score is "
    f"{best_row['threshold']}. It achieved a precision of "
    f"{best_row['precision'] * 100:.2f}% and recall of "
    f"{best_row['recall'] * 100:.2f}% on this test sample."
)

st.warning(
    "Important: This is the best observed threshold among the "
    "tested values, not necessarily the optimal production threshold. "
    "Additional chronological validation and business-cost analysis "
    "are required before real-world deployment."
)

st.divider()

# ==========================================================
# 12. FOOTER
# ==========================================================

st.caption(
    "Real-Time Financial Fraud Detection Pipeline | "
    "Day 10 Performance Dashboard | "
    "Python | Streamlit | Pandas | Plotly"
)
