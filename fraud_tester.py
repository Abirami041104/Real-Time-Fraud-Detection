
import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "models/fraud_detection_model.pkl"

st.set_page_config(
    page_title="Fraud Transaction Tester",
    page_icon="🔎",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.stop()

st.title("🔎 Interactive Fraud Transaction Tester")
st.caption(
    "Enter transaction details and test your existing "
    "Random Forest model."
)

st.info(
    "This is a model prediction, not confirmation that "
    "a real transaction is fraudulent."
)

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction step",
            min_value=1,
            value=100,
            step=1
        )

        transaction_type = st.selectbox(
            "Transaction type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
        )

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=15000.0,
            step=100.0
        )

    with col2:
        oldbalance_org = st.number_input(
            "Sender's original balance",
            min_value=0.0,
            value=20000.0,
            step=100.0
        )

        oldbalance_dest = st.number_input(
            "Receiver's original balance",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

    submitted = st.form_submit_button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True
    )

if submitted:
    ratio = amount / (oldbalance_org + 1)

    transaction = pd.DataFrame([{
        "step": int(step),
        "type": transaction_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalance_org),
        "oldbalanceDest": float(oldbalance_dest),
        "amount_to_origin_balance": float(ratio)
    }])

    try:
        prediction = int(model.predict(transaction)[0])
        probabilities = model.predict_proba(transaction)[0]
        classes = list(model.classes_)
        fraud_probability = float(
            probabilities[classes.index(1)]
        )

        st.subheader("Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("🚨 FRAUD ALERT")
            else:
                st.success("✅ NOT FLAGGED")

        with col2:
            st.metric(
                "Estimated fraud probability",
                f"{fraud_probability:.2%}"
            )

        st.progress(fraud_probability)

        st.write("**Transaction details**")
        st.dataframe(
            transaction,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The prediction depends on the model's learned "
            "patterns and is not a guarantee of transaction safety."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
