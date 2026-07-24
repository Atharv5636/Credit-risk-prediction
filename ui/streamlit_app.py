"""Credit Card Default Risk Predictor — Streamlit UI.

Run:  streamlit run ui/streamlit_app.py
The form collects the 9 highest-signal inputs; the remaining features are
auto-filled from training-set medians, and the 2 engineered features are
computed on the fly — so the saved scaler/model always receive all 25.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURE_ORDER = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + [f"PAY_{i}" for i in range(1, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
    + ["AVG_UTILIZATION", "TOTAL_DELAY"]
)


@st.cache_resource
def load_artifacts():
    return (joblib.load(os.path.join(MODELS, "sgd_best.joblib")),
            joblib.load(os.path.join(MODELS, "scaler.joblib")),
            joblib.load(os.path.join(MODELS, "defaults.joblib")))


def assemble_row(inputs: dict, defaults: dict) -> pd.DataFrame:
    """Start from medians, override with user inputs, recompute engineered features."""
    row = dict(defaults)
    row.update(inputs)
    bills = [row[f"BILL_AMT{i}"] for i in range(1, 7)]
    delays = [row[f"PAY_{i}"] for i in range(1, 7)]
    row["AVG_UTILIZATION"] = float(np.clip(np.mean(bills) / max(row["LIMIT_BAL"], 1), 0, 5))
    row["TOTAL_DELAY"] = float(sum(max(d, 0) for d in delays))
    return pd.DataFrame([row])[FEATURE_ORDER]


PAY_LABELS = {-1: "Paid duly", 0: "Revolving credit", **{i: f"{i} mo late" for i in range(1, 9)}}

st.set_page_config(page_title="Credit Default Risk", page_icon="💳", layout="centered")
st.title("💳 Credit Card Default Risk Predictor")
st.caption("Tuned SGD classifier · UCI Taiwan dataset · ~83% accuracy")

try:
    model, scaler, defaults = load_artifacts()
except FileNotFoundError:
    st.error("Model not found. Run `python src/train.py` first to generate the artifacts.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    limit = st.number_input("Credit Limit (NT$)", 10_000, 1_000_000, 200_000, step=10_000)
    age = st.slider("Age", 21, 79, 35)
    edu = st.selectbox("Education", [1, 2, 3, 4],
                       format_func=lambda x: {1: "Grad school", 2: "University",
                                              3: "High school", 4: "Other"}[x])
    marriage = st.selectbox("Marital status", [1, 2, 3],
                            format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x])
with c2:
    pay_1 = st.selectbox("Most recent repayment status", list(PAY_LABELS),
                         index=1, format_func=lambda x: PAY_LABELS[x])
    pay_2 = st.selectbox("Repayment status (1 mo prior)", list(PAY_LABELS),
                         index=1, format_func=lambda x: PAY_LABELS[x])
    bill_1 = st.number_input("Latest bill amount (NT$)", 0, 1_000_000, 50_000, step=5_000)
    pay_amt_1 = st.number_input("Latest payment made (NT$)", 0, 1_000_000, 5_000, step=1_000)

threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05,
                      help="Lower = catch more defaulters (higher recall); higher = fewer false alarms.")

if st.button("Predict Risk", type="primary", use_container_width=True):
    inputs = {"LIMIT_BAL": limit, "AGE": age, "EDUCATION": edu, "MARRIAGE": marriage,
              "PAY_1": pay_1, "PAY_2": pay_2, "BILL_AMT1": bill_1, "PAY_AMT1": pay_amt_1}
    X = assemble_row(inputs, defaults)
    proba = float(model.predict_proba(scaler.transform(X))[0, 1])

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Default Probability", f"{proba:.1%}")
    m2.metric("Decision", "HIGH RISK" if proba >= threshold else "LOW RISK")
    st.progress(min(proba, 1.0))
    if proba >= threshold:
        st.error(f"⚠️ Flagged as **HIGH RISK** at the {threshold:.0%} threshold.")
    else:
        st.success(f"✅ **LOW RISK** at the {threshold:.0%} threshold.")
    with st.expander("Full feature vector sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}))
