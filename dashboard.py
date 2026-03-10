import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# ── Auto-train model if model.pkl doesn't exist ──────────────────────────────
@st.cache_resource
def load_or_train_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    # Generate synthetic training data based on known market relationships:
    # Risk ON  (sp500 up) → oil up, gold flat/down
    # Risk OFF (sp500 down) → oil down, gold up
    np.random.seed(42)
    n = 1000

    oil_return  = np.concatenate([np.random.normal( 0.01, 0.02, n//2),
                                   np.random.normal(-0.01, 0.02, n//2)])
    gold_return = np.concatenate([np.random.normal(-0.005, 0.015, n//2),
                                   np.random.normal( 0.010, 0.015, n//2)])
    target      = np.array([1]*(n//2) + [0]*(n//2))

    df = pd.DataFrame({"oil_return": oil_return,
                        "gold_return": gold_return,
                        "target": target})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(df[["oil_return", "gold_return"]], df["target"])
    joblib.dump(model, "model.pkl")
    return model

model = load_or_train_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Global Risk Dashboard", page_icon="🌍", layout="centered")

st.title("🌍 Global Risk Dashboard")
st.markdown("Adjust the market signals below to get a **Risk ON / Risk OFF** prediction.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    oil = st.slider("🛢️ Oil Return", -0.10, 0.10, 0.01, step=0.005,
                    help="Daily % return of crude oil (e.g. 0.02 = +2%)")
with col2:
    gold = st.slider("🥇 Gold Return", -0.10, 0.10, 0.01, step=0.005,
                     help="Daily % return of gold (e.g. -0.01 = -1%)")

st.divider()

data = pd.DataFrame([{"oil_return": oil, "gold_return": gold}])
prediction = model.predict(data)[0]
proba      = model.predict_proba(data)[0]
confidence = int(max(proba) * 100)

if prediction == 1:
    st.success(f"### ✅ Risk ON — Markets Bullish  ({confidence}% confidence)")
    st.markdown("Oil rising + Gold flat/falling → investors are risk-seeking.")
else:
    st.warning(f"### ⚠️ Risk OFF — Markets Defensive  ({confidence}% confidence)")
    st.markdown("Oil falling + Gold rising → investors are seeking safe havens.")

st.divider()
st.caption("Model trained on synthetic market relationship data. Not financial advice.")
