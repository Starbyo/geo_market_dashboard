
import streamlit as st
import joblib
import pandas as pd

st.title("Global Risk Dashboard")

model = joblib.load("model.pkl")

oil = st.slider("Oil movement signal",-0.1,0.1,0.01)
gold = st.slider("Gold movement signal",-0.1,0.1,0.01)

data = pd.DataFrame([{
"oil_return": oil,
"gold_return": gold
}])

prediction = model.predict(data)

if prediction[0] == 1:
    st.success("Risk ON – markets bullish")
else:
    st.warning("Risk OFF – markets defensive")
