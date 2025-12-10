import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Load trained model
model = joblib.load("electricity_forecaster.pkl")

st.title("Electricity Usage Forecasting App")
st.write("Predict next-hour electricity consumption based on recent history.")

st.subheader("Input Features")

# 1. Date & time inputs
col1, col2 = st.columns(2)

with col1:
    date_input = st.date_input("Date", datetime(2020, 1, 4))
with col2:
    time_input = st.time_input("Time", datetime(2020, 1, 4, 12).time())

dt = datetime.combine(date_input, time_input)

hour = dt.hour
day = dt.day
weekday = dt.weekday()
month = dt.month

st.write(f"Hour: {hour}, Day: {day}, Weekday: {weekday}, Month: {month}")

# 2. Last 3 hours consumption (user input)
st.subheader("Last 3 Hours Energy Usage (kWh)")
lag1 = st.number_input("1 hour ago (lag1)", min_value=0.0, value=3.5, step=0.1)
lag2 = st.number_input("2 hours ago (lag2)", min_value=0.0, value=3.4, step=0.1)
lag3 = st.number_input("3 hours ago (lag3)", min_value=0.0, value=3.3, step=0.1)

# 3. Rolling mean of last 3 hours (user can leave default or recompute)
rolling_mean_3 = st.number_input(
    "Rolling mean (last 3 hours)", 
    value=float(np.mean([lag1, lag2, lag3])), 
    step=0.1
)

if st.button("Predict Next-Hour Usage"):
    # Build a single-row DataFrame with same feature names
    input_df = pd.DataFrame([{
        "hour": hour,
        "day": day,
        "weekday": weekday,
        "month": month,
        "lag1": lag1,
        "lag2": lag2,
        "lag3": lag3,
        "rolling_mean_3": rolling_mean_3
    }])

    # Predict
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction")
    st.write(f"**Predicted next-hour energy usage:** {prediction:.3f} kWh")