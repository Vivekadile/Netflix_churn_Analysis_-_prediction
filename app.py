import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# load model + scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction")

# user inputs (keep important ones)
age = st.slider("Age", 18, 60, 30)
watch_hours = st.slider("Watch Hours", 0.0, 30.0, 5.0)
last_login_days = st.slider("Last Login Days", 0, 60, 5)
monthly_fee = st.selectbox("Monthly Fee", [8.99, 13.99, 17.99])
subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

if st.button("Predict"):

    # create base dataframe with ALL features = 0
    input_data = pd.DataFrame(np.zeros((1, len(scaler.feature_names_in_))),
                             columns=scaler.feature_names_in_)

    # fill known values
    input_data["age"] = age
    input_data["watch_hours"] = watch_hours
    input_data["last_login_days"] = last_login_days
    input_data["monthly_fee"] = monthly_fee

    # subscription encoding
    if subscription == "Premium":
        input_data["subscription_type_Premium"] = 1
    elif subscription == "Standard":
        input_data["subscription_type_Standard"] = 1

    # scale
    input_scaled = scaler.transform(input_data)

    # predict
    prob = model.predict(input_scaled)[0][0]

    st.write("Churn Probability:", round(prob, 3))

    if prob > 0.5:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")