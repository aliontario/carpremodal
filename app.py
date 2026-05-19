import pickle
import numpy as np
import pandas as pd
import streamlit as st

# 1. Model aur Columns load karna
with open("car_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# 2. UI Setup
st.title("🚗 Used Car Price Prediction App")
st.write("Enter the car features to predict its estimated selling price:")

# Inputs
present_price = st.number_input(
    "Present Showroom Price (in Lakhs, e.g., 5.5)",
    min_value=0.1,
    max_value=100.0,
    value=6.0,
)
kms_driven = st.number_input(
    "Kilometers Driven", min_value=0, max_value=500000, value=30000
)
car_age = st.number_input(
    "Age of the Car (in Years)", min_value=0, max_value=30, value=5
)
owner = st.selectbox("Number of Previous Owners", [0, 1, 3])

seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

# 3. Prediction Button
if st.button("Predict Selling Price"):
    input_data = {
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Owner": owner,
        "Car_Age": car_age,
        "Transmission": 1 if transmission == "Automatic" else 0,
        "Seller_Type": 1 if seller_type == "Individual" else 0,
        "Fuel_Type_Diesel": 1 if fuel_type == "Diesel" else 0,
        "Fuel_Type_Petrol": 1 if fuel_type == "Petrol" else 0,
    }

    input_df = pd.DataFrame([input_data]).reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_df)
    final_price = max(0.0, prediction[0])

    st.success(f"💰 Estimated Selling Price: {final_price:.2f} Lakhs")
