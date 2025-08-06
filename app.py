import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --- Load model and objects with caching ---
@st.cache_resource
def load_model_file():
    return load_model('model.keras')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

@st.cache_resource
def load_features():
    return joblib.load('features.pkl')

# Load once
model = load_model_file()
scaler = load_scaler()
features = load_features()

# --- Page Config ---
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📱 Telco Customer Churn Prediction")
st.write("Fill out the form below to predict whether a customer will churn.")

# --- Input Form ---
def user_input_features():
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen_display = st.selectbox("Senior Citizen", ["No", "Yes"])
    SeniorCitizen = 1 if SeniorCitizen_display == "Yes" else 0
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
    TotalCharges = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

    return pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

# --- Get user input ---
input_df = user_input_features()

# --- Display input summary ---
st.subheader("🔍 Input Summary")
st.write(input_df)

# --- Encoding map ---
label_encodings = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "Yes": 1, "No phone service": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "Yes": 1, "No internet service": 2},
    "OnlineBackup": {"No": 0, "Yes": 1, "No internet service": 2},
    "DeviceProtection": {"No": 0, "Yes": 1, "No internet service": 2},
    "TechSupport": {"No": 0, "Yes": 1, "No internet service": 2},
    "StreamingTV": {"No": 0, "Yes": 1, "No internet service": 2},
    "StreamingMovies": {"No": 0, "Yes": 1, "No internet service": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
}

# --- Encode input safely ---
encoded_input = input_df.copy()
for col in label_encodings:
    if col in encoded_input.columns:
        encoded_input[col] = encoded_input[col].map(label_encodings[col])
        if encoded_input[col].isnull().any():
            st.error(f"🚫 Unexpected input in '{col}' — please use valid options.")
            st.stop()

# --- Validate feature match ---
try:
    encoded_input = encoded_input[features]
except KeyError as e:
    st.error(f"Feature mismatch: {e}")
    st.stop()

# --- Scale input ---
scaled_input = scaler.transform(encoded_input)

# --- Optional Input Warning ---
if input_df["tenure"].iloc[0] > 0 and input_df["TotalCharges"].iloc[0] == 0:
    st.warning("⚠️ Total Charges are 0 despite non-zero tenure. This may affect prediction accuracy.")

# --- Prediction Button ---
if st.button("📊 Predict Churn"):
    probability = float(model.predict(scaled_input).squeeze())
    prediction = "Yes (Churn)" if probability > 0.5 else "No (Retain)"

    st.subheader("📈 Prediction Result")
    st.write(f"**Churn Probability:** {probability:.2%}")
    st.progress(min(int(probability * 100), 100))  # Progress bar

    # Emoji feedback
    if prediction == "Yes (Churn)":
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")

