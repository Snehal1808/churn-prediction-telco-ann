import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="SignalStay", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #1E3A8A;'>Signal<span style='color:#14B8A6;'>Lens</span></h1>
    <h4 style='text-align: center; color: white;'>Know Who’s Leaving—Before They Do.</h4>
""", unsafe_allow_html=True)

st.write("Fill out the form below to predict whether a customer will churn.")

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

# --- Helper function for empty default ---
def select_with_prompt(label, options):
    selection = st.selectbox(label, ["-- Select --"] + options)
    return selection if selection != "-- Select --" else None

# --- Input Form ---
def user_input_features():
    st.sidebar.header("🧾 Customer Profile")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    PhoneService = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.header("📋 Customer Services & Charges")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    return pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
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

    if prediction == "Yes (Churn)":
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
