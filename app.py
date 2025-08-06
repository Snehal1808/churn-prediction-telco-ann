import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --- Load model and preprocessors ---
@st.cache_resource
def load_model_file():
    return load_model('model.keras')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

@st.cache_resource
def load_features():
    return joblib.load('features.pkl')

model = load_model_file()
scaler = load_scaler()
features = load_features()

# --- Page Setup ---
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📱 Telco Customer Churn Prediction")

# --- Input Collection ---
st.sidebar.header("🧾 Customer Profile Input")

# Helper for dropdowns
# Sidebar or main column input (choose location)
st.sidebar.title("Customer Information")  # You can move this to st.sidebar if preferred

gender = st.selectbox("Gender", ["-- Select --", "Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", ["-- Select --", "Yes", "No"])
Partner = st.selectbox("Partner", ["-- Select --", "Yes", "No"])
Dependents = st.selectbox("Dependents", ["-- Select --", "Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["-- Select --", "Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["-- Select --", "No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["-- Select --", "DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["-- Select --", "No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["-- Select --", "No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["-- Select --", "No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["-- Select --", "No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["-- Select --", "No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["-- Select --", "No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["-- Select --", "Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["-- Select --", "Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "-- Select --",
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, step=1.0)
tenure = st.slider("Tenure (in months)", 0, 72, 1)

# If any are "--select--", show warning
input_list = [gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
              InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
              StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod]
if any(i == "--select--" for i in input_list):
    st.warning("⚠️ Please fill all fields in the sidebar to proceed.")
    st.stop()

# Map inputs
data = pd.DataFrame([{
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

# Encoding
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
for col, mapping in label_encodings.items():
    data[col] = data[col].map(mapping)

# Reorder and scale
X = data[features]
X_scaled = scaler.transform(X)

# --- PDF Generation ---
def generate_pdf(probability, prediction, tenure, charges, contract):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from io import BytesIO

    styles = getSampleStyleSheet()
    styleN = styles['Normal']

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    report = []

    report.append(Paragraph("Customer Churn Prediction Report", styles['Title']))
    report.append(Spacer(1, 12))
    report.append(Paragraph(f"Prediction: <b>{prediction}</b>", styleN))
    report.append(Paragraph(f"Churn Probability: {probability:.2%}", styleN))
    report.append(Paragraph(f"Tenure: {tenure} months", styleN))
    report.append(Paragraph(f"Monthly Charges: ${charges}", styleN))
    report.append(Paragraph(f"Contract: {contract}", styleN))  # ✅ Here: contract is a string now

    doc.build(report)
    buffer.seek(0)
    return buffer.read()

# Predict
if st.button("📊 Predict Churn"):
    probability = float(model.predict(X_scaled).squeeze())
    prediction = "Yes (Churn)" if probability > 0.5 else "No (Retain)"

    # --- Output 1: Churn Gauge ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if probability > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
        }
    ))
    st.plotly_chart(fig)

    # --- Output 2: Risk Bar ---
    st.subheader("📈 Prediction Result")
    st.write(f"**Churn Probability:** `{probability:.2%}`")
    st.success("✅ Likely to Stay") if prediction == "No (Retain)" else st.error("⚠️ Likely to Churn")

    # --- Output 3: Feature Importance (Static) ---
    st.subheader("🧠 Feature Importance (Generic)")
    importance = pd.Series([0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01],
                           index=["Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod",
                                  "TechSupport", "OnlineSecurity", "StreamingTV", "OnlineBackup",
                                  "DeviceProtection", "Partner", "SeniorCitizen", "PhoneService", "Dependents"])
    importance.plot(kind='barh')
    st.pyplot(plt.gcf())
    plt.clf()

    # --- Output 4: Radar Profile ---
    st.subheader("📊 Customer Profile Radar Chart")
    from math import pi
    radar_df = pd.DataFrame({
        'Metric': ['tenure', 'MonthlyCharges', 'TotalCharges'],
        'Value': [tenure/72, MonthlyCharges/150, TotalCharges/10000]
    })
    categories = list(radar_df['Metric'])
    values = radar_df['Value'].tolist()
    values += values[:1]
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    fig_radar = plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)
    ax.set_thetagrids([a * 180/np.pi for a in angles[:-1]], categories)
    st.pyplot(fig_radar)
    plt.clf()

    # --- Output 5: Risk Summary Table ---
    st.subheader("📋 Input Summary with Highlights")
    styled = data.copy()
    styled['RiskFlag'] = ["⚠️" if (tenure < 6 or MonthlyCharges > 100 or Contract == 0) else "✅"]
    st.dataframe(styled)

    # --- Output 6: PDF Report ---
    if st.download_button("📄 Download PDF Report", 
                      file_name="churn_report.pdf",
                      mime="application/pdf",
                      data=lambda: generate_pdf(probability, prediction, tenure, charges, contract)):
        st.success("Report downloaded.")
    
    # --- Output 7: Recommendations ---
    st.subheader("💡 Recommendations")
    if probability > 0.5:
        st.markdown("- Offer discounts for long-term contracts.")
        st.markdown("- Provide personalized customer service follow-ups.")
        st.markdown("- Address high charges with plan reviews.")
    else:
        st.markdown("- Maintain good engagement and proactive support.")
