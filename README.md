# 📊 SignalStay - Customer Churn Prediction System

An AI-powered solution predicting customer churn with **78% accuracy** using artificial neural networks. Built for telecom companies to reduce attrition through proactive interventions.

## ✨ Key Features
- **Predictive Analytics**: 3-layer ANN model (val_acc: 81.5%)
- **Interactive Dashboard**: 
  - Real-time probability gauges
  - Customer profile radar charts
  - Feature importance visualization
- **Automated Reporting**: PDF generation with risk analysis
- **Actionable Insights**: Retention recommendations

## 🚀 Try It Now!  
The app is live on Streamlit—no installation needed!  
👉 **[Launch App](https://signalstay.streamlit.app/)**

## 🛠 Tech Stack
| Component | Technologies |
|-----------|--------------|
| **Machine Learning** | TensorFlow/Keras, Scikit-learn |
| **Frontend** | Streamlit, Plotly, Matplotlib |
| **Data Processing** | Pandas, NumPy, Joblib |
| **Reporting** | ReportLab, PDF generation |

## 📂 Project Structure
```
signalstay/
├── app.py                 # Streamlit application
├── model.keras            # Trained ANN model
├── scaler.pkl             # Feature scaler
├── features.pkl           # Feature list
├── requirements.txt       # Dependencies
├── LICENSE
└── Readme.md
```

## Architecture
```python
model = Sequential([
    Dense(units=16, activation='relu', input_shape=(19,)),
    Dense(units=8, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
```

## 📊 Performance Metrics
| Metric | Value |
|--------|-------|
| Test Accuracy   | 77.54%  |
| Precision    | 0.85 (No Churn) / 0.58 (Churn) |
| ROC AUC     | 0.83  |

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

## 📜 License
Distributed under the MIT License. See [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) for more information.
## ✉️ Contact
Snehal Kumar Subudhi - snehalsubu18@gmail.com
```
