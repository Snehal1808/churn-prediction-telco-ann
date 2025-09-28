# 📊 SignalStay - Customer Churn Prediction System

An AI-powered solution predicting customer churn with **81.5% accuracy** using artificial neural networks. Built for telecom companies to reduce attrition through proactive interventions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://signalstay.streamlit.app/) 

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

## 🖼️ Screenshots
<img width="1901" height="882" alt="Screenshot (261)" src="https://github.com/user-attachments/assets/7633d59a-de32-44e1-aecd-42a26aefbd51" />
<img width="1909" height="894" alt="Screenshot (262)" src="https://github.com/user-attachments/assets/6fd61b19-52fa-4464-9b37-b4aa31a84906" />
<img width="1899" height="881" alt="Screenshot (263)" src="https://github.com/user-attachments/assets/683dca63-7185-47d5-9a3f-de3d23dcba06" />
<img width="1920" height="1080" alt="Screenshot (264)" src="https://github.com/user-attachments/assets/2b07f905-1eff-47e2-934b-7cc76248462f" />
<img width="1899" height="889" alt="Screenshot (265)" src="https://github.com/user-attachments/assets/28852fbd-0980-402e-b035-1916feb64a2f" />
<img width="1897" height="877" alt="Screenshot (266)" src="https://github.com/user-attachments/assets/3ab6a4ed-a8a4-4c06-a9b9-02dfea03c645" />
<img width="1909" height="891" alt="Screenshot (267)" src="https://github.com/user-attachments/assets/7f18e952-848d-40c1-bb47-65b1e8d437b1" />
<img width="1898" height="879" alt="Screenshot (268)" src="https://github.com/user-attachments/assets/9e4aa85c-ae5d-4078-804d-913ebf33b9cc" />

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
Snehal Kumar Subudhi - snehalsubudhi.tech@gmail.com
