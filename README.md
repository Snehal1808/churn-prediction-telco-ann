<img width="1920" height="1080" alt="Screenshot (261)" src="https://github.com/user-attachments/assets/1f5b8b98-0227-41b7-b90a-3f41061af6b0" /># 📊 SignalStay - Customer Churn Prediction System

An AI-powered solution predicting customer churn with **81.5% accuracy** using artificial neural networks. Built for telecom companies to reduce attrition through proactive interventions.

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
<img width="1920" height="1080" alt="Screenshot (261)" src="https://github.com/user-attachments/assets/b384c41f-aef9-4a5f-8166-a751f641abb1" />
<img width="1920" height="1080" alt="Screenshot (262)" src="https://github.com/user-attachments/assets/96ad1310-9e28-4339-a473-dae91b24b2c2" />
<img width="1920" height="1080" alt="Screenshot (263)" src="https://github.com/user-attachments/assets/70d0c673-9e4e-4d26-a9cc-3bb9653e312f" />
<img width="1920" height="1080" alt="Screenshot (264)" src="https://github.com/user-attachments/assets/e13791c5-8ab4-4692-aaf8-e4c1f2f5c09a" />
<img width="1920" height="1080" alt="Screenshot (265)" src="https://github.com/user-attachments/assets/d77739f1-0af6-4154-83fb-2709aee76582" />
<img width="1920" height="1080" alt="Screenshot (266)" src="https://github.com/user-attachments/assets/5e2cdd3e-5711-42f5-93ca-62f26f256181" />
<img width="1920" height="1080" alt="Screenshot (267)" src="https://github.com/user-attachments/assets/a201668a-96db-48fa-9265-53ecf87078aa" />
<img width="1920" height="1080" alt="Screenshot (268)" src="https://github.com/user-attachments/assets/65226c42-6c9f-4831-a503-1341a60bcb1b" />


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
