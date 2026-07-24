# 💳 Credit Card Default Prediction (Credit Risk Assessment)

An end-to-end Machine Learning project that predicts whether a credit card customer is likely to default on the next month's payment. The system helps financial institutions identify high-risk customers and make better credit approval decisions through data-driven risk assessment.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Feature Engineering
- Multiple Machine Learning Models
- Hyperparameter Tuning using GridSearchCV
- 5-Fold Cross Validation
- Credit Risk Prediction
- Interactive Streamlit Web Application
- Model Performance Evaluation

---

## 📌 Problem Statement

Financial institutions face significant losses when customers fail to repay their credit card dues. Identifying high-risk customers before issuing credit can reduce financial losses and improve lending decisions.

This project builds a binary classification model that predicts whether a customer is likely to default based on historical financial and repayment information.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Default of Credit Card Clients
- **Records:** 30,000
- **Target Variable:** `DEFAULT`

### Target Labels

| Value | Meaning |
|-------|---------|
| 0 | No Default |
| 1 | Default |

### Feature Categories

- Customer Demographics
- Credit Limit
- Repayment History
- Bill Statements
- Payment History

---

## 🛠️ Data Preprocessing

The following preprocessing steps were performed before training the models:

- Removed unnecessary `ID` column
- Renamed `PAY_0` to `PAY_1`
- Renamed target column to `DEFAULT`
- Cleaned invalid categorical values
    - **EDUCATION:** `0, 5, 6 → 4 (Others)`
    - **MARRIAGE:** `0 → 3 (Others)`
- Feature Engineering
    - Average Credit Utilization
    - Total Payment Delay
- Train-Test Split (80:20)
- Feature Scaling using **StandardScaler**

---

## 🤖 Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Random Forest Classifier
- SGDClassifier

The final deployed model is **SGDClassifier** after hyperparameter tuning.

---

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** with **5-Fold Cross Validation**.

### Tuned Hyperparameters

- Loss Function
- Regularization Strength (`alpha`)
- Penalty
- L1 Ratio
- Class Weights

### Best Parameters

```python
loss = "log_loss"
alpha = 0.01
penalty = "elasticnet"
l1_ratio = 0.5
class_weight = {0: 1, 1: 2}
```

---

## 📈 Model Evaluation

The model was evaluated using multiple classification metrics:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- ROC-AUC Score

### Performance

| Metric | Score |
|---------|-------|
| Accuracy | **~80%** |
| ROC-AUC | **~0.73** |

---

## 🔄 Project Workflow

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
      │
      ▼
GridSearchCV
      │
      ▼
5-Fold Cross Validation
      │
      ▼
Best Hyperparameters
      │
      ▼
Final SGDClassifier
      │
      ▼
Model Evaluation
      │
      ▼
Streamlit Deployment
```

---

## 🧰 Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 📁 Project Structure

```text
Credit-Card-Default-Prediction/
│
├── data/
│   └── credit_card_default.csv
│
├── notebooks/
│   └── Credit_Risk_Prediction.ipynb
│
├── models/
│   ├── sgd_classifier.pkl
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ▶️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Credit-Card-Default-Prediction.git
```

### Navigate to Project Directory

```bash
cd Credit-Card-Default-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 📱 Streamlit Application

The web application allows users to:

- Enter customer information
- Predict default risk
- View the prediction result instantly
- Support credit risk assessment through an interactive interface

---

## 📚 Key Learnings

Through this project, I gained practical experience in:

- Data Cleaning
- Feature Engineering
- Binary Classification
- Credit Risk Modeling
- Hyperparameter Tuning
- Cross Validation
- Model Evaluation
- Streamlit Deployment
- End-to-End Machine Learning Pipeline

---

## 🚀 Future Enhancements

- XGBoost & LightGBM
- SHAP Explainability
- Threshold Optimization
- Probability Calibration
- Docker Containerization
- Cloud Deployment (AWS/GCP/Azure)
- CI/CD Integration
- Model Monitoring

---

## 👨‍💻 Author

**Atharv Bodake**

B.Tech Electronics & Telecommunication Engineering  
Sardar Patel Institute of Technology (SPIT)

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!
