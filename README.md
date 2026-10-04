# 💳 Credit Risk Assessment Using XGBoost & SHAP

An end-to-end **Credit Risk Assessment and Explainable AI system** that predicts the probability of loan default using **XGBoost** and explains individual predictions using **SHAP (SHapley Additive exPlanations)**.

The project covers the complete machine learning workflow — from data preprocessing and model training to probability calibration, threshold optimization, explainability, and deployment through a **FastAPI REST API** with a web interface.

---

## 🚀 Project Overview

Credit risk assessment is a binary classification problem where the objective is to determine whether a loan applicant is likely to default.

This project builds a machine learning pipeline that:

- Processes numerical and categorical borrower information
- Handles missing values and categorical features
- Trains an **XGBoost classifier**
- Uses **5-fold cross-validation with probability calibration**
- Optimizes the classification threshold for risk decisions
- Uses **SHAP** to explain model predictions
- Exposes the model through a **FastAPI REST API**
- Provides a web interface for making predictions
- Can be deployed as a web service

### 🎯 Goal

Given information about a loan applicant, the system predicts:

> **Probability of default → Risk classification → Explanation of the prediction**

---

## 🧠 Machine Learning Pipeline

```text
                Credit Risk Dataset
                       │
                       ▼
              Data Preprocessing
                       │
          ┌────────────┴────────────┐
          │                         │
     Numerical                    Categorical
      Features                     Features
          │                         │
     Missing Value              Missing Value
      Imputation                 Handling
          │                         │
          └────────────┬────────────┘
                       ▼
              Feature Transformation
                       │
                       ▼
                XGBoost Classifier
                       │
                       ▼
             5-Fold Calibration
                       │
                       ▼
             Calibrated Probability
                       │
                       ▼
             Optimized Threshold
                       │
                       ▼
              ┌────────────────┐
              │   Risk Result  │
              ├────────────────┤
              │  High Risk     │
              │  Low Risk      │
              └────────────────┘
                       │
                       ▼
                 SHAP Analysis
                       │
                       ▼
            Explainable Prediction
```

---

## 📊 Dataset

The project uses a credit risk dataset containing borrower and loan-related attributes.

### Features

| Feature | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Previous default indicator |
| `cb_person_cred_hist_length` | Length of credit history |

The target represents whether the applicant defaults on the loan.

---

# 🏗️ Model Architecture

## XGBoost

The primary predictive model is **XGBoost**, a gradient boosting algorithm particularly effective for structured/tabular datasets.

The model learns nonlinear relationships between:

- Income
- Loan amount
- Interest rate
- Employment history
- Credit history
- Loan purpose
- Loan grade
- Previous defaults

---

## 🔄 Probability Calibration

Instead of directly using the raw XGBoost probability, the project applies **probability calibration** using `CalibratedClassifierCV`.

The trained model uses **5 calibrated classifiers**, corresponding to 5 folds.

```text
                Training Data
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Fold 1       Fold 2       Fold 3
        │            │            │
       XGB          XGB          XGB
        │            │            │
   Calibration  Calibration  Calibration
        │            │            │
        └────────────┼────────────┘
                     │
                  ...
                     │
                  Fold 5
                     │
                     ▼
             Average Probability
```

This produces a better-calibrated estimate of the probability of default.

---

# 🎯 Threshold Optimization

A default probability of `0.5` is not necessarily the best decision threshold for a credit-risk problem.

Therefore, the project uses an optimized threshold stored in:

```text
deployment/best_threshold.pkl
```

The final decision is:

```python
prediction = int(probability >= threshold)
```

Therefore:

```text
Probability >= threshold
        ↓
     High Risk

Probability < threshold
        ↓
     Low Risk
```

This separates **probability estimation** from the final business decision.

---

# 🔍 Explainable AI with SHAP

A major focus of this project is **model interpretability**.

XGBoost can provide strong predictive performance, but understanding *why* a borrower was classified as high risk is equally important in credit-related applications.

The project uses **SHAP (SHapley Additive exPlanations)** to analyze feature contributions.

SHAP assigns each feature a contribution value that explains how it influences the model prediction.

### Global Explainability

Global SHAP analysis helps answer:

> **Which features are generally most important for predicting credit risk?**

This can be visualized using SHAP summary/feature-importance plots.

### Local Explainability

Local SHAP explanations answer:

> **Why was this particular applicant classified as high or low risk?**

For example:

```text
Applicant
   │
   ├── High loan amount       → increases risk
   ├── High interest rate     → increases risk
   ├── Low income             → increases risk
   ├── Long credit history    → decreases risk
   └── Stable employment      → decreases risk
                              │
                              ▼
                         Final Prediction
```

This makes the model's decision more transparent and interpretable.

---

# 🌐 FastAPI Deployment

The trained model is exposed through a **FastAPI REST API**.

### Endpoint

```text
POST /predict
```

### Example Request

```json
{
  "person_age": 25,
  "person_income": 50000,
  "person_home_ownership": "RENT",
  "person_emp_length": 2,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 10.5,
  "loan_percent_income": 0.2,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

### Example Response

```json
{
  "default_probability": 0.18,
  "default_prediction": 0,
  "threshold": 0.35,
  "Result": "Low Risk"
}
```

The exact probability and threshold depend on the trained model.

---

# 📦 Portable Model Deployment

The original model was a `CalibratedClassifierCV` containing multiple XGBoost models.

For reliable deployment, the model was separated into portable components:

```text
deployment/
│
├── best_threshold.pkl
│
├── calibrator_0.pkl
├── calibrator_1.pkl
├── calibrator_2.pkl
├── calibrator_3.pkl
├── calibrator_4.pkl
│
├── preprocessor_0.pkl
├── preprocessor_1.pkl
├── preprocessor_2.pkl
├── preprocessor_3.pkl
├── preprocessor_4.pkl
│
├── xgb_model_0.json
├── xgb_model_1.json
├── xgb_model_2.json
├── xgb_model_3.json
└── xgb_model_4.json
```

The XGBoost models are stored using XGBoost's native model format rather than relying on the original serialized XGBoost state.

---

# 📁 Project Structure

```text
Credit_Risk-Assesment-Using-SHAP/
│
├── deployment/
│   ├── best_threshold.pkl
│   ├── calibrator_0.pkl
│   ├── calibrator_1.pkl
│   ├── calibrator_2.pkl
│   ├── calibrator_3.pkl
│   ├── calibrator_4.pkl
│   ├── preprocessor_0.pkl
│   ├── preprocessor_1.pkl
│   ├── preprocessor_2.pkl
│   ├── preprocessor_3.pkl
│   ├── preprocessor_4.pkl
│   ├── xgb_model_0.json
│   ├── xgb_model_1.json
│   ├── xgb_model_2.json
│   ├── xgb_model_3.json
│   └── xgb_model_4.json
│
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── credit_risk_dataset.csv
├── credit_risk_model.pkl
├── Credit_Risk.ipynb
├── main.py
├── render.yaml
└── README.md
```

---

# ⚙️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model | XGBoost |
| Explainability | SHAP |
| Model Calibration | CalibratedClassifierCV |
| API | FastAPI |
| API Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Model Serialization | Joblib, XGBoost JSON |
| Deployment | Render |

---

---

# ☁️ Deployment

The project includes:

```text
render.yaml
```

for deployment using Render.

The production architecture is:

```text
                 User
                  │
                  ▼
             Web Interface
                  │
                  ▼
             FastAPI API
                  │
                  ▼
          Preprocessing Pipeline
                  │
                  ▼
        5 XGBoost Models
                  │
                  ▼
       Probability Calibration
                  │
                  ▼
        Average Probability
                  │
                  ▼
        Optimized Threshold
                  │
                  ▼
          Risk Classification
```

---

# 📈 Key Features

### Machine Learning

- XGBoost-based binary classification
- Numerical and categorical preprocessing
- Missing-value handling
- Cross-validated model calibration
- Optimized classification threshold

### Explainability

- SHAP-based feature attribution
- Global model interpretation
- Local prediction explanations
- Feature contribution analysis

### Deployment

- FastAPI REST API
- Interactive Swagger documentation
- Web-based frontend
- Portable model artifacts
- Render deployment configuration

---

# ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

Credit risk predictions should not be used as the sole basis for real-world lending decisions without appropriate:

- Model validation
- Data-quality checks
- Fairness and bias assessment
- Regulatory review
- Human oversight
- Monitoring and retraining procedures

---

## 🚀 Live Demo

The deployed Credit Risk Assessment application is available here:

**[Credit Risk Assessment — Live Demo](https://credit-risk-assesment-using-shap-rzkh.onrender.com/)**

You can use the live application to submit a loan application and receive a predicted credit-risk classification along with the estimated default probability.
# 👨‍💻 Author

**Rujul Ganjoo**
