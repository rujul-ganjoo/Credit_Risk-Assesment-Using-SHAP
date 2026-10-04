from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb

from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles


# Store all model components
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    models = []

    # Load all 5 calibrated models
    for i in range(5):

        # Load preprocessing pipeline
        preprocessor = joblib.load(
            f"deployment/preprocessor_{i}.pkl"
        )

        # Load XGBoost model
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(
            f"deployment/xgb_model_{i}.json"
        )

        # Load sigmoid calibration parameters
        calibration = joblib.load(
            f"deployment/calibrator_{i}.pkl"
        )

        models.append({
            "preprocessor": preprocessor,
            "model": xgb_model,
            "a": calibration["a"],
            "b": calibration["b"]
        })

    # Load optimized threshold
    threshold = joblib.load(
        "deployment/best_threshold.pkl"
    )

    ml_model["models"] = models
    ml_model["threshold"] = threshold

    print("====================================")
    print("All 5 models loaded successfully!")
    print("Threshold:", threshold)
    print("====================================")

    yield

    ml_model.clear()


app = FastAPI(lifespan=lifespan)


# Request format
class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.post("/predict")
def predict(data: LoanApplication):

    # Convert input into DataFrame
    input_df = pd.DataFrame([data.dict()])

    probabilities = []

    # Run all 5 calibrated models
    for model_info in ml_model["models"]:

        # Apply preprocessing
        X_processed = model_info["preprocessor"].transform(
            input_df
        )

        # Get raw XGBoost probability
        raw_probability = model_info["model"].predict_proba(
            X_processed
        )[0, 1]

        # Apply sigmoid calibration
        a = model_info["a"]
        b = model_info["b"]

        calibrated_probability = 1 / (
            1 + np.exp(a * raw_probability + b)
        )

        probabilities.append(calibrated_probability)

    # Average probabilities from all 5 folds
    probability = float(np.mean(probabilities))

    # Apply optimized threshold
    prediction = int(
        probability >= ml_model["threshold"]
    )

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": ml_model["threshold"],
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }


# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")