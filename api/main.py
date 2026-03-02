"""
api/main.py — FastAPI inference service for the churn prediction model.

Loads the trained XGBoost model and scaler from /models/ and exposes
a REST endpoint for real-time churn probability predictions.

Usage:
    uvicorn api.main:app --reload
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts the probability of customer churn using a trained XGBoost model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load artefacts at startup
@app.on_event("startup")
def load_model():
    global model, scaler, feature_names
    try:
        model = joblib.load(MODELS_DIR / "xgboost_churn_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        with open(MODELS_DIR / "feature_names.json") as f:
            feature_names = json.load(f)
        log.info(f"Model loaded. Features: {len(feature_names)}")
    except FileNotFoundError:
        log.warning("Model artefacts not found. Run src/train.py first.")
        model = scaler = feature_names = None


class CustomerInput(BaseModel):
    """Input schema for a single customer prediction request."""

    gender: str = Field(..., example="Male", description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12, description="Months with the company")
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=843.0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    top_risk_factors: list


def preprocess(customer: CustomerInput) -> np.ndarray:
    """Convert a CustomerInput into the feature vector expected by the model."""
    binary_map = {
        "Yes": 1, "No": 0, "Male": 1, "Female": 0,
        "No phone service": 0, "No internet service": 0,
    }

    row = {
        "gender": binary_map.get(customer.gender, 0),
        "SeniorCitizen": customer.SeniorCitizen,
        "Partner": binary_map.get(customer.Partner, 0),
        "Dependents": binary_map.get(customer.Dependents, 0),
        "tenure": customer.tenure,
        "PhoneService": binary_map.get(customer.PhoneService, 0),
        "MultipleLines": binary_map.get(customer.MultipleLines, 0),
        "OnlineSecurity": binary_map.get(customer.OnlineSecurity, 0),
        "OnlineBackup": binary_map.get(customer.OnlineBackup, 0),
        "DeviceProtection": binary_map.get(customer.DeviceProtection, 0),
        "TechSupport": binary_map.get(customer.TechSupport, 0),
        "StreamingTV": binary_map.get(customer.StreamingTV, 0),
        "StreamingMovies": binary_map.get(customer.StreamingMovies, 0),
        "PaperlessBilling": binary_map.get(customer.PaperlessBilling, 0),
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": customer.TotalCharges,
        "InternetService_DSL": 1 if customer.InternetService == "DSL" else 0,
        "InternetService_Fiber optic": 1 if customer.InternetService == "Fiber optic" else 0,
        "InternetService_No": 1 if customer.InternetService == "No" else 0,
        "Contract_Month-to-month": 1 if customer.Contract == "Month-to-month" else 0,
        "Contract_One year": 1 if customer.Contract == "One year" else 0,
        "Contract_Two year": 1 if customer.Contract == "Two year" else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if customer.PaymentMethod == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if customer.PaymentMethod == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if customer.PaymentMethod == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if customer.PaymentMethod == "Mailed check" else 0,
    }

    df_row = {feat: row.get(feat, 0) for feat in feature_names}
    return np.array(list(df_row.values())).reshape(1, -1)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run src/train.py first.")

    X = preprocess(customer)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0][1])
    prediction = prob >= 0.5

    if prob >= 0.75:
        risk = "High"
    elif prob >= 0.45:
        risk = "Medium"
    else:
        risk = "Low"

    # Simple rule-based risk factors for explainability
    risk_factors = []
    if customer.Contract == "Month-to-month":
        risk_factors.append("Month-to-month contract (highest churn risk)")
    if customer.tenure < 12:
        risk_factors.append(f"Low tenure ({customer.tenure} months)")
    if customer.InternetService == "Fiber optic":
        risk_factors.append("Fiber optic service (higher churn rate)")
    if customer.MonthlyCharges > 65:
        risk_factors.append(f"High monthly charges (${customer.MonthlyCharges:.2f})")
    if customer.OnlineSecurity == "No":
        risk_factors.append("No online security add-on")

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        risk_level=risk,
        top_risk_factors=risk_factors[:3],
    )


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
