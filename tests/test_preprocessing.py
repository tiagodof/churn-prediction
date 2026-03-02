"""
tests/test_preprocessing.py — Tests for data loading and preparation logic.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def make_sample_df(n=100):
    """Generate a minimal synthetic dataframe that mirrors the Telco schema."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": rng.integers(0, 72, n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], n),
        "MonthlyCharges": rng.uniform(20, 120, n).round(2),
        "TotalCharges": [str(round(v, 2)) if i > 0 else " "
                         for i, v in enumerate(rng.uniform(20, 8000, n))],
        "Churn": rng.choice(["Yes", "No"], n),
    })
    return df


def prepare(df):
    """Replicate the preparation logic from train.py."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df = df.drop(columns=["customerID"])

    binary_map = {
        "Yes": 1, "No": 0, "Male": 1, "Female": 0,
        "No phone service": 0, "No internet service": 0,
    }
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in binary_cols:
        df[col] = df[col].map(binary_map).fillna(df[col])

    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"])
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def test_total_charges_imputation():
    df = make_sample_df(50)
    result = prepare(df)
    assert result["TotalCharges"].isna().sum() == 0


def test_no_object_columns_after_preparation():
    df = make_sample_df(50)
    result = prepare(df)
    object_cols = result.select_dtypes(include="object").columns.tolist()
    assert object_cols == [], f"Object columns remain: {object_cols}"


def test_churn_binary_encoding():
    df = make_sample_df(50)
    result = prepare(df)
    assert set(result["Churn"].unique()).issubset({0, 1})


def test_one_hot_columns_present():
    df = make_sample_df(50)
    result = prepare(df)
    expected_cols = [
        "InternetService_DSL", "InternetService_Fiber optic",
        "Contract_Month-to-month", "Contract_Two year",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_no_customer_id_after_preparation():
    df = make_sample_df(50)
    result = prepare(df)
    assert "customerID" not in result.columns


def test_row_count_preserved():
    df = make_sample_df(100)
    result = prepare(df)
    assert len(result) == 100
