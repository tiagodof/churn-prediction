"""
train.py — Standalone training script for the churn prediction model.

Trains an XGBoost classifier with SMOTE oversampling on the Telco dataset,
evaluates it on a held-out test set, and saves the model artefacts to /models/.

Usage:
    python src/train.py
    python src/train.py --data data/Telco-Customer-Churn.csv --output models/
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_and_prepare(data_path: str) -> tuple:
    log.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

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

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)
    log.info(f"Features: {X.shape[1]}  |  Churn rate: {y.mean():.2%}")
    return X, y


def train(data_path: str, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X, y = load_and_prepare(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log.info("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    log.info(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    log.info("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_resampled, y_resampled,
                                cv=cv, scoring="roc_auc")
    log.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    log.info("Training final model on full training set...")
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    log.info(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")
    log.info(f"Test ROC-AUC: {auc:.4f}")

    joblib.dump(model, output_path / "xgboost_churn_model.pkl")
    joblib.dump(scaler, output_path / "scaler.pkl")
    with open(output_path / "feature_names.json", "w") as f:
        json.dump(X.columns.tolist(), f)

    log.info(f"Artefacts saved to {output_path}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--data", default="data/Telco-Customer-Churn.csv")
    parser.add_argument("--output", default="models/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.output)
