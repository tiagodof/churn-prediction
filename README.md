# Customer Churn Prediction Pipeline

An end-to-end Machine Learning pipeline for predicting customer churn in the telecommunications industry. This project covers the entire lifecycle from Exploratory Data Analysis (EDA) and class imbalance handling, to model interpretability using SHAP and deployment via a FastAPI inference service.

## Project Architecture

The project follows the CRISP-DM methodology and is structured into clear, reproducible components:

1. **Data Understanding & EDA:** Comprehensive analysis of the IBM Telco dataset to identify key churn drivers.
2. **Modelling & Evaluation:** Comparison of class imbalance strategies (SMOTE vs `scale_pos_weight`) using XGBoost, Random Forest, and Logistic Regression.
3. **Interpretability:** Global and local feature importance analysis using SHAP (SHapley Additive exPlanations).
4. **Deployment:** A standalone training script and a RESTful FastAPI service for real-time predictions.

## Key Findings & Business Impact

The final model (XGBoost + SMOTE) achieved an **ROC-AUC of 0.84** and a **Recall of 0.84** for the minority class (churners). 

SHAP analysis revealed the top risk factors:
- **Contract Type:** Month-to-month contracts are the strongest predictor of churn.
- **Tenure:** Customers in their first 12 months are at the highest risk.
- **Internet Service:** Fiber optic customers show higher churn rates despite higher costs, indicating potential service quality issues.

**Business Recommendation:** Targeting the top 20% highest-risk customers with a retention offer (e.g., 15% discount to switch to an annual contract) could preserve significant monthly recurring revenue.

## Repository Structure

```
├── api/
│   └── main.py                 # FastAPI inference service
├── data/
│   └── Telco-Customer-Churn.csv # Raw dataset
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_modelling_and_evaluation.ipynb
├── src/
│   └── train.py                # Standalone model training script
├── tests/
│   ├── test_api.py             # API endpoint tests
│   └── test_preprocessing.py   # Data pipeline tests
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
This will process the data, train the XGBoost model with SMOTE, and save the artefacts to the `models/` directory.
```bash
python src/train.py
```

### 3. Run the API
Start the FastAPI server to serve predictions.
```bash
uvicorn api.main:app --reload
```
The API documentation (Swagger UI) will be available at `http://127.0.0.1:8000/docs`.

### 4. Run Tests
```bash
pytest tests/
```

## Tech Stack

- **Data Processing:** Pandas, NumPy, Scikit-learn, Imbalanced-learn
- **Modelling:** XGBoost, Scikit-learn
- **Interpretability:** SHAP
- **API:** FastAPI, Uvicorn, Pydantic
- **Testing:** Pytest
