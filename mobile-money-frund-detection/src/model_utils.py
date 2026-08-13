from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "hour",
    "day",
    "origin_balance_change",
    "amount_balance_ratio",
    "destination_zero_balance",
    "large_transaction",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


def load_model_artifacts():
    """Load the API model and metadata artifact for inference."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def make_payload_row(payload: dict, preprocessor: dict | None = None) -> pd.DataFrame:
    """Apply the same feature engineering steps used in the notebook."""
    type_value = payload.get("type")
    transaction_types = {
        "CASH_OUT": "type_CASH_OUT",
        "DEBIT": "type_DEBIT",
        "PAYMENT": "type_PAYMENT",
        "TRANSFER": "type_TRANSFER",
    }

    if type_value not in transaction_types:
        raise ValueError("The transaction type is invalid for this model contract.")

    step = float(payload["step"])
    amount = float(payload["amount"])
    oldbalance_org = float(payload["oldbalanceOrg"])
    newbalance_orig = float(payload["newbalanceOrig"])
    oldbalance_dest = float(payload["oldbalanceDest"])
    newbalance_dest = float(payload["newbalanceDest"])
    is_flagged_fraud = int(payload.get("isFlaggedFraud", 0))

    threshold = 0.0
    if isinstance(preprocessor, dict):
        threshold = float(preprocessor.get("large_transaction_threshold", 0.0) or 0.0)

    row = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "isFlaggedFraud": is_flagged_fraud,
        "hour": step % 24,
        "day": step // 24,
        "origin_balance_change": oldbalance_org - newbalance_orig,
        "amount_balance_ratio": amount / (oldbalance_org + 1.0),
        "destination_zero_balance": int(oldbalance_dest == 0),
        "large_transaction": int(amount > threshold),
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 0,
    }

    row[transaction_types[type_value]] = 1
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_with_model(raw_payload: dict, model, preprocessor=None):
    """Prepare data and run a single inference request."""
    X = make_payload_row(raw_payload, preprocessor)
    prediction = model.predict(X)[0]
    response = {
        "prediction": int(prediction),
        "is_fraud": bool(int(prediction) == 1),
        "prediction_label": "Fraud" if int(prediction) == 1 else "Legitimate",
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        # In sklearn a binary class order is available on classes_.
        try:
            class_index = list(model.classes_).index(1)
        except (ValueError, AttributeError):
            class_index = 1
        response["fraud_probability"] = float(proba[class_index])
    return response
