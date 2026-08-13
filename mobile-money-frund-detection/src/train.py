from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.preprocessing import encode_transaction_type, engineer_features, selected_training_columns

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
PREPROCESSOR_PATH = Path(__file__).resolve().parents[1] / "models" / "preprocessor.joblib"
DATASET_CANDIDATES = [
    Path(__file__).resolve().parents[1] / "data" / "raw" / "PS_20174392719_1491204439457_log.csv",
    Path(__file__).resolve().parents[1] / "data" / "PS_20174392719_1491204439457_log.csv",
]


def load_training_frame():
    for path in DATASET_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path, nrows=100000)
            return df

    rng = np.random.default_rng(42)
    n_rows = 2000
    df = pd.DataFrame(
        {
            "step": rng.integers(1, 100, size=n_rows),
            "type": rng.choice(["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], size=n_rows),
            "amount": np.abs(rng.normal(100, 800, size=n_rows)),
            "nameOrig": [f"orig_{i}" for i in range(n_rows)],
            "nameDest": [f"dest_{i}" for i in range(n_rows)],
            "oldbalanceOrg": np.abs(rng.normal(5000, 5000, size=n_rows)),
            "newbalanceOrig": np.abs(rng.normal(5000, 5000, size=n_rows)),
            "oldbalanceDest": np.abs(rng.normal(5000, 5000, size=n_rows)),
            "newbalanceDest": np.abs(rng.normal(5000, 5000, size=n_rows)),
            "isFraud": rng.binomial(1, 0.03, size=n_rows),
            "isFlaggedFraud": rng.binomial(1, 0.001, size=n_rows),
        }
    )
    return df


def train_model():
    raw_df = load_training_frame()
    engineered = engineer_features(raw_df)
    encoded = encode_transaction_type(engineered)
    X = selected_training_columns(encoded)
    y = raw_df["isFraud"]
    threshold = float(engineered["amount"].quantile(0.95))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    preprocessor_metadata = {
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "large_transaction_threshold": threshold,
        "encoding": {
            "type_columns": ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"],
        },
    }
    joblib.dump(preprocessor_metadata, PREPROCESSOR_PATH)
    return model, metrics


if __name__ == "__main__":
    train_model()
