from __future__ import annotations

import pandas as pd

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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the feature set seen in the notebook cells."""
    working = df.copy()
    working["hour"] = working["step"] % 24
    working["day"] = working["step"] // 24
    working["origin_balance_change"] = working["oldbalanceOrg"] - working["newbalanceOrig"]
    working["amount_balance_ratio"] = working["amount"] / (working["oldbalanceOrg"] + 1)
    working["destination_zero_balance"] = (working["oldbalanceDest"] == 0).astype(int)
    working["large_transaction"] = (working["amount"] > working["amount"].quantile(0.95)).astype(int)
    return working


def encode_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same one-hot encoding used in the notebook."""
    encoded = pd.get_dummies(df, columns=["type"], drop_first=True)
    for column in ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        if column not in encoded.columns:
            encoded[column] = 0
    return encoded


def selected_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric feature matrix used by the model contract."""
    working = df.copy()
    working = working.drop(columns=["nameOrig", "nameDest"], errors="ignore")
    missing = [column for column in FEATURE_COLUMNS if column not in working.columns]
    for column in missing:
        working[column] = 0
    return working[FEATURE_COLUMNS]
