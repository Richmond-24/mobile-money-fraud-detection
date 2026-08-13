from __future__ import annotations

import joblib

from src.model_utils import FEATURE_COLUMNS, load_model_artifacts, predict_with_model


def predict_from_payload(payload: dict) -> dict:
    """Convenience wrapper used by the API and validation tests."""
    model, _ = load_model_artifacts()
    return predict_with_model(payload, model)
