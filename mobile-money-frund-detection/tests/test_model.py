import importlib

from src.model_utils import FEATURE_COLUMNS, load_model_artifacts


def test_model_artifact_loading():
    model, preprocessor = load_model_artifacts()
    assert model is not None
    assert preprocessor is not None


def test_feature_contract_is_documented():
    assert "step" in FEATURE_COLUMNS
    assert "amount" in FEATURE_COLUMNS
    assert "oldbalanceOrg" in FEATURE_COLUMNS
    assert "newbalanceOrig" in FEATURE_COLUMNS
    assert "oldbalanceDest" in FEATURE_COLUMNS
    assert "newbalanceDest" in FEATURE_COLUMNS
    assert "isFlaggedFraud" in FEATURE_COLUMNS
    assert "type_CASH_OUT" in FEATURE_COLUMNS
    assert "type_DEBIT" in FEATURE_COLUMNS
    assert "type_PAYMENT" in FEATURE_COLUMNS
    assert "type_TRANSFER" in FEATURE_COLUMNS
