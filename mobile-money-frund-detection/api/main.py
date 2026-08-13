from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.model_utils import make_payload_row, load_model_artifacts

APP_VERSION = "1.0.0"
MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


class FraudRequest(BaseModel):
    """Request body for a single transaction classification."""

    step: int = Field(..., description="Transaction step/order value used by the notebook pipeline.")
    type: Literal["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(
        ..., description="Transaction type encoded as a categorical one-hot feature."
    )
    amount: float = Field(..., ge=0, description="Transaction amount in the dataset currency.")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender/origin account balance before the transaction.")
    newbalanceOrig: float = Field(..., ge=0, description="Sender/origin balance after the transaction.")
    oldbalanceDest: float = Field(..., ge=0, description="Destination account balance before the transaction.")
    newbalanceDest: float = Field(..., ge=0, description="Destination account balance after the transaction.")
    isFlaggedFraud: int = Field(0, ge=0, le=1, description="Flag derived from the dataset.")

    @field_validator("step")
    @classmethod
    def validate_step(cls, value: int) -> int:
        if value < 0:
            raise ValueError("step must be a non-negative integer")
        return value


class FraudResponse(BaseModel):
    prediction: str
    fraud_probability: float | None = None
    is_fraud: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.preprocessor = None
    try:
        app.state.model, app.state.preprocessor = load_model_artifacts()
        app.state.model_loaded = True
    except Exception:
        app.state.model_loaded = False
    yield


app = FastAPI(
    title="Mobile Money Fraud Detection API",
    description="REST API for deployment of the Mobile Money Fraud Detection machine learning workflow.",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.get("/", tags=["Metadata"])
def root() -> dict:
    """Return API metadata."""
    return {"name": "Mobile Money Fraud Detection API", "version": APP_VERSION, "status": "running"}


@app.get("/health", tags=["Health"])
def health() -> dict:
    """Return readiness and artifact loading status."""
    model_loaded = getattr(app.state, "model_loaded", False) and getattr(app.state, "model", None) is not None
    return {"status": "healthy" if model_loaded else "degraded", "model_loaded": model_loaded}


@app.post("/predict", tags=["Prediction"], response_model=FraudResponse)
def predict(request: FraudRequest, request_obj: Request):
    """Score one transaction and return the fraud classification."""
    if not getattr(request_obj.app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model artifacts are not available for inference.")

    try:
        model = request_obj.app.state.model
        preprocessor = request_obj.app.state.preprocessor
        payload = request.model_dump()
        X = make_payload_row(payload, preprocessor)
        prediction = int(model.predict(X)[0])
        response = {
            "prediction": "Fraud" if prediction == 1 else "Legitimate",
            "is_fraud": bool(prediction == 1),
        }

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            try:
                class_index = list(model.classes_).index(1)
            except (ValueError, AttributeError):
                class_index = 1
            response["fraud_probability"] = float(proba[class_index])
        else:
            response["fraud_probability"] = None

        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed during inference.") from exc


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
