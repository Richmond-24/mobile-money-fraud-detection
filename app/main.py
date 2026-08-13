from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ensure the mobile-money project package is importable from this app directory.
ROOT = Path(__file__).resolve().parents[1]
SRC_PACKAGE_ROOT = ROOT / "mobile-money-frund-detection"
sys.path.insert(0, str(SRC_PACKAGE_ROOT))

from src.model_utils import load_model_artifacts, predict_with_model  # type: ignore

APP_VERSION = "1.0.0"


class FraudRequest(BaseModel):
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
    prediction: int
    result: str
    fraud_probability: float | None = None


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
    description="Detect potentially fraudulent mobile money transactions using the trained model.",
    version=APP_VERSION,
    lifespan=lifespan,
)

# Allow local frontends to call the API; keep CORS minimal and explicit.
origins = ["http://localhost", "http://127.0.0.1:8000", "http://localhost:8000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/", tags=["Metadata"])
def root() -> dict:
    return {"message": "Mobile Money Fraud Detection API is running", "version": APP_VERSION}


@app.get("/health", tags=["Health"]) 
def health() -> dict:
    model_loaded = getattr(app.state, "model_loaded", False) and getattr(app.state, "model", None) is not None
    return {"status": "healthy" if model_loaded else "degraded", "model_loaded": model_loaded}


@app.post("/predict", tags=["Prediction"], response_model=FraudResponse)
def predict(request: FraudRequest, request_obj: Request):
    if not getattr(request_obj.app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model artifacts are not available for inference.")

    try:
        payload = request.model_dump()
        model = request_obj.app.state.model
        preprocessor = request_obj.app.state.preprocessor
        # Use the canonical preprocessing + inference implemented in the training package.
        response = predict_with_model(payload, model, preprocessor)

        return {
            "prediction": int(response.get("prediction", 0)),
            "result": response.get("prediction_label", "Fraud" if int(response.get("prediction", 0)) == 1 else "Legitimate"),
            "fraud_probability": response.get("fraud_probability"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed during inference.")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})