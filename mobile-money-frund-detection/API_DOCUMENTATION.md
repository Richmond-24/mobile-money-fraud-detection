# API Documentation

This service exposes a fraud prediction API for the Mobile Money Fraud Detection pipeline.

## Endpoints

### GET /
Returns metadata about the service.

### GET /health
Returns service health and the model-loading status.

### POST /predict
Submits a transaction payload that follows the notebook-derived feature contract.

Example:

```json
{
  "step": 1,
  "type": "CASH_OUT",
  "amount": 5000.0,
  "oldbalanceOrg": 10000.0,
  "newbalanceOrig": 7000.0,
  "oldbalanceDest": 2000.0,
  "newbalanceDest": 7000.0,
  "isFlaggedFraud": 0
}
```

The body is validated by Pydantic before it reaches the model.

## Swagger
The API automatically exposes OpenAPI docs at:

- <http://localhost:8000/docs>
- <http://localhost:8000/redoc>
