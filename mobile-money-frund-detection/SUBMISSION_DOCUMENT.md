# Mobile Money Fraud Detection — Technical Assignment

## 1. Executive Summary
This project implements a machine learning fraud-detection system for mobile money transaction data. The solution performs feature engineering and binary classification using the existing notebook workflow, then exposes the model through a REST API for automated inference.

## 2. Assignment Requirement
The assignment requirement is to build a simple model pipeline for a public dataset, including preprocessing, training, and evaluation, then deploy the trained model as a REST API with a short production-monitoring analysis.

## 3. Solution Architecture
Dataset
→ Preprocessing
→ Training
→ Evaluation
→ Model Serialization
→ FastAPI
→ Prediction
→ Monitoring

## 4. Data Preprocessing
The notebook-based preprocessing transforms the raw transaction frame into engineered columns such as hour, day, origin balance change, amount balance ratio, destination-zero-balance indicator, and an amount-threshold indicator. It also binarizes the transaction `type` column by converting it to one-hot encodings such as `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, and `type_TRANSFER` and removes non-model columns such as `nameOrig` and `nameDest`.

## 5. Model Training
The training notebook trains a `LogisticRegression` classifier using `class_weight="balanced"` and `max_iter=1000`. The train/test split uses `test_size=0.2`, `random_state=42`, and `stratify=y`.

## 6. Model Evaluation
The project’s notebook text and generated code show that predictions and probability outputs exist, but the repository does not contain a final saved evaluation report with the exact metrics. The API therefore documents the training contract and the probability-supporting logistic regression; however, any accuracy, precision, recall, F1, and confusion-matrix artifacts need to be re-extracted from a full dataset run.

## 7. REST API Implementation
The service is delivered using FastAPI and Pydantic. The API validates a single request body, maps it to the notebook feature engineering logic, loads the serialized estimator at startup, scores it using the same binary fraud target contract, and returns the classification with a probability when available.

The API includes:
- `/predict`
- `/health`
- `/docs`
- `/redoc`

## 8. Example Prediction
Example request:

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

Example response:

```json
{
  "prediction": "Fraud",
  "fraud_probability": 0.94,
  "is_fraud": true
}
```

## 9. Production Monitoring Strategy
Production monitoring should track data drift, prediction drift, precision, recall, F1-score, false positives, false negatives, API latency, error rate, throughput, service availability, and model-loading failures. When there are significant changes in transaction volumes or feature distributions, the model should be retrained and validated rather than automatically replaced.

## 10. Security Considerations
The API validates the request schema and avoids logging sensitive transaction detail. A production deployment would require authentication and authorization, HTTPS, secure deployment secrets, and protected storage for model and dataset assets.

## 11. Testing
The repository includes API and model tests using pytest. They cover the public health endpoints, a valid inference body, missing and malformed payloads, and the model artifact-loading contract.

## 12. Deployment
The project is designed to run locally through `uvicorn api.main:app --reload` and through a container image built by Docker or docker-compose.

## 13. Limitations
This implementation depends on the original public dataset file being restored because the current snapshot omits the raw CSV. It also uses the notebook to define the training contract and therefore does not currently publish evaluation artifacts that were absent from the repository.

## 14. Future Improvements
Potential future improvements include consistent dataset packaging, more robust metrics reporting, a training reproducibility script, drift-alerting dashboards, and an authentication layer before production exposure.

## 15. Conclusion
The repository now provides a deployable API scaffold over the existing fraud-detection pipeline, preserving the notebook-derived training and feature contract. It satisfies the assignment’s core delivery points of preprocessing, training/evaluation, serialization, REST API deployment, and production monitoring documentation.
