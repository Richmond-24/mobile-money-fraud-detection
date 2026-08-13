# Mobile Money Fraud Detection and Prediction

## 1. Project Overview
This repository implements a machine learning workflow for mobile money fraud detection. It follows the notebook-driven exploratory analysis in the project and exposes a trained logistic-regression fraud classifier through a FastAPI REST service.

## 2. Problem Statement
Mobile money transactions are exposed to fraud through account compromise, unauthorized transfers, and suspicious transaction patterns. A fraud-detection system can reduce risk if it identifies likely fraudulent behavior early enough to stop or investigate the transaction.

## 3. Objective
The objective is to train a supervised binary fraud classifier using transaction-level features, evaluate the learned model, persist the artifact, and serve prediction over HTTP through an API.

## 4. Machine Learning Pipeline
Data
↓
Data Cleaning
↓
Feature Engineering
↓
Preprocessing
↓
Train/Test Split
↓
Model Training
↓
Evaluation
↓
Saved Model
↓
REST API
↓
Prediction

## 5. Dataset
The notebook workflow loads the public PaySim synthetic mobile money dataset from the repository’s historical dataset location. The notebook refers to a CSV file named `PS_20174392719_1491204439457_log.csv`, loaded through a relative file path and sampled to 100000 rows in the code cells. The raw data file is not present in the current repository snapshot, so the training artifact is documented but the original CSV must be restored for full reproducibility.

## 6. Features
The final preprocessing contract builds the following model input fields:

- `step`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `isFlaggedFraud`
- `hour`
- `day`
- `origin_balance_change`
- `amount_balance_ratio`
- `destination_zero_balance`
- `large_transaction`
- `type_CASH_OUT`
- `type_DEBIT`
- `type_PAYMENT`
- `type_TRANSFER`

The original transaction type is converted via one-hot encoding. The notebook shows the categorical `type` columns being transformed into one-hot columns through `pd.get_dummies(..., columns=["type"], drop_first=True)`.

## 7. Model
The notebook training cells show the selected final classifier workflow as a `LogisticRegression` model configured with `class_weight="balanced"` and `max_iter=1000`. The existing project does not currently persist the notebook-trained model artifact, so the repository now contains a serialization contract that expects a joblib model artifact.

## 8. Evaluation
The project notebooks show that the fraud pipeline was centered on a train/test split using `test_size=0.2`, `random_state=42`, and `stratify=y`. The notebook output shows model prediction outputs with fraud probabilities but the exact evaluation metrics are not stored in the repository source. The current repository therefore documents the training code and model usage, but metrics need to be generated from the original dataset before claiming specific published values.

## 9. REST API
The API is implemented with FastAPI and provides the endpoints `/`, `/health`, and `/predict`.

## 10. Endpoints
- `GET /`
- `GET /health`
- `POST /predict`

## 11. Example API Request
Example request body built from the notebook feature contract:

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

## 12. Example API Response

```json
{
  "prediction": "Fraud",
  "fraud_probability": 0.94,
  "is_fraud": true
}
```

## 13. Running Locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

The Swagger documentation is available at <http://localhost:8000/docs>.

## 14. Testing
The test suite is assembled under the `tests` directory and can be executed with:

```bash
pytest
```

## 15. Swagger Documentation
The API service exposes OpenAPI/Swagger at <http://localhost:8000/docs> and ReDoc at <http://localhost:8000/redoc>.

## 16. Docker
The repository includes a simple Dockerfile and docker-compose definition so the API can be containerized.

```bash
docker build -t mobile-money-fraud-api .
docker run -p 8000:8000 mobile-money-fraud-api
```

## 17. Production Monitoring
Monitoring must be used to identify data drift, prediction drift, precision/recall deterioration, false positives, false negatives, API latency, error rates, and retraining needs. The monitoring guidance is described in the dedicated monitoring guide.

## 18. Project Structure
```text
mobile-money-fraud-detection/
├── api/
├── data/
├── models/
├── monitoring/
├── notebooks/
├── src/
└── tests/
```

## 19. Limitations
This repository is built as a small interview/academic assignment sample. The deployed implementation is intentionally straightforward and does not include full production authentication, permanent dashboards, model registry workflows, or secure enterprise data ingestion.

## 20. Future Improvements
Future improvements described by the work include richer feature engineering, more robust training/evaluation reporting, better monitoring, retraining automation, and cloud/container production hardening.

## 21. Author/Team
The repository does not contain a formal team metadata section. This project is presented as the Mobile Money Fraud Detection assignment.
