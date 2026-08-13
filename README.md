# Mobile Money Fraud Detection — API

This repository exposes a trained mobile-money fraud detection model via a FastAPI service.

Quick start

- Activate the project's virtual environment:

```bash
source .venv/bin/activate
```

- Install API dependencies:

```bash
pip install -r requirements.txt
```

- Start the API (from repository root):

```bash
python -m uvicorn app.main:app --reload
```

API URLs

- Local API root: http://127.0.0.1:8000/
- Swagger UI: http://127.0.0.1:8000/docs

Available endpoints

- GET `/` — basic metadata
- GET `/health` — readiness and model loaded status
- POST `/predict` — score a single transaction

Example `POST /predict` JSON (fields must match the training features exactly):

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
  "prediction": 0,
  "result": "Legitimate",
  "fraud_probability": 0.02
}
```

How the model is loaded

- The FastAPI app loads serialized artifacts from `mobile-money-frund-detection/models/` on startup. The artifacts are `model.joblib` (trained classifier) and `preprocessor.joblib` (metadata about feature columns and thresholds).
- The API delegates feature engineering and shape validation to the same helper functions used during training, so the preprocessing is never duplicated.

Preprocessing

- Feature engineering and encoding are implemented in `mobile-money-frund-detection/src/preprocessing.py` and `mobile-money-frund-detection/src/model_utils.py`.
- The API uses `predict_with_model` from `src.model_utils` to apply the same transformations and preserve feature order.

Monitoring (brief)

- Log prediction inputs and outputs (not implemented here). Monitor prediction distributions, input feature drift, and model confidence (e.g. distribution of `fraud_probability`).
- Track model performance on a rolling window of labeled feedback and alert when metrics degrade beyond thresholds.
hint: Updates were rejected because the remote contains work that you do not
hint: have locally. This is usually caused by another repository pushing to
hint: the same ref. If you want to integrate the remote changes, use
hint: 'git pull' before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
(.venv) (base) richmond@Richy-johnson:~/Downloads/Mobile-Money-Fraud-Prediction$ git pull origin main --rebase
remote: Enumerating objects: 4, done.
remote: Counting objects: 100% (4/4), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
Unpacking objects: 100% (3/3), 5.01 KiB | 2.50 MiB/s, done.
From https://github.com/Richmond-24/mobile-money-fraud-detection
 * branch            main       -> FETCH_HEAD
   ad68827..a0225c0  main       -> origin/main
Auto-merging README.md
CONFLICT (add/add): Merge conflict in README.md
error: could not apply db026f4... Add FastAPI fraud detection API
hint: Resolve all conflicts manually, mark them as resolved with
hint: "git add/rm <conflicted_files>", then run "git rebase --continue".
hint: You can instead skip this commit: run "git rebase --skip".
hint: To abort and get back to the state before "git rebase", run "git rebase --abort".
hint: Disable this message with "git config set advice.mergeConflict false"
Could not apply db026f4... Add FastAPI fraud detection API
(.venv) (base) richmond@Richy-johnson:~/Downloads/Mobile-Money-Fraud-Prediction$ 