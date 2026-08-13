# Model Monitoring and Production Drift Strategy

## 1. Data Drift
Data drift is monitored by comparing live transaction distributions with the training-data baseline used to fit the fraud model. The project’s notebook logic identifies important variables such as `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, and the encoded transaction-type indicators. In production, these numeric and categorical distributions should be tracked to detect shifts that may invalidate the assumptions built into the feature column contract.

Examples of monitoring areas include:
- Distribution of transaction amount
- Distribution of transaction types (`CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`)
- Account-balance distributions associated with origin and destination accounts
- Derived features such as `amount_balance_ratio` and `origin_balance_change`

## 2. Prediction Drift
Prediction drift is monitored by sampling prediction results and comparing the rate of fraud classifications with the training-era baseline. If the baseline training fraction is 2% fraud and the production rate climbs to 15%, this may indicate emerging fraud patterns, a change in incoming transaction mix, or a data-quality/labeling problem that should be investigated.

## 3. Model Performance
Once ground-truth fraud labels become available after transaction resolution, the team should calculate:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For a fraud-detection assignment, precision and recall are especially important. Precision describes the fraction of predicted frauds that are correctly identified frauds, while recall defines the fraction of actual frauds that the model successfully catches.

## 4. False Positives
A false positive is a legitimate user transaction that is classified as fraudulent. These cases can cause unnecessary account checks, customer alerts, manual investigations, operational cost, and reputational issues. The monitoring strategy should report the false-positive rate per period and investigate whether the model threshold or feature distribution is causing legitimate transactions to be over-flagged.

## 5. False Negatives
A false negative is a fraudulent transaction that the model has classified as legitimate. This is critical because it represents the missed-fraud area that can produce financial loss. The monitoring dashboard should report ratio and volume of missed fraud events so the model can be improved or retrained.

## 6. API Performance
The API should track:
- Request latency
- Error rate
- Throughput
- API availability
- Model-loading failures

These health checks can be built into logs and a monitoring platform or Sentry-style error pipeline. This API currently exposes `/health` and `GET /` for basic status discovery.

## 7. Retraining
Retraining should be triggered only when evidence suggests that the deployed model is no longer suitable. Examples include: significant data drift, precision decline, recall decline, false-negative increase, new fraud patterns, or changes in transaction behavior across the production population.

It is recommended that retraining is validated using a fresh hold-out/review process before the new model is promoted. A production retraining pipeline should not silently replace a production model without verification.
