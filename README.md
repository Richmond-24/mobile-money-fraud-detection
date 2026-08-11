 Mobile Money Fraud Detection

A machine learning project for detecting potentially fraudulent mobile money transactions using transaction-level behavioral and financial features.

  project Overview

Mobile money services have become an important part of digital financial transactions, especially in developing economies. However, the increasing volume of mobile money transactions also creates opportunities for fraudulent activities.

This project develops a machine learning-based fraud detection system that analyzes transaction characteristics and predicts whether a transaction is potentially fraudulent.

The project covers the complete machine learning workflow, including:

* Data understanding and exploration
* Data preprocessing
* Feature engineering
* Exploratory data analysis
* Model training
* Model evaluation
* Fraud prediction
* Model deployment/application development

Project Objectives

The main objectives of this project are to:

1. Analyze mobile money transaction data.
2. Identify patterns associated with fraudulent transactions.
3. Engineer useful features for fraud detection.
4. Train machine learning models to classify transactions.
5. Evaluate model performance using appropriate classification metrics.
6. Develop a prediction pipeline that can be used to identify potentially fraudulent transactions.
7. Provide a foundation for deploying the fraud detection system in a real-world application.

 Dataset

The project uses the **PaySim mobile money transaction dataset**, originally created from a simulation of mobile money transactions.

The dataset contains transaction information such as:

* Transaction type
* Transaction amount
* Sender balance before transaction
* Sender balance after transaction
* Receiver balance before transaction
* Receiver balance after transaction
* Transaction time/step
* Fraud label
* Other transaction-related attributes

### Dataset Source

The dataset is available on Kaggle:

**PaySim – Mobile Money Fraud Detection Dataset**

https://www.kaggle.com/datasets/ealaxi/paysim1

Important Dataset Note

The original CSV dataset is approximately 470 MB and is **not included in this GitHub repository** because GitHub has a 100 MB file-size limit for individual files.

To reproduce the project:

1. Download the dataset from Kaggle.
2. Extract the CSV file.
3. Place it in:

```text
data/raw/
```

The expected dataset filename is:

```text
PS_20174392719_1491204439457_log.csv
```

The dataset is intentionally excluded from Git using `.gitignore`.

## Project Structure

```text
mobile-money-frund-detection/
│
├── app/
│   └── Application files
│
├── data/
│   └── raw/
│       └── PS_20174392719_1491204439457_log.csv
│
├── models/
│   └── Trained model files
│
├── notebooks/
│   ├── data_understanding.ipynb
│   └── feature_engineering.ipynb
│
├── reports/
│   └── Project reports and evaluation results
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   └── train.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

> Note: The dataset folder shown above is for the local project setup. The large CSV file is ignored by Git and is not stored in this repository.

 Technologies Used

The project uses Python and several machine learning and data science libraries.

Programming Language

* Python 3

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* Joblib

Additional dependencies are listed in:

```text
requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Richmond-24/mobile-money-fraud-detection.git
```

Move into the project directory:

```bash
cd mobile-money-fraud-detection
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the PaySim dataset from Kaggle:

```text
https://www.kaggle.com/datasets/ealaxi/paysim1
```

Place the CSV file inside:

```text
data/raw/
```

The expected path is:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

## Data Understanding

The first stage of the project focuses on understanding the structure and characteristics of the dataset.

The analysis includes:

* Dataset dimensions
* Data types
* Missing values
* Duplicate records
* Transaction distributions
* Fraud distribution
* Transaction types
* Transaction amounts
* Balance changes
* Relationships between transaction variables

The data understanding notebook is available at:

```text
notebooks/data_understanding.ipynb
```

## Feature Engineering

Feature engineering is an important part of the fraud detection process.

The project derives additional transaction-level features from the original variables to help the machine learning model identify suspicious behavior.

Examples include:

* Balance differences
* Transaction amount relationships
* Sender balance changes
* Receiver balance changes
* Transaction-type indicators
* Other behavioral transaction features

The feature engineering notebook is available at:

```text
notebooks/feature_engineering.ipynb
```

## Machine Learning Pipeline

The general workflow used in this project is:

```text
Raw Transaction Data
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Fraud Prediction
```

## Fraud Detection Challenge

Fraud detection is a highly imbalanced classification problem.

In the dataset, legitimate transactions significantly outnumber fraudulent transactions.

Because of this imbalance, accuracy alone is not sufficient for evaluating the model.

The project therefore considers metrics such as:

* Precision
* Recall
* F1-score
* Confusion matrix
* ROC-AUC
* Precision-Recall performance

### Why Recall Matters

For fraud detection, failing to identify an actual fraudulent transaction can be costly.

Therefore, recall is an important metric because it measures how many of the actual fraudulent transactions are successfully detected.

At the same time, precision must also be considered because incorrectly flagging legitimate transactions can negatively affect users and financial institutions.

## Model Training

The model training pipeline is implemented in:

```text
src/train.py
```

The training process generally involves:

1. Loading the processed dataset.
2. Selecting relevant features.
3. Separating features and target labels.
4. Splitting the data into training and testing sets.
5. Training the machine learning model.
6. Saving the trained model.
7. Preparing the model for evaluation and prediction.

## Model Evaluation

Model evaluation is implemented in:

```text
src/evaluate.py
```

The evaluation process examines the model using classification metrics such as:

```text
Precision
Recall
F1-score
Confusion Matrix
ROC-AUC
```

Example evaluation workflow:

```bash
python src/evaluate.py
```

The exact command may depend on the implementation of the evaluation script and the location of the processed data/model files.

## Making Predictions

The prediction pipeline is implemented in:

```text
src/predict.py
```

The prediction module can be used to process transaction information and generate a fraud prediction using the trained model.

A typical workflow is:

```text
Transaction
     │
     ▼
Feature Processing
     │
     ▼
Trained Model
     │
     ▼
Prediction
     │
     ├── Legitimate
     │
     └── Potential Fraud
```

## Running the Project

After installing the dependencies and downloading the dataset, the project can be executed through the different stages.

### Data Processing

```bash
python src/data_processing.py
```

### Feature Engineering

```bash
python src/feature_engineering.py
```

### Model Training

```bash
python src/train.py
```

### Model Evaluation

```bash
python src/evaluate.py
```

### Prediction

```bash
python src/predict.py
```

> The exact arguments required by these scripts depend on their implementation. Check the source files for the available command-line options.

## Notebooks

The project includes Jupyter notebooks for analysis and experimentation.

### Data Understanding

```text
notebooks/data_understanding.ipynb
```

This notebook explores the dataset and identifies important patterns and characteristics.

### Feature Engineering

```text
notebooks/feature_engineering.ipynb
```

This notebook demonstrates the creation and analysis of features used for fraud detection.

To start Jupyter Notebook:

```bash
jupyter notebook
```

## Results

The model should be evaluated primarily using fraud-sensitive metrics rather than accuracy alone.

Recommended results to report include:

| Metric    |        Result |
| --------- | ------------: |
| Accuracy  | To be updated |
| Precision | To be updated |
| Recall    | To be updated |
| F1-Score  | To be updated |
| ROC-AUC   | To be updated |

These values should be replaced with the actual results obtained from the final trained model.

## Limitations

This project has several limitations:

1. The dataset is simulated rather than collected from a real mobile money platform.
2. Fraud patterns in real-world environments may differ from those represented in the dataset.
3. The dataset is highly imbalanced.
4. A model trained on this dataset may require additional validation before being used in a production financial environment.
5. Fraud patterns can change over time as attackers adapt to detection systems.

## Future Improvements

Future versions of the project could include:

* Real-time fraud detection
* Streaming transaction analysis
* Advanced anomaly detection
* Ensemble learning
* Hyperparameter optimization
* Explainable AI for fraud predictions
* Real-time dashboards
* API deployment
* Cloud deployment
* Model monitoring
* Fraud-risk scoring
* Continuous model retraining
* Integration with mobile money transaction systems

## Ethical Considerations

Fraud detection systems should be designed carefully to avoid unfairly blocking legitimate users.

A production system should consider:

* False positives
* False negatives
* User privacy
* Data protection
* Model transparency
* Bias and fairness
* Human review of high-risk transactions

Predictions from a machine learning model should be treated as decision-support information rather than automatically assuming that every flagged transaction is fraudulent.

## Team Members

| Name                     | Role                            | GitHub                         |
| ------------             | ------------------------------- | ------------------------------ |
| Richmond Afoblikame      | Project Lead / Machine Learning | https://github.com/Richmond-24 |
| Npangna Balstignan Issac | Data Analysis                   | https://github.com/isaac2334   |
| Barnabas Natan           | Feature Engineering        | https://github.com/N0542426    |
                         


## Project Repository

GitHub:

https://github.com/Richmond-24/mobile-money-fraud-detection

## Acknowledgements

This project uses the PaySim mobile money transaction dataset available through Kaggle.

Dataset:

[https://www.kaggle.com/datasets/ealaxi/paysim1]

The project is intended for educational and research purposes and demonstrates how machine learning techniques can be applied to financial fraud detection.

## License

This project is intended for educational and research purposes.

If this project is later distributed publicly, an appropriate open-source license such as the MIT License can be added after reviewing the licensing requirements of the project and its dependencies.


## Author

Richmond Afoblikame

GitHub:
[https://github.com/Richmond-24]


**Mobile Money Fraud Detection — Using Machine Learning to Identify Suspicious Transactions**
