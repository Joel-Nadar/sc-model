# ANN-Based Loan Default Risk Prediction

An Artificial Neural Network (ANN) built with TensorFlow/Keras that predicts the probability a loan applicant will default, based on their financial and credit-related information.

## Overview

Banks need to assess, before approving a loan, how likely an applicant is to default. This project frames that as a **binary classification** problem:

- `0` → Non-default
- `1` → Default

The model takes in a customer's financial details and outputs a probability between 0 and 1, which is then converted into a risk category.

## Why an ANN?

Loan default depends on multiple factors simultaneously (income, loan amount, credit history, employment length, etc.), and the relationship between these factors and default risk isn't simple or linear. A neural network with hidden layers can learn these nonlinear interactions better than a single formula or rule-based system.

## Dataset

**Source:** [Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
**File:** `credit_risk_dataset.csv`
**Rows:** ~32,580
**Target column:** `loan_status` (`0` = No Default, `1` = Default)

| Column | Description | Type |
|---|---|---|
| `person_age` | Borrower age (years) | Numeric |
| `person_income` | Annual income | Numeric |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER | Categorical |
| `person_emp_length` | Employment length (years) | Numeric |
| `loan_intent` | Purpose of the loan | Categorical |
| `loan_grade` | Credit grade (A–G) | Categorical |
| `loan_amnt` | Loan amount | Numeric |
| `loan_int_rate` | Interest rate (%) | Numeric |
| `loan_status` | Target variable | Binary |
| `loan_percent_income` | Loan amount as % of income | Numeric |
| `cb_person_default_on_file` | Prior default on record (Y/N) | Categorical |
| `cb_person_cred_hist_length` | Credit history length (years) | Numeric |

## Project Architecture

```
Loan Dataset
     │
     ▼
Data Preprocessing
     │
  ┌──┴──┐
Numeric  Categorical
  │        │
Scaling  Encoding
  │        │
  └───┬────┘
      ▼
Train/Test Split
      │
      ▼
   ANN Model
(Dense 64 → Dropout → Dense 32 → Dropout → Dense 1)
      │
      ▼
Default Probability
      │
      ▼
Evaluation & Analysis
```

## Model Architecture

```
Input
  ↓
Dense(64, ReLU)
  ↓
Dropout(0.3)
  ↓
Dense(32, ReLU)
  ↓
Dropout(0.2)
  ↓
Dense(1, Sigmoid)
  ↓
Default Probability
```

- **ReLU** in hidden layers lets the network learn nonlinear patterns.
- **Sigmoid** on the output layer squashes the result into a 0–1 probability.
- **Dropout** randomly deactivates neurons during training to reduce overfitting.
- **Adam** optimizer with **binary cross-entropy** loss (standard for two-class problems).
- **Early stopping** halts training once validation loss stops improving.
- **Class weights** correct for the imbalance between defaulters and non-defaulters.

## Project Structure

```
Loan_Default_ANN/
│
├── credit_risk_dataset.csv     # Dataset
├── loan_default_ann.py         # Full pipeline: preprocessing, training, evaluation
├── loan_default_ann.keras      # Saved trained model (generated after running)
└── README.md                   # This file
```

## Requirements

```
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

1. Place `credit_risk_dataset.csv` in the same folder as `loan_default_ann.py` (or upload both to your Colab session's `/content` folder).
2. Run:
   ```
   python loan_default_ann.py
   ```
3. The script will:
   - Load and explore the dataset
   - Clean missing values
   - Preprocess numeric (StandardScaler) and categorical (OneHotEncoder) features
   - Split data into 80% training / 20% testing (stratified)
   - Build, compile, and train the ANN with early stopping and class weighting
   - Evaluate on the test set: Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Plot the confusion matrix, training/validation accuracy & loss curves, and ROC curve
   - Run a sample real-time prediction on a new customer
   - Save the trained model as `loan_default_ann.keras`

## Evaluation Metrics

Accuracy alone is misleading here because the dataset is imbalanced (most applicants don't default). A model that always predicts "no default" can still look ~90%+ accurate while catching zero real defaulters. That's why the project reports:

- **Accuracy** – overall correctness
- **Precision** – of predicted defaulters, how many actually defaulted
- **Recall** – of actual defaulters, how many were correctly caught (critical for banks — missing a defaulter is costly)
- **F1-score** – balance of precision and recall
- **ROC-AUC** – how well the model separates the two classes across thresholds
- **Confusion Matrix** – breakdown of TP / TN / FP / FN

## Risk Categories (for demo purposes)

| Probability Range | Risk Category |
|---|---|
| 0.00 – 0.30 | Low Risk |
| 0.30 – 0.60 | Medium Risk |
| 0.60 – 1.00 | High Risk |

*Note: these thresholds are for demonstration. In a real banking setting, cutoffs would be determined using business costs, regulatory requirements, and validation data.*

## Real-Time Prediction Example

```python
example_customer = {
    "person_age": [28],
    "person_income": [60000],
    "person_home_ownership": ["RENT"],
    "person_emp_length": [4],
    "loan_intent": ["PERSONAL"],
    "loan_grade": ["B"],
    "loan_amnt": [10000],
    "loan_int_rate": [11.5],
    "loan_percent_income": [0.17],
    "cb_person_default_on_file": ["N"],
    "cb_person_cred_hist_length": [5],
}
```

Output:
```
Default Probability: 73%
Risk Category: High Risk
```

## Possible Extensions

- Build a Streamlit or Gradio interface for interactive predictions
- Try alternative models (Random Forest, XGBoost) for comparison
- Perform a fairness audit across demographic subgroups
- Tune the classification threshold based on business cost of false negatives vs. false positives

## License / Attribution

Dataset sourced from the [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), originally compiled for credit risk research and educational use.
