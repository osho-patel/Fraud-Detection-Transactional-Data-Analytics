# Fraud Detection & Risk Assessment Using Machine Learning

An end-to-end machine learning project for identifying potentially fraudulent financial transactions using Python, exploratory data analysis, feature engineering, and ensemble learning.

## Project Overview

Financial fraud detection is a challenging classification problem because fraudulent transactions represent a very small proportion of total transactions.

This project investigates fraud patterns in the PaySim synthetic financial transaction dataset and compares multiple machine learning algorithms to identify suspicious transactions.

The project includes exploratory data analysis, feature engineering, chronological data splitting, model training, hyperparameter tuning, performance evaluation, and model inference.

## Tech Stack

**Programming:** Python

**Data Analysis:** Pandas, NumPy

**Visualization:** Matplotlib, Seaborn

**Machine Learning:** Scikit-learn, XGBoost, LightGBM

**Development Environment:** Google Colab

**Model Serialization:** Joblib

## Dataset

**Source:** [PaySim — Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)

| Dataset characteristic | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Fraudulent transactions | 8,213 |
| Legitimate transactions | 6,354,407 |
| Fraud prevalence | 0.129% |
| Original features | 11 |

The dataset contains five transaction types: CASH_IN, CASH_OUT, DEBIT, PAYMENT, and TRANSFER.

Fraudulent transactions occur in the TRANSFER and CASH_OUT categories.

## Project Workflow

1. **Data Exploration:** Investigated class imbalance, transaction distributions, and fraud patterns.
2. **Feature Engineering:** Created transaction hour, transaction day, log-transformed amount, and amount-to-sender-balance ratio features.
3. **Data Splitting:** Used chronological training, validation, and testing partitions to better reflect future transaction prediction.
4. **Model Development:** Trained Logistic Regression, Random Forest, XGBoost, and LightGBM.
5. **Model Optimization:** Tuned Random Forest hyperparameters and optimized classification thresholds on validation data.
6. **Model Evaluation:** Compared fraud precision, recall, F1-score, ROC-AUC, and Average Precision.
7. **Robustness Analysis:** Compared machine learning performance with a simple rule-based detector and examined less obvious fraud cases.
8. **Inference:** Saved the selected model and tested predictions on synthetic transaction examples.

## Feature Engineering

Important engineered features included:

| Feature | Description |
|---|---|
| transaction_hour | Hour derived from the simulation step |
| transaction_day | Day derived from the simulation step |
| log_amount | Logarithmic transformation of transaction amount |
| amount_to_oldbalance | Ratio of transaction amount to sender's original balance |
| Transaction type indicators | One-hot encoded transaction categories |

The primary model used transaction attributes available before execution rather than post-transaction balances.

## Model Comparison

The following results were obtained on the validation dataset using model-specific classification thresholds.

| Model | Fraud F1-score |
|---|---:|
| Logistic Regression | 0.5745 |
| Random Forest | 0.9970 |
| XGBoost | 0.9763 |
| LightGBM | 0.9784 |
| **Tuned Random Forest** | **0.9979** |

### Selected Random Forest Configuration

- Number of estimators: 250
- Maximum depth: 25
- Minimum samples per leaf: 2
- Maximum features: sqrt
- Classification threshold: approximately 0.7187

**Validation performance:**

| Metric | Result |
|---|---:|
| Accuracy | 99.9974% |
| Fraud precision | 99.83% |
| Fraud recall | 99.75% |
| Fraud F1-score | 0.9979 |

These results describe performance on the synthetic validation dataset and should not be interpreted as expected performance on real banking transactions.

## Robustness Analysis

Exploratory analysis revealed that approximately 97.63% of fraudulent transactions involved transferring the sender's entire available balance.

A simple rule-based detector using this relationship achieved a validation fraud F1-score of approximately 0.9919.

To assess whether Random Forest learned additional patterns, the tuned model was evaluated on transactions that did not match the full-balance rule.

| Metric | Non-full-balance subset |
|---|---:|
| Transactions | 189,986 |
| Fraud cases | 19 |
| Fraud detected | 16 |
| Fraud missed | 3 |
| False positives | 2 |
| Fraud precision | 88.89% |
| Fraud recall | 84.21% |
| Fraud F1-score | 0.8649 |

This analysis demonstrates that the model identified some fraud cases beyond the dominant full-balance pattern, although the difficult subset was small.

## Model Inference

The selected model was saved using Joblib along with its classification threshold and expected feature names.

Five illustrative synthetic transactions were tested to verify that the saved model could generate predictions.

These examples are demonstration inputs rather than independently labeled real-world transactions.

## Limitations

- The dataset is synthetic and contains highly predictive patterns that may not generalize to real financial transactions.
- The dataset has significant class imbalance and chronological variation in fraud prevalence.
- The model relies heavily on sender-balance relationships.
- The difficult non-full-balance subset contains only 19 fraudulent validation cases.
- Predicted scores have not been calibrated as real-world fraud probabilities.
- Real-world deployment would require additional validation, monitoring, and investigation workflows.

## Future Improvements

- Evaluate performance on an independent, representative transaction dataset.
- Investigate model calibration and threshold selection under realistic fraud prevalence.
- Study concept drift and changing fraud behavior.
- Incorporate additional transaction history and behavioral features.
- Develop an API or dashboard for demonstrating fraud-risk predictions.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for fraud detection using a highly imbalanced synthetic financial dataset.

The tuned Random Forest achieved a validation fraud F1-score of approximately 0.9979. Further robustness experiments showed that this strong performance was substantially influenced by a synthetic-data pattern involving full-balance transfers.

The project emphasizes not only model development and optimization but also careful evaluation, interpretation, and recognition of limitations.
