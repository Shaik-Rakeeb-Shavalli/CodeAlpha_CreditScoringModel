# Credit Scoring Model – CodeAlpha Internship Project

## Overview

This project aims to predict an individual’s credit delinquency using historical financial data. Developed as part of the CodeAlpha Machine Learning Internship, the solution leverages Python, pandas, scikit-learn, imbalanced-learn, and XGBoost to build, train, and evaluate a robust credit scoring classification model.

## Objective

- **Task**: Predict credit delinquency (Delinquent_Account: 0 - Not delinquent, 1 - Delinquent)
- **Approach**: Classify individuals using engineered financial features and payment histories
- **Algorithms Used**: Logistic Regression, Random Forest, XGBoost (with class imbalance handling)

## Data

The dataset contains 500 customer records and 44 features, including:
- Demographics: Age, Employment Status, Location
- Financial: Income, Credit_Score, Credit_Utilization, Loan_Balance, Debt_to_Income_Ratio, Account_Tenure, Credit_Card_Type
- Payment History: Status for Months 1–6 mapped to numeric scores and aggregated into Recent_Payment_Score
- Target: Delinquent_Account (0/1)

## Data Exploration & Visualization

- Histograms and boxplots used for visualizing distributions of key features (Age, Income, Credit Score, Loan Balance, etc.)
- **Delinquent_Account** found to be highly imbalanced, with ~83% non-delinquent, ~17% delinquent.
- Feature correlations measured; Recent_Payment_Score, Credit_Utilization, and Missed_Payments emerged as top predictors.

## Feature Engineering

- Categorical features (Employment_Status, Location, Credit_Card_Type, Month_1–Month_6) one-hot encoded
- Recent_Payment_Score created as the sum of six-month payment statuses ('On-time' = 0, 'Late' = 1, 'Missed' = 2)
- Uninformative columns (Customer_ID, original month columns) dropped

## Modeling Approach

- Data split (80% train, 20% test) with stratification for target balance
- Class imbalance handled using **SMOTE** oversampling on the training set
- Main models trained and compared:
    - Logistic Regression
    - Random Forest
    - XGBoost (with `scale_pos_weight`)
- Hyperparameter tuning performed for XGBoost using RandomizedSearchCV

## Model Performance

All models were evaluated using:
- **Classification Report**: Precision, Recall, F1-Score for each class
- **ROC-AUC Score**
- **Confusion Matrix**

| Model               | Accuracy | ROC-AUC | Notes                                |
|---------------------|---------:|--------:|--------------------------------------|
| Logistic Regression | (score)  | (score) |                                      |
| Random Forest       | (score)  | (score) | Best overall balance                 |
| XGBoost             | (score)  | (score) | Tuned for class imbalance            |

*Note: Scores to be filled from your actual outputs.*

## Feature Importance

Top features for predicting credit delinquency (from Random Forest/XGBoost):
1. Recent_Payment_Score
2. Credit_Utilization
3. Missed_Payments
4. Debt_to_Income_Ratio
5. Account_Tenure

## Results Visualization

Model results visualized using:
- **ROC Curve**: `ROC_Curve.png`
- **Confusion Matrix**: `Confusion_Matrix.png`

## How to Run

1. Clone this repository or download the files
2. Open the notebook or script in Google Colab or locally
3. Place the dataset (`Delinquency_prediction_dataset.csv`) in your working directory
4. Run `pip install -r requirements.txt` (if requirements file is provided)
5. Execute cells sequentially for preprocessing, feature engineering, modeling, and evaluation
6. Final model saved as `credit_scoring_model.pkl`

## Files

- `Delinquency_prediction_dataset.csv`: Data file
- `credit_scoring_model.pkl`: Final trained model
- `main_notebook.ipynb` or `credit_scoring.py`: Main code
- `ROC_Curve.png`, `Confusion_Matrix.png`: Output visualizations

## Internship Submission

- Repository name: `CodeAlpha_CreditScoringModel`
- Explanation video: [LinkedIn link here]
- Submission form: [As shared via WhatsApp]

## Contact

- CodeAlpha: www.codealpha.tech
- For assistance: services@codealpha.tech

**Prepared by:** [Your Name]  
**Location:** Tiruchirappalli, Tamil Nadu, India  
**Date:** August 2025

*For any questions, please raise an issue or contact via mail.*
