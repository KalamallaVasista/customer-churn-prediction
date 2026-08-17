# Customer Churn Prediction

## Project Overview

Customer churn prediction is a data analytics and machine learning project designed to identify customers who are likely to leave a telecommunications service.

The project uses the Telco Customer Churn dataset containing 7,043 customer records and 21 original attributes.

## Objectives

The main objectives of this project are:

1. Understand customer churn patterns.
2. Clean and validate the customer dataset.
3. Perform exploratory data analysis.
4. Identify important factors associated with customer churn.
5. Perform customer churn analysis using PostgreSQL.
6. Build machine learning models to predict customer churn.
7. Compare multiple classification models.
8. Identify important churn-related features.
9. Estimate customer lifetime value.
10. Generate retention priorities for customers.
11. Explain model predictions using SHAP.
12. Provide business-oriented insights and recommendations.

## Dataset

The project uses the Telco Customer Churn dataset.

### Dataset Size

- Records: 7,043
- Original columns: 21
- Target variable: `Churn`

### Target Variable

The `Churn` column indicates whether a customer left the telecommunications service.

- `Yes` → Customer churned
- `No` → Customer did not churn

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- PostgreSQL
- SQLAlchemy
- Matplotlib
- Git
- GitHub

## Machine Learning Models

The project evaluates the following models:

1. Logistic Regression
2. Balanced Logistic Regression
3. Decision Tree
4. Random Forest
5. Balanced Random Forest
6. XGBoost

The models are compared using:

- Accuracy
- Precision
- Recall
- F1 Score

## PostgreSQL Analysis

PostgreSQL is used to analyze customer churn patterns based on:

- Contract type
- Payment method
- Internet service
- Tenure groups
- Technical support

The PostgreSQL database contains all 7,043 customer records.

## Explainability

SHAP is used to understand the contribution of features to individual and overall model predictions.

Feature importance analysis is also performed to identify the major factors influencing churn.

## Business Analysis

The project also includes:

- Customer Lifetime Value analysis
- Retention priority analysis
- Churn risk identification
- Business recommendations

## Project Outcome

The final project combines data cleaning, exploratory analysis, SQL analysis, machine learning, explainability, and business insights into a complete customer churn analytics pipeline.