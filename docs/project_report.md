# Customer Churn Prediction & Retention Analytics

## 1. Project Overview

This project focuses on analyzing customer churn and identifying customers who are at high risk of leaving a telecommunications service.

The project combines:

- Python
- Pandas
- Scikit-learn
- XGBoost
- Random Forest
- SHAP
- PostgreSQL
- SQLAlchemy
- Data Visualization
- Customer Lifetime Value (LTV)
- Retention Priority Analysis

The objective is to understand customer churn patterns, build machine learning models to predict churn, explain the factors influencing churn, estimate customer lifetime value, and identify customers who require retention attention.

---

## 2. Business Problem

Customer churn is an important business problem for subscription-based telecommunications companies.

When customers leave, the company loses recurring revenue and may need to spend additional resources to acquire new customers.

The main business questions addressed in this project are:

1. What percentage of customers are churning?
2. Which customer segments have the highest churn rates?
3. Which services and payment methods are associated with higher churn?
4. Can machine learning predict customers who are likely to churn?
5. Which factors have the greatest influence on churn predictions?
6. Which customers have higher lifetime value?
7. Which customers should receive retention attention?

---

## 3. Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains:

- 7,043 customers
- Customer demographic information
- Account information
- Service information
- Billing information
- Churn status

The target variable is:

`Churn`

where:

- `Yes` = Customer churned
- `No` = Customer did not churn

---

## 4. Data Processing

The raw customer data was inspected and cleaned before analysis.

The data processing workflow included:

1. Loading the raw dataset
2. Inspecting columns and data types
3. Checking missing values
4. Checking duplicate records
5. Cleaning numerical fields
6. Converting appropriate columns to numerical format
7. Validating the cleaned dataset
8. Saving the processed dataset

The cleaned dataset contains 7,043 customer records.

---

## 5. Exploratory Data Analysis

Exploratory data analysis was performed to understand customer behavior and identify important churn patterns.

The analysis examined:

- Overall churn
- Churn by contract type
- Churn by payment method
- Churn by internet service
- Churn by tenure
- Churn by technical support
- Monthly charges
- Total charges
- Customer tenure

---

## 6. PostgreSQL Analysis

The customer dataset was imported into PostgreSQL for SQL-based analysis.

PostgreSQL was connected to Python using SQLAlchemy and psycopg2.

The database connection was successfully validated.

### Database validation

Total customers:

`7,043`

Total churned customers:

`1,869`

Total non-churned customers:

`5,174`

Overall churn rate:

`26.54%`

### SQL Analysis

The project includes SQL analysis for:

- Overall churn
- Churn by contract
- Churn by payment method
- Churn by internet service
- Churn by tenure group
- Churn by technical support
- High-risk customer segments
- High monthly charge customers
- Short-tenure churned customers
- Churned customer revenue

SQL queries are available in:

`sql/churn_analysis.sql`

---

## 7. Key PostgreSQL Findings

### Churn by Contract

| Contract | Total Customers | Churned Customers | Churn Rate |
|---|---:|---:|---:|
| Month-to-month | 3,875 | 1,655 | 42.71% |
| One year | 1,473 | 166 | 11.27% |
| Two year | 1,695 | 48 | 2.83% |

Month-to-month customers have the highest churn rate.

---

### Churn by Payment Method

| Payment Method | Total Customers | Churned Customers | Churn Rate |
|---|---:|---:|---:|
| Electronic check | 2,365 | 1,071 | 45.29% |
| Mailed check | 1,612 | 308 | 19.11% |
| Bank transfer (automatic) | 1,544 | 258 | 16.71% |
| Credit card (automatic) | 1,522 | 232 | 15.24% |

Electronic check customers have the highest churn rate.

---

### Churn by Internet Service

| Internet Service | Total Customers | Churned Customers | Churn Rate |
|---|---:|---:|---:|
| Fiber optic | 3,096 | 1,297 | 41.89% |
| DSL | 2,421 | 459 | 18.96% |
| No | 1,526 | 113 | 7.40% |

Fiber optic customers have a substantially higher churn rate than the other groups.

---

### Churn by Tenure

| Tenure Group | Total Customers | Churned Customers | Churn Rate |
|---|---:|---:|---:|
| 0-12 months | 2,186 | 1,037 | 47.44% |
| 13-24 months | 1,024 | 294 | 28.71% |
| 25-48 months | 1,594 | 325 | 20.39% |
| 49-72 months | 2,239 | 213 | 9.51% |

Customers with shorter tenure show substantially higher churn.

---

### Churn by Technical Support

| Technical Support | Total Customers | Churned Customers | Churn Rate |
|---|---:|---:|---:|
| No | 3,473 | 1,446 | 41.64% |
| Yes | 2,044 | 310 | 15.17% |
| No internet service | 1,526 | 113 | 7.40% |

Customers without technical support have a considerably higher churn rate.

---

## 8. Machine Learning

Machine learning models were trained to predict customer churn.

The dataset was divided into:

- Training data: 5,634 customers
- Testing data: 1,409 customers

The target distribution was:

### Training

- No Churn: 4,139
- Churn: 1,495

### Testing

- No Churn: 1,035
- Churn: 374

The models evaluated were:

1. Logistic Regression
2. Balanced Logistic Regression
3. Decision Tree
4. Random Forest
5. Balanced Random Forest
6. XGBoost

---

## 9. Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 |
| Balanced Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 |
| Decision Tree | 0.7984 | 0.6347 | 0.5668 | 0.5989 |
| Random Forest | 0.8062 | 0.6797 | 0.5107 | 0.5832 |
| Balanced Random Forest | 0.7559 | 0.5265 | 0.7968 | 0.6340 |
| XGBoost | 0.7984 | 0.6541 | 0.5107 | 0.5736 |

---

## 10. Final Model Selection

The project uses **Balanced Random Forest** as the selected model.

The selection is based on F1 Score.

Balanced Random Forest achieved:

- Accuracy: 75.59%
- Precision: 52.65%
- Recall: 79.68%
- F1 Score: 63.40%

The higher recall is particularly useful for churn prediction because identifying more potential churners can help the business target customers for retention activities.

The trained model was saved as:

`Data/Processed/balanced_random_forest.pkl`

---

## 11. Feature Importance

Feature importance was calculated using the Balanced Random Forest model.

The top features were:

| Feature | Importance |
|---|---:|
| Contract_Month-to-month | 0.1382 |
| tenure | 0.1163 |
| TotalCharges | 0.0965 |
| Contract_Two year | 0.0750 |
| OnlineSecurity_No | 0.0682 |
| MonthlyCharges | 0.0607 |
| TechSupport_No | 0.0546 |
| InternetService_Fiber optic | 0.0523 |
| PaymentMethod_Electronic check | 0.0465 |

The results indicate that contract type, tenure, charges, security services, technical support, internet service, and payment method are important predictive features.

---

## 12. SHAP Explainability

SHAP was used to explain machine learning predictions.

The analysis provides:

- Global feature importance
- Feature impact on predictions
- Individual customer explanations

The generated SHAP outputs include:

- `shap_feature_importance.csv`
- `shap_feature_importance.png`
- `shap_summary.png`
- `customer_shap_explanation.csv`

SHAP helps make the machine learning model more interpretable by showing which features contribute to churn predictions.

---

## 13. Customer Lifetime Value

Customer Lifetime Value (LTV) analysis was performed to estimate the potential value of customers.

The analysis uses customer billing and tenure information to estimate customer value and support business prioritization.

The generated outputs include:

- `customer_ltv.csv`
- `ltv_regression_predictions.csv`
- `ltv_regression_model.pkl`

---

## 14. Retention Priority

A retention-priority analysis was developed to identify customers who may require additional attention.

The analysis combines customer characteristics and churn-related information to help prioritize retention efforts.

The generated output is:

`retention_priority.csv`

This allows the business to focus retention resources on customers with higher potential churn risk and business value.

---

## 15. Revenue Impact

The PostgreSQL analysis identified:

- Churned customers: 1,869
- Estimated total monthly revenue associated with churned customers: 139,130.85
- Average monthly charge among churned customers: 74.44

This highlights the potential financial impact of customer churn.

---

## 16. Project Structure

```text
Customer Churn prediction/
│
├── Data/
│   ├── raw/
│   └── Processed/
│
├── dashboard/
│
├── docs/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── resources/
│
├── screenshots/
│
├── sql/
│   └── churn_analysis.sql
│
├── src/
│   ├── app.py
│   ├── clean_data.py
│   ├── customer_shap.py
│   ├── data_inspection.py
│   ├── database_connection.py
│   ├── eda.py
│   ├── eda_summary.py
│   ├── feature_engineering.py
│   ├── investigate_data.py
│   ├── load_data.py
│   ├── ltv_engine.py
│   ├── ltv_regression.py
│   ├── postgresql_analysis.py
│   ├── predict.py
│   ├── retention_priority.py
│   ├── shap_explainability.py
│   ├── test_environment.py
│   ├── train_model.py
│   └── validate_data.py
│
├── .gitignore
├── README.md
└── requirements.txt