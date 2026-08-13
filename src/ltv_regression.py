import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CUSTOMER LTV REGRESSION MODEL
# ============================================================

print("=" * 60)
print("CUSTOMER LTV REGRESSION MODEL")
print("=" * 60)


# ============================================================
# STEP 1 — LOAD CLEANED DATA
# ============================================================

file_path = "Data/Processed/cleaned_telco_churn.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("Dataset shape:")
print(df.shape)


# ============================================================
# STEP 2 — SELECT ACTIVE CUSTOMERS
# ============================================================

active_customers = df[
    df["Churn"] == "No"
].copy()

print("\nActive customers:")
print(len(active_customers))


# ============================================================
# STEP 3 — CREATE TARGET VARIABLE
# ============================================================

# Baseline lifetime revenue:
#
# MonthlyCharges × EstimatedLifetimeMonths
#
# We use the average tenure of active customers
# as the estimated future lifetime.

average_active_tenure = (
    active_customers["tenure"].mean()
)

active_customers["EstimatedLifetimeMonths"] = np.maximum(
    active_customers["tenure"],
    average_active_tenure
)

active_customers["LifetimeRevenue"] = (
    active_customers["MonthlyCharges"]
    * active_customers["EstimatedLifetimeMonths"]
)


print("\nAverage active customer tenure:")
print(
    round(average_active_tenure, 2)
)

print("\nTarget variable created:")
print("LifetimeRevenue")


# ============================================================
# STEP 4 — SELECT REGRESSION FEATURES
# ============================================================

features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X = active_customers[features]

y = active_customers["LifetimeRevenue"]


print("\nRegression features:")
print(features)

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# STEP 5 — TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n" + "=" * 60)
print("LTV REGRESSION TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining features:")
print(X_train.shape)

print("\nTesting features:")
print(X_test.shape)


# ============================================================
# STEP 6 — TRAIN RANDOM FOREST REGRESSOR
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST REGRESSOR TRAINING")
print("=" * 60)

regressor = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=10
)

regressor.fit(
    X_train,
    y_train
)

print("\nLTV regression model trained successfully!")


# ============================================================
# STEP 7 — MAKE PREDICTIONS
# ============================================================

y_pred = regressor.predict(
    X_test
)

print("\nPredictions generated successfully!")

print("Number of predictions:")
print(len(y_pred))


# ============================================================
# STEP 8 — MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("LTV REGRESSION MODEL EVALUATION")
print("=" * 60)

print("\nMean Absolute Error:")
print(round(mae, 2))

print("\nRoot Mean Squared Error:")
print(round(rmse, 2))

print("\nR2 Score:")
print(round(r2, 4))


# ============================================================
# STEP 9 — DISPLAY SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "Actual_LTV": y_test.values,

    "Predicted_LTV": y_pred

})


print("\n" + "=" * 60)
print("SAMPLE LTV PREDICTIONS")
print("=" * 60)

print(
    results.head(10).to_string(
        index=False
    )
)


# ============================================================
# STEP 10 — SAVE MODEL
# ============================================================

processed_path = "Data/Processed"

os.makedirs(
    processed_path,
    exist_ok=True
)

model_path = (
    f"{processed_path}/ltv_regression_model.pkl"
)

joblib.dump(
    regressor,
    model_path
)

print("\n" + "=" * 60)
print("LTV REGRESSION MODEL SAVING")
print("=" * 60)

print("\nModel saved successfully!")

print("File:")
print(model_path)


# ============================================================
# STEP 11 — SAVE PREDICTIONS
# ============================================================

prediction_path = (
    f"{processed_path}/ltv_regression_predictions.csv"
)

results.to_csv(
    prediction_path,
    index=False
)

print("\nPredictions saved successfully!")

print("File:")
print(prediction_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("LTV REGRESSION COMPLETED SUCCESSFULLY!")
print("=" * 60)