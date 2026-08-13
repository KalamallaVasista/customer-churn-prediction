import pandas as pd
import numpy as np
import joblib
import shap

# ============================================================
# INDIVIDUAL CUSTOMER SHAP EXPLANATION
# ============================================================

print("=" * 60)
print("INDIVIDUAL CUSTOMER SHAP EXPLANATION")
print("=" * 60)

# ============================================================
# STEP 1 — LOAD MODEL
# ============================================================

model = joblib.load(
    "Data/Processed/balanced_random_forest.pkl"
)

print("\nModel loaded successfully!")
print("Model: Balanced Random Forest")

# ============================================================
# STEP 2 — LOAD TEST DATA
# ============================================================

X_test = pd.read_csv(
    "Data/Processed/X_test.csv"
)

print("\nTest data loaded successfully!")
print("Test data shape:")
print(X_test.shape)

# ============================================================
# STEP 3 — LOAD PREPROCESSOR
# ============================================================

preprocessor = joblib.load(
    "Data/Processed/preprocessor.pkl"
)

encoded_feature_names = (
    preprocessor.get_feature_names_out()
)

print("\nFeature names loaded successfully!")
print("Number of features:")
print(len(encoded_feature_names))

# ============================================================
# STEP 4 — SELECT ONE CUSTOMER
# ============================================================

customer_index = 0

customer = X_test.iloc[
    [customer_index]
]

print("\n" + "=" * 60)
print("CUSTOMER SELECTED")
print("=" * 60)

print("\nCustomer index:")
print(customer_index)

# ============================================================
# STEP 5 — PREDICT CUSTOMER
# ============================================================

prediction = model.predict(customer)

probability = model.predict_proba(customer)

churn_probability = probability[0][1]

print("\nPrediction:")

if prediction[0] == 1:
    print("CUSTOMER WILL CHURN")
else:
    print("CUSTOMER WILL NOT CHURN")

print("\nChurn probability:")
print(
    f"{churn_probability * 100:.2f}%"
)

# ============================================================
# STEP 6 — CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)

print("SHAP explainer created successfully!")

# ============================================================
# STEP 7 — CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating customer SHAP values...")

shap_values = explainer.shap_values(
    customer
)

# ============================================================
# STEP 8 — HANDLE SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):

    customer_shap_values = shap_values[1][0]

elif len(shap_values.shape) == 3:

    customer_shap_values = shap_values[0, :, 1]

else:

    customer_shap_values = shap_values[0]

print("\nSHAP values calculated successfully!")

# ============================================================
# STEP 9 — CREATE SHAP EXPLANATION TABLE
# ============================================================

shap_explanation = pd.DataFrame({
    "Feature": encoded_feature_names,
    "SHAP_Value": customer_shap_values,
    "Absolute_SHAP": np.abs(customer_shap_values)
})

# Sort by strongest impact

shap_explanation = shap_explanation.sort_values(
    by="Absolute_SHAP",
    ascending=False
)

# ============================================================
# STEP 10 — DISPLAY TOP FEATURES
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 CUSTOMER-SPECIFIC SHAP FEATURES")
print("=" * 60)

print(
    shap_explanation[
        [
            "Feature",
            "SHAP_Value"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# ============================================================
# STEP 11 — RISK-INCREASING FEATURES
# ============================================================

risk_increasing = shap_explanation[
    shap_explanation["SHAP_Value"] > 0
].head(5)

print("\n" + "=" * 60)
print("FACTORS INCREASING CHURN RISK")
print("=" * 60)

if len(risk_increasing) > 0:

    print(
        risk_increasing[
            ["Feature", "SHAP_Value"]
        ].to_string(index=False)
    )

else:

    print("No strong risk-increasing features found.")

# ============================================================
# STEP 12 — RISK-DECREASING FEATURES
# ============================================================

risk_decreasing = shap_explanation[
    shap_explanation["SHAP_Value"] < 0
].sort_values(
    by="SHAP_Value",
    ascending=False
)

print("\n" + "=" * 60)
print("FACTORS DECREASING CHURN RISK")
print("=" * 60)

if len(risk_decreasing) > 0:

    print(
        risk_decreasing[
            ["Feature", "SHAP_Value"]
        ].head(5).to_string(index=False)
    )

else:

    print("No strong risk-decreasing features found.")

# ============================================================
# STEP 13 — SAVE CUSTOMER EXPLANATION
# ============================================================

output_path = (
    "Data/Processed/customer_shap_explanation.csv"
)

shap_explanation.to_csv(
    output_path,
    index=False
)

print("\nCustomer SHAP explanation saved:")
print(output_path)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("INDIVIDUAL CUSTOMER SHAP EXPLANATION COMPLETED!")
print("=" * 60)