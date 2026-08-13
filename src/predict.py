import joblib
import pandas as pd

# ==========================================
# STEP 59 — LOAD SAVED MODEL
# ==========================================

model_path = "Data/Processed/balanced_random_forest.pkl"

model = joblib.load(model_path)

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

print("\nSaved model loaded successfully!")

# ==========================================
# STEP 60 — LOAD TEST CUSTOMER DATA
# ==========================================

# Load one customer from the test dataset
X_test = pd.read_csv(
    "Data/Processed/X_test.csv"
)

# Select the first customer
customer = X_test.iloc[[0]]

print("\n" + "=" * 60)
print("TEST CUSTOMER")
print("=" * 60)

print("\nCustomer feature shape:")
print(customer.shape)

print("\nNumber of features:")
print(customer.shape[1])

# ==========================================
# STEP 61 — MAKE PREDICTION
# ==========================================

prediction = model.predict(customer)

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("\nPredicted value:")
print(prediction[0])

if prediction[0] == 1:
    print("\nPrediction: CUSTOMER WILL CHURN")
else:
    print("\nPrediction: CUSTOMER WILL NOT CHURN")
    
# ==========================================
# STEP 62 — CHURN PROBABILITY
# ==========================================

probability = model.predict_proba(customer)

churn_probability = probability[0][1]
no_churn_probability = probability[0][0]

print("\n" + "=" * 60)
print("CHURN PROBABILITY")
print("=" * 60)

print("\nNo Churn Probability:")
print(f"{no_churn_probability * 100:.2f}%")

print("\nChurn Probability:")
print(f"{churn_probability * 100:.2f}%")

# ==========================================
# STEP 63 — CUSTOMER RISK LEVEL
# ==========================================

if churn_probability < 0.30:
    risk_level = "LOW RISK"
elif churn_probability < 0.60:
    risk_level = "MEDIUM RISK"
else:
    risk_level = "HIGH RISK"

print("\n" + "=" * 60)
print("CUSTOMER RISK LEVEL")
print("=" * 60)

print("\nRisk Level:")
print(risk_level)

# ==========================================
# STEP 64 — FIND DIFFERENT RISK CUSTOMERS
# ==========================================

all_probabilities = model.predict_proba(X_test)[:, 1]

risk_results = pd.DataFrame({
    "Churn_Probability": all_probabilities
})

# Assign risk levels
risk_results["Risk_Level"] = risk_results["Churn_Probability"].apply(
    lambda x: "LOW RISK" if x < 0.30
    else "MEDIUM RISK" if x < 0.60
    else "HIGH RISK"
)

print("\n" + "=" * 60)
print("RISK LEVEL DISTRIBUTION")
print("=" * 60)

print(
    risk_results["Risk_Level"].value_counts()
)

print("\n" + "=" * 60)
print("HIGHEST RISK CUSTOMER")
print("=" * 60)

highest_risk = risk_results.loc[
    risk_results["Churn_Probability"].idxmax()
]

print(highest_risk)

print("\n" + "=" * 60)
print("LOWEST RISK CUSTOMER")
print("=" * 60)

lowest_risk = risk_results.loc[
    risk_results["Churn_Probability"].idxmin()
]

print(lowest_risk)

# ==========================================
# STEP 66 — FINAL PREDICTION SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("FINAL CUSTOMER CHURN SUMMARY")
print("=" * 60)

print("\nPrediction:")
print(
    "CUSTOMER WILL CHURN"
    if prediction[0] == 1
    else "CUSTOMER WILL NOT CHURN"
)

print("\nChurn Probability:")
print(f"{churn_probability * 100:.2f}%")

print("\nRisk Level:")
print(risk_level)

print("\nModel Used:")
print("Balanced Random Forest")

print("\nPrediction completed successfully!")