import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# RETENTION PRIORITY ENGINE
# ============================================================

print("=" * 60)
print("RETENTION PRIORITY ENGINE")
print("=" * 60)

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

ltv_path = "Data/Processed/customer_ltv.csv"

df = pd.read_csv(ltv_path)

print("\nLTV data loaded successfully!")

print("Dataset shape:")
print(df.shape)

# ============================================================
# STEP 2 — LOAD MODEL
# ============================================================

model = joblib.load(
    "Data/Processed/balanced_random_forest.pkl"
)

print("\nBalanced Random Forest loaded successfully!")

# ============================================================
# STEP 3 — LOAD TEST DATA
# ============================================================

X_test = pd.read_csv(
    "Data/Processed/X_test.csv"
)

print("\nTest data loaded successfully!")

print("Test data shape:")
print(X_test.shape)

# ============================================================
# STEP 4 — PREDICT CHURN PROBABILITY
# ============================================================

probability = model.predict_proba(X_test)

churn_probability = probability[:, 1]

# ============================================================
# STEP 5 — CREATE PREDICTION DATAFRAME
# ============================================================

prediction_df = pd.DataFrame({

    "Churn_Probability": churn_probability

})

# ============================================================
# STEP 6 — CREATE RISK LEVEL
# ============================================================

def assign_risk(probability):

    if probability < 0.30:
        return "LOW RISK"

    elif probability < 0.60:
        return "MEDIUM RISK"

    else:
        return "HIGH RISK"


prediction_df["Risk_Level"] = (
    prediction_df["Churn_Probability"]
    .apply(assign_risk)
)

# ============================================================
# STEP 7 — MATCH LTV DATA
# ============================================================

# The test data contains 1409 customers.
# The LTV dataset contains all 7043 customers.
#
# We use the same random_state and stratified split
# to reproduce the test customer positions.

cleaned_df = pd.read_csv(
    "Data/Processed/cleaned_telco_churn.csv"
)

from sklearn.model_selection import train_test_split

customer_indices = np.arange(
    len(cleaned_df)
)

train_indices, test_indices = train_test_split(
    customer_indices,
    test_size=0.20,
    random_state=42,
    stratify=cleaned_df["Churn"]
)

test_ltv = df.iloc[test_indices].copy()

test_ltv = test_ltv.reset_index(
    drop=True
)

# ============================================================
# STEP 8 — COMBINE CHURN + LTV
# ============================================================

retention_df = pd.DataFrame({

    "customerID":
        test_ltv["customerID"].values,

    "Churn_Probability":
        prediction_df["Churn_Probability"].values,

    "Risk_Level":
        prediction_df["Risk_Level"].values,

    "Base_LTV":
        test_ltv["Base_LTV"].values,

    "LTV_Segment":
        test_ltv["LTV_Segment"].values

})

# ============================================================
# STEP 9 — RETENTION PRIORITY
# ============================================================

def calculate_priority(row):

    risk = row["Risk_Level"]

    ltv = row["LTV_Segment"]

    if risk == "HIGH RISK" and ltv == "HIGH LTV":
        return "CRITICAL"

    elif risk == "HIGH RISK" and ltv == "MEDIUM LTV":
        return "HIGH"

    elif risk == "HIGH RISK" and ltv == "LOW LTV":
        return "MEDIUM"

    elif risk == "MEDIUM RISK" and ltv == "HIGH LTV":
        return "HIGH"

    elif risk == "MEDIUM RISK" and ltv == "MEDIUM LTV":
        return "MEDIUM"

    elif risk == "MEDIUM RISK" and ltv == "LOW LTV":
        return "LOW"

    elif risk == "LOW RISK" and ltv == "HIGH LTV":
        return "MAINTAIN"

    elif risk == "LOW RISK" and ltv == "MEDIUM LTV":
        return "LOW"

    else:
        return "LOW"


retention_df["Retention_Priority"] = (
    retention_df.apply(
        calculate_priority,
        axis=1
    )
)

# ============================================================
# STEP 10 — RECOMMENDED ACTION
# ============================================================

def recommended_action(priority):

    if priority == "CRITICAL":

        return (
            "Immediate retention intervention. "
            "Offer personalized incentive and "
            "proactive customer support."
        )

    elif priority == "HIGH":

        return (
            "Prioritize customer for retention campaign "
            "and consider a targeted offer."
        )

    elif priority == "MEDIUM":

        return (
            "Monitor customer and consider "
            "a suitable service or pricing offer."
        )

    elif priority == "MAINTAIN":

        return (
            "Maintain relationship and provide "
            "good customer experience."
        )

    else:

        return (
            "Continue regular engagement "
            "and monitor customer behavior."
        )


retention_df["Recommended_Action"] = (
    retention_df["Retention_Priority"]
    .apply(recommended_action)
)

# ============================================================
# STEP 11 — DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RETENTION PRIORITY DISTRIBUTION")
print("=" * 60)

print(
    retention_df[
        "Retention_Priority"
    ].value_counts()
)

# ============================================================
# STEP 12 — TOP PRIORITY CUSTOMERS
# ============================================================

priority_order = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "MAINTAIN": 4,
    "LOW": 5
}

retention_df["Priority_Rank"] = (
    retention_df["Retention_Priority"]
    .map(priority_order)
)

top_customers = (
    retention_df
    .sort_values(
        by=[
            "Priority_Rank",
            "Churn_Probability",
            "Base_LTV"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .head(20)
)

print("\n" + "=" * 60)
print("TOP 20 RETENTION PRIORITY CUSTOMERS")
print("=" * 60)

print(
    top_customers[
        [
            "customerID",
            "Churn_Probability",
            "Risk_Level",
            "Base_LTV",
            "LTV_Segment",
            "Retention_Priority"
        ]
    ].to_string(index=False)
)

# ============================================================
# STEP 13 — REMOVE TEMPORARY RANK
# ============================================================

retention_df = retention_df.drop(
    columns=["Priority_Rank"]
)

# ============================================================
# STEP 14 — SAVE RESULTS
# ============================================================

output_path = (
    "Data/Processed/retention_priority.csv"
)

retention_df.to_csv(
    output_path,
    index=False
)

print("\nRetention priority data saved successfully!")

print("File:")
print(output_path)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("RETENTION PRIORITY ENGINE COMPLETED SUCCESSFULLY!")
print("=" * 60)