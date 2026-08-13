import pandas as pd
import numpy as np
import os

# ============================================================
# CUSTOMER LIFETIME VALUE (LTV) ENGINE
# ============================================================

print("=" * 60)
print("CUSTOMER LIFETIME VALUE (LTV) ENGINE")
print("=" * 60)

# ============================================================
# STEP 1 — LOAD CLEANED DATA
# ============================================================

file_path = "Data/Processed/cleaned_telco_churn.csv"

df = pd.read_csv(file_path)

print("\nCleaned dataset loaded successfully!")

print("Dataset shape:")
print(df.shape)

# ============================================================
# STEP 2 — CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

print("\nChecking required columns...")

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("Missing columns:")
    print(missing_columns)

    raise ValueError(
        "Required columns are missing!"
    )

print("All required columns are available!")

# ============================================================
# STEP 3 — CALCULATE MONTHLY REVENUE
# ============================================================

df["MonthlyRevenue"] = df["MonthlyCharges"]

# ============================================================
# STEP 4 — ESTIMATE REMAINING CUSTOMER LIFETIME
# ============================================================

# Average tenure of customers who have not churned

active_customers = df[
    df["Churn"] == "No"
]

average_active_tenure = (
    active_customers["tenure"].mean()
)

print("\nAverage tenure of active customers:")
print(
    round(average_active_tenure, 2),
    "months"
)

# ============================================================
# STEP 5 — ESTIMATE CUSTOMER LIFETIME
# ============================================================

# Customers with longer tenure are assumed
# to have stronger retention potential.

df["EstimatedLifetimeMonths"] = np.maximum(
    df["tenure"],
    average_active_tenure
)

# ============================================================
# STEP 6 — CALCULATE BASE LTV
# ============================================================

df["Base_LTV"] = (
    df["MonthlyRevenue"]
    * df["EstimatedLifetimeMonths"]
)

# ============================================================
# STEP 7 — DISPLAY LTV RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LTV SUMMARY")
print("=" * 60)

print("\nAverage LTV:")
print(
    round(df["Base_LTV"].mean(), 2)
)

print("\nMinimum LTV:")
print(
    round(df["Base_LTV"].min(), 2)
)

print("\nMaximum LTV:")
print(
    round(df["Base_LTV"].max(), 2)
)

# ============================================================
# STEP 8 — CREATE LTV SEGMENTS
# ============================================================

low_threshold = df["Base_LTV"].quantile(0.33)

high_threshold = df["Base_LTV"].quantile(0.66)

def assign_ltv_segment(ltv):

    if ltv <= low_threshold:
        return "LOW LTV"

    elif ltv <= high_threshold:
        return "MEDIUM LTV"

    else:
        return "HIGH LTV"


df["LTV_Segment"] = df["Base_LTV"].apply(
    assign_ltv_segment
)

# ============================================================
# STEP 9 — LTV SEGMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("LTV SEGMENT DISTRIBUTION")
print("=" * 60)

print(
    df["LTV_Segment"].value_counts()
)

# ============================================================
# STEP 10 — TOP HIGH-VALUE CUSTOMERS
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 HIGH-VALUE CUSTOMERS")
print("=" * 60)

top_customers = df[
    [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "Base_LTV",
        "LTV_Segment"
    ]
].sort_values(
    by="Base_LTV",
    ascending=False
).head(10)

print(
    top_customers.to_string(index=False)
)

# ============================================================
# STEP 11 — SAVE LTV DATA
# ============================================================

processed_path = "Data/Processed"

os.makedirs(
    processed_path,
    exist_ok=True
)

output_path = (
    f"{processed_path}/customer_ltv.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nLTV data saved successfully!")

print("File:")
print(output_path)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("LTV ENGINE COMPLETED SUCCESSFULLY!")
print("=" * 60)