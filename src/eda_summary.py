import pandas as pd

# Load cleaned dataset
file_path = "Data/Processed/cleaned_telco_churn.csv"
df = pd.read_csv(file_path)

print("=" * 70)
print("CUSTOMER CHURN - EDA SUMMARY")
print("=" * 70)


# ==========================================
# 1. CONTRACT
# ==========================================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

contract_highest = contract_churn["Yes"].idxmax()
contract_rate = contract_churn["Yes"].max()


# ==========================================
# 2. TENURE
# ==========================================

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]
)

tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

tenure_highest = tenure_churn["Yes"].idxmax()
tenure_rate = tenure_churn["Yes"].max()


# ==========================================
# 3. MONTHLY CHARGES
# ==========================================

df["MonthlyChargeGroup"] = pd.qcut(
    df["MonthlyCharges"],
    q=4,
    labels=[
        "Low",
        "Medium-Low",
        "Medium-High",
        "High"
    ]
)

charge_churn = pd.crosstab(
    df["MonthlyChargeGroup"],
    df["Churn"],
    normalize="index"
) * 100

charge_highest = charge_churn["Yes"].idxmax()
charge_rate = charge_churn["Yes"].max()


# ==========================================
# 4. INTERNET SERVICE
# ==========================================

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

internet_highest = internet_churn["Yes"].idxmax()
internet_rate = internet_churn["Yes"].max()


# ==========================================
# 5. PAYMENT METHOD
# ==========================================

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

payment_highest = payment_churn["Yes"].idxmax()
payment_rate = payment_churn["Yes"].max()


# ==========================================
# 6. TECH SUPPORT
# ==========================================

tech_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

tech_highest = tech_churn["Yes"].idxmax()
tech_rate = tech_churn["Yes"].max()


# ==========================================
# 7. ONLINE SECURITY
# ==========================================

security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

security_highest = security_churn["Yes"].idxmax()
security_rate = security_churn["Yes"].max()


# ==========================================
# 8. ONLINE BACKUP
# ==========================================

backup_churn = pd.crosstab(
    df["OnlineBackup"],
    df["Churn"],
    normalize="index"
) * 100

backup_highest = backup_churn["Yes"].idxmax()
backup_rate = backup_churn["Yes"].max()


# ==========================================
# 9. DEVICE PROTECTION
# ==========================================

device_churn = pd.crosstab(
    df["DeviceProtection"],
    df["Churn"],
    normalize="index"
) * 100

device_highest = device_churn["Yes"].idxmax()
device_rate = device_churn["Yes"].max()


# ==========================================
# CREATE SUMMARY TABLE
# ==========================================

summary = pd.DataFrame({

    "Factor": [
        "Contract",
        "Tenure",
        "Monthly Charges",
        "Internet Service",
        "Payment Method",
        "Tech Support",
        "Online Security",
        "Online Backup",
        "Device Protection"
    ],

    "Highest_Risk_Group": [
        contract_highest,
        tenure_highest,
        charge_highest,
        internet_highest,
        payment_highest,
        tech_highest,
        security_highest,
        backup_highest,
        device_highest
    ],

    "Churn_Rate": [
        contract_rate,
        tenure_rate,
        charge_rate,
        internet_rate,
        payment_rate,
        tech_rate,
        security_rate,
        backup_rate,
        device_rate
    ]
})


# ==========================================
# SORT BY CHURN RATE
# ==========================================

summary = summary.sort_values(
    by="Churn_Rate",
    ascending=False
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nEDA SUMMARY:")
print(summary.to_string(index=False))


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n" + "=" * 70)
print("EDA SUMMARY COMPLETED")
print("=" * 70)