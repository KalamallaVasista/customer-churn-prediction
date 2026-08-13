import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = "Data/Processed/cleaned_telco_churn.csv"
df = pd.read_csv(file_path)

# Display churn counts
print("=" * 50)
print("CHURN DISTRIBUTION")
print("=" * 50)
print(df["Churn"].value_counts())

# Calculate churn percentages
churn_percentage = df["Churn"].value_counts(normalize=True) * 100

print("\nChurn percentages:")
print(churn_percentage)

# Create churn count plot
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ==========================================
# CHURN BY CONTRACT TYPE
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY CONTRACT TYPE")
print("=" * 50)

# Count customers by contract and churn
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

print("\nCustomer counts:")
print(contract_churn)

# Calculate churn percentage within each contract type
contract_churn_rate = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by contract type:")
print(contract_churn_rate)

# Create visualization
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ==========================================
# CHURN BY TENURE GROUP
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY TENURE GROUP")
print("=" * 50)

# Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
)

# Calculate churn percentage by tenure group
tenure_churn_rate = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by tenure group:")
print(tenure_churn_rate)

# Create visualization
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="TenureGroup",
    hue="Churn"
)

plt.title("Customer Churn by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# CHURN BY MONTHLY CHARGES
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY MONTHLY CHARGES")
print("=" * 50)

# Create monthly charge groups using quartiles
df["MonthlyChargeGroup"] = pd.qcut(
    df["MonthlyCharges"],
    q=4,
    labels=["Low", "Medium-Low", "Medium-High", "High"]
)

# Calculate churn percentage
monthly_charge_churn = pd.crosstab(
    df["MonthlyChargeGroup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by monthly charge group:")
print(monthly_charge_churn)

# Create visualization
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="MonthlyChargeGroup",
    hue="Churn"
)

plt.title("Customer Churn by Monthly Charges")
plt.xlabel("Monthly Charge Group")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# CHURN BY INTERNET SERVICE
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY INTERNET SERVICE")
print("=" * 50)

# Count customers by internet service and churn
internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"]
)

print("\nCustomer counts:")
print(internet_churn)

# Calculate churn percentage
internet_churn_rate = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by internet service:")
print(internet_churn_rate)

# Create visualization
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ==========================================
# CHURN BY PAYMENT METHOD
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY PAYMENT METHOD")
print("=" * 50)

# Count customers by payment method and churn
payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"]
)

print("\nCustomer counts:")
print(payment_churn)

# Calculate churn percentage
payment_churn_rate = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by payment method:")
print(payment_churn_rate)

# Create visualization
plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()

# ==========================================
# CHURN BY TECH SUPPORT
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY TECH SUPPORT")
print("=" * 50)

# Count customers by tech support and churn
tech_support_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"]
)

print("\nCustomer counts:")
print(tech_support_churn)

# Calculate churn percentage
tech_support_churn_rate = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by tech support:")
print(tech_support_churn_rate)

# Create visualization
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="TechSupport",
    hue="Churn"
)

plt.title("Customer Churn by Tech Support")
plt.xlabel("Tech Support")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ==========================================
# CHURN BY ONLINE SECURITY
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY ONLINE SECURITY")
print("=" * 50)

# Count customers by online security and churn
security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"]
)

print("\nCustomer counts:")
print(security_churn)

# Calculate churn percentage
security_churn_rate = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by online security:")
print(security_churn_rate)

# Create visualization
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="OnlineSecurity",
    hue="Churn"
)

plt.title("Customer Churn by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ==========================================
# CHURN BY ONLINE BACKUP
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY ONLINE BACKUP")
print("=" * 50)

# Count customers by online backup and churn
backup_churn = pd.crosstab(
    df["OnlineBackup"],
    df["Churn"]
)

print("\nCustomer counts:")
print(backup_churn)

# Calculate churn percentage
backup_churn_rate = pd.crosstab(
    df["OnlineBackup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by online backup:")
print(backup_churn_rate)

# Create visualization
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="OnlineBackup",
    hue="Churn"
)

plt.title("Customer Churn by Online Backup")
plt.xlabel("Online Backup")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ==========================================
# CHURN BY DEVICE PROTECTION
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY DEVICE PROTECTION")
print("=" * 50)

# Count customers by device protection and churn
device_churn = pd.crosstab(
    df["DeviceProtection"],
    df["Churn"]
)

print("\nCustomer counts:")
print(device_churn)

# Calculate churn percentage
device_churn_rate = pd.crosstab(
    df["DeviceProtection"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by device protection:")
print(device_churn_rate)

# Create visualization
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="DeviceProtection",
    hue="Churn"
)

plt.title("Customer Churn by Device Protection")
plt.xlabel("Device Protection")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ==========================================
# CHURN BY STREAMING TV
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY STREAMING TV")
print("=" * 50)

streaming_tv_churn = pd.crosstab(
    df["StreamingTV"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by Streaming TV:")
print(streaming_tv_churn)

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="StreamingTV",
    hue="Churn"
)

plt.title("Customer Churn by Streaming TV")
plt.xlabel("Streaming TV")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# CHURN BY STREAMING MOVIES
# ==========================================

print("\n" + "=" * 50)
print("CHURN BY STREAMING MOVIES")
print("=" * 50)

streaming_movies_churn = pd.crosstab(
    df["StreamingMovies"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by Streaming Movies:")
print(streaming_movies_churn)

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="StreamingMovies",
    hue="Churn"
)

plt.title("Customer Churn by Streaming Movies")
plt.xlabel("Streaming Movies")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()