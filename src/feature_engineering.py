import pandas as pd


# ==========================================
# STEP 27 — LOAD CLEANED DATA
# ==========================================

file_path = "Data/Processed/cleaned_telco_churn.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print("\nOriginal dataset shape:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nOriginal columns:")
print(df.columns.tolist())

print("\nTarget variable:")
print(df["Churn"].value_counts())


# ==========================================
# STEP 28 — REMOVE CUSTOMER ID
# AND SEPARATE FEATURES AND TARGET
# ==========================================

# Remove customer ID because it is only an identifier
df = df.drop(columns=["customerID"])

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nAfter removing customerID:")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget values:")
print(y.value_counts())


# ==========================================
# STEP 29 — CONVERT TARGET TO NUMERIC
# ==========================================

# Convert:
# No  -> 0
# Yes -> 1

y = y.map({
    "No": 0,
    "Yes": 1
})

print("\nTarget after encoding:")
print(y.value_counts())

print("\nTarget data type:")
print(y.dtype)


# ==========================================
# STEP 30 — IDENTIFY NUMERICAL
# AND CATEGORICAL FEATURES
# ==========================================

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

print("\nNumerical features:")
print(numerical_features)

print("\nNumber of numerical features:")
print(len(numerical_features))

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of categorical features:")
print(len(categorical_features))


# ==========================================
# VALIDATION
# ==========================================

print("\nFeature count check:")
print(
    "Numerical + Categorical =",
    len(numerical_features) + len(categorical_features)
)

print("Total X features =", X.shape[1])

print("\nFeature engineering steps completed successfully!")

# ==========================================
# STEP 31 — ONE-HOT ENCODING
# ==========================================

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

X_encoded = preprocessor.fit_transform(X)

print("\n" + "=" * 60)
print("ONE-HOT ENCODING")
print("=" * 60)

print("\nOriginal feature shape:")
print(X.shape)

print("\nEncoded feature shape:")
print(X_encoded.shape)

print("\nNumber of encoded features:")
print(X_encoded.shape[1])

# ==========================================
# STEP 32 — VALIDATE ENCODED DATA
# ==========================================

print("\n" + "=" * 60)
print("ENCODED DATA VALIDATION")
print("=" * 60)

# Check missing values
print("\nMissing values in encoded data:")
print(pd.DataFrame(X_encoded).isnull().sum().sum())

# Check data type
print("\nEncoded data type:")
print(X_encoded.dtype)

# Check rows
print("\nEncoded rows:", X_encoded.shape[0])

# Check target rows
print("Target rows:", y.shape[0])

# Check whether all values are numeric
print("\nAll encoded values are numeric:")
print(pd.api.types.is_numeric_dtype(X_encoded))

# Display first 5 rows
print("\nFirst 5 rows of encoded data:")
print(X_encoded[:5])

# ==========================================
# STEP 33 — TRAIN / TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining features shape:")
print(X_train.shape)

print("\nTesting features shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)

print("\nTraining churn distribution:")
print(y_train.value_counts())

print("\nTesting churn distribution:")
print(y_test.value_counts())

# ==========================================
# STEP 34 — FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Scale only the numerical columns
# The last 4 columns are our numerical features
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[:, -4:] = scaler.fit_transform(
    X_train[:, -4:]
)

X_test_scaled[:, -4:] = scaler.transform(
    X_test[:, -4:]
)

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

print("\nTraining data shape after scaling:")
print(X_train_scaled.shape)

print("\nTesting data shape after scaling:")
print(X_test_scaled.shape)

print("\nMean of scaled training numerical features:")
print(X_train_scaled[:, -4:].mean(axis=0))

print("\nStandard deviation of scaled training numerical features:")
print(X_train_scaled[:, -4:].std(axis=0))

# ==========================================
# STEP 35 — SAVE PROCESSED ML DATA
# ==========================================

import os

processed_path = "Data/Processed"

# Create folder if it does not exist
os.makedirs(processed_path, exist_ok=True)

# Save training features
pd.DataFrame(X_train_scaled).to_csv(
    f"{processed_path}/X_train.csv",
    index=False
)

# Save testing features
pd.DataFrame(X_test_scaled).to_csv(
    f"{processed_path}/X_test.csv",
    index=False
)

# Save training target
pd.DataFrame(y_train).to_csv(
    f"{processed_path}/y_train.csv",
    index=False
)

# Save testing target
pd.DataFrame(y_test).to_csv(
    f"{processed_path}/y_test.csv",
    index=False
)

print("\n" + "=" * 60)
print("PROCESSED DATA SAVING")
print("=" * 60)

print("\nSaved files:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")

print("\nProcessed data saved successfully!")

# ==========================================
# STEP 53 — ENCODED FEATURE NAMES
# ==========================================

encoded_feature_names = preprocessor.get_feature_names_out()

print("\n" + "=" * 60)
print("ENCODED FEATURE NAMES")
print("=" * 60)

print("\nNumber of encoded feature names:")
print(len(encoded_feature_names))

print("\nFirst 20 encoded feature names:")
print(encoded_feature_names[:20])

# ==========================================
# STEP 70 — SAVE PREPROCESSOR AND SCALER
# ==========================================

import joblib

joblib.dump(
    preprocessor,
    "Data/Processed/preprocessor.pkl"
)

joblib.dump(
    scaler,
    "Data/Processed/scaler.pkl"
)

print("\n" + "=" * 60)
print("PREPROCESSOR AND SCALER SAVING")
print("=" * 60)

print("\nPreprocessor saved:")
print("Data/Processed/preprocessor.pkl")

print("\nScaler saved:")
print("Data/Processed/scaler.pkl")

print("\nPreprocessor and scaler saved successfully!")