import pandas as pd

# Load the cleaned dataset
file_path = "Data/Processed/cleaned_telco_churn.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("CLEANED DATASET VALIDATION")
print("=" * 60)

# 1. Shape
print("\nDataset shape:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. TotalCharges data type
print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

# 3. Missing values
print("\nMissing values:")
print(df.isnull().sum().sum())

# 4. Blank TotalCharges
print("\nBlank TotalCharges:")
print((df["TotalCharges"].astype(str).str.strip() == "").sum())

# 5. Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# 6. Churn distribution
print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\n" + "=" * 60)
print("DATA VALIDATION COMPLETED")
print("=" * 60)