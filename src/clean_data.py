import pandas as pd

# Load the original raw dataset
file_path = "Data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("STARTING DATA CLEANING")
print("=" * 60)

# Check blank TotalCharges values
blank_total_charges = df["TotalCharges"].astype(str).str.strip() == ""

print("Blank TotalCharges before cleaning:", blank_total_charges.sum())

# Replace blank TotalCharges with 0
df.loc[blank_total_charges, "TotalCharges"] = "0"

# Convert TotalCharges from string to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

print("TotalCharges data type after conversion:", df["TotalCharges"].dtype)

print("\nBlank TotalCharges after cleaning:",
      df["TotalCharges"].isna().sum())

print("\nData cleaning completed successfully!")

# Save the cleaned dataset
output_path = "Data/Processed/cleaned_telco_churn.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved to:", output_path)