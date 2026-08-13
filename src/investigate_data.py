import pandas as pd

# Load the raw dataset
file_path = "Data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("TOTAL CHARGES DATA TYPE")
print("=" * 60)
print(df["TotalCharges"].dtype)

print("\n" + "=" * 60)
print("BLANK TOTAL CHARGES")
print("=" * 60)

blank_total_charges = df["TotalCharges"].astype(str).str.strip() == ""

print("Number of blank TotalCharges values:", blank_total_charges.sum())

print("\nRows with blank TotalCharges:")
print(df.loc[blank_total_charges, ["customerID", "tenure", "TotalCharges", "Churn"]])

print("\n" + "=" * 60)
print("CUSTOMERS WITH ZERO TENURE")
print("=" * 60)

zero_tenure = df["tenure"] == 0

print("Number of customers with zero tenure:", zero_tenure.sum())

print("\nZero-tenure customers:")
print(df.loc[zero_tenure, ["customerID", "tenure", "TotalCharges", "Churn"]])