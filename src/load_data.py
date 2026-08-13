import pandas as pd

# Path to the raw dataset
file_path = "Data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Display basic information
print("Dataset loaded successfully!")

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())