import pandas as pd

# Load the dataset
file_path = "Data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# 1. Dataset shape
print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Data types
print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

# 3. Missing values
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# 4. Duplicate rows
print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)
print("Number of duplicate rows:", df.duplicated().sum())

# 5. Statistical summary
print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())

# 6. Churn distribution
print("\n" + "=" * 50)
print("CHURN DISTRIBUTION")
print("=" * 50)
print(df["Churn"].value_counts())