import pandas as pd

# Load data
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
print("Data Loaded\n", df.head())

# Data types and uniques
print("\nTypes and Unique Counts:")
for c in df.columns:
    print(f"{c}: {df[c].dtype}, Unique: {df[c].nunique()}")

# Encoding hints
print("\nCategorical Encoding:")
for c in df.columns:
    if df[c].dtype == 'object':
        print(f"{c}: One-Hot Encoding")

# Numeric stats
print("\nStats for Numeric Columns:")
nums = df.select_dtypes(include='number')
for c in nums.columns:
    print(f"{c}: Min={df[c].min()}, Max={df[c].max()}, Mean={df[c].mean():.2f}, Std={df[c].std():.2f}")

# Missing data
print("\nMissing Data:")
print(df.isnull().sum())

# IQR-based outliers
print("\nOutliers:")
for c in nums.columns:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    l = q1 - 1.5 * iqr
    u = q3 + 1.5 * iqr
    print(f"{c}: {(df[c] < l).sum() + (df[c] > u).sum()} outliers")
