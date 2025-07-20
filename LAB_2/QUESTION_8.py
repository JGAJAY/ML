import pandas as pd
import numpy as np

def load_data(filepath, sheet_name):
    return pd.read_excel(filepath, sheet_name=sheet_name)

def impute_with_constants(df):
    df_imputed = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if df[col].dtype == 'object':
            df_imputed[col].fillna("Unknown", inplace=True)
            print(f"{col}: Imputed with 'Unknown'")

        elif np.issubdtype(df[col].dtype, np.number):
            median_val = df[col].median()
            df_imputed[col].fillna(median_val, inplace=True)
            print(f"{col}: Imputed with Median")

    return df_imputed

def A8_version3():
    filepath = "Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"

    df = load_data(filepath, sheet_name)
    print("🔍 Missing values before imputation:\n", df.isnull().sum())

    df_imputed = impute_with_constants(df)

    print("\n✅ Missing values after constant imputation:\n", df_imputed.isnull().sum())

if _name_ == "_main_":
    A8_version3()