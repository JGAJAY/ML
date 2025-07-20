import pandas as pd
import numpy as np

def load_data(filepath, sheet_name):
    return pd.read_excel(filepath, sheet_name=sheet_name)

def identify_scaling_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if df[col].nunique() > 2]

def manual_min_max_scale(df, columns):
    df_scaled = df.copy()
    methods_used = {}

    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min == col_max:
            methods_used[col] = "Skipped (constant value)"
            continue

        df_scaled[col] = (df[col] - col_min) / (col_max - col_min)
        methods_used[col] = "Manual Min-Max Scaling"

    return df_scaled, methods_used

def A9_version3():
    filepath = "Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"

    df = load_data(filepath, sheet_name)

    scaling_cols = identify_scaling_columns(df)
    print(" Columns selected for manual min-max scaling:", scaling_cols)

    df_scaled, methods = manual_min_max_scale(df, scaling_cols)

    print("\nNormalization Methods Used:")
    for col, method in methods.items():
        print(f"{col}: {method}")

    return df_scaled

if _name_ == "_main_":
    normalized_df = A9_version3()