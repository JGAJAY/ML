import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(path, feature_cols, label_col, cutoff):
    df = pd.read_csv(path, encoding='latin1')
    df[feature_cols] = df[feature_cols].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    df = df.dropna(subset=feature_cols)
    df['Label'] = df[label_col].apply(lambda x: 1 if x >= cutoff else 0)
    return df

def export_splits(df, feature_cols, label_col='Label', split_ratio=0.3):
    X = df[feature_cols]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=1)
    pd.concat([X_train, y_train], axis=1).to_csv("groundwater_train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv("groundwater_test.csv", index=False)

# Main
columns = [f'Pre-monsoon_{yr} (meters below ground level)' for yr in range(2015, 2023)]
target = 'Pre-monsoon_2022 (meters below ground level)'
df = preprocess_data("dataset.csv", columns, target, 25)
export_splits(df, columns)
print("Data split and saved successfully.")
