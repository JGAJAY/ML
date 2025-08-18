import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("rajasthan.xlsx")  


rainfall_cols = [col for col in df.columns if col.endswith("R/F_2020")]
target_col = "JAN_R/F_2021"  

# Remove target column from features
feature_cols = [col for col in rainfall_cols if col != target_col]

# Drop rows where any feature or target is missing
df_clean = df.dropna(subset=feature_cols + [target_col])

# Extract X and y
X = df_clean[feature_cols]
y = df_clean[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
reg = LinearRegression().fit(X_train, y_train)

# Predict on both sets
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Metrics function
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

# Train metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mape = mape(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Test metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mape = mape(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics
print("Train set metrics (using all features):")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"MAPE: {train_mape:.2f}%")
print(f"R²: {train_r2:.2f}")

print("\nTest set metrics (using all features):")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAPE: {test_mape:.2f}%")
print(f"R²: {test_r2:.2f}")

