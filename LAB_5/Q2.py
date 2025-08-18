import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("rajasthan.xlsx")  


feature_cols = ["JAN_R/F_2018", "FEB_R/F_2018", "MAR_R/F_2018", "APR_R/F_2018", "MAY_R/F_2018","JUN_R/F_2018", "JUL_R/F_2018", "AUG_R/F_2018", "SEP_R/F_2018", "OCT_R/F_2018", "NOV_R/F_2018", "DEC_R/F_2018", "JAN_R/F_2019", "FEB_R/F_2019", "MAR_R/F_2019", "APR_R/F_2019", "MAY_R/F_2019", "JUN_R/F_2019", "JUL_R/F_2019", "AUG_R/F_2019", "SEP_R/F_2019", "OCT_R/F_2019", "NOV_R/F_2019", "DEC_R/F_2019", "JAN_R/F_2020"]
target_col = "FEB_R/F_2020"


df_clean = df.dropna(subset=feature_cols + [target_col])

# Extract features and target
X = df_clean[feature_cols]
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Make predictions
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Define metrics functions
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100

# Calculate metrics for train set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mape = mape(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate metrics for test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mape = mape(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics
print("Train set metrics:")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"MAPE: {train_mape:.2f}%")
print(f"R²: {train_r2:.2f}")

print("\nTest set metrics:")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAPE: {test_mape:.2f}%")
print(f"R²: {test_r2:.2f}")


