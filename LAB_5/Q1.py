import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_excel("rajasthan.xlsx")  


feature_cols = ["JAN_R/F_2018", "FEB_R/F_2018", "MAR_R/F_2018", "APR_R/F_2018", "MAY_R/F_2018","JUN_R/F_2018", "JUL_R/F_2018", "AUG_R/F_2018", "SEP_R/F_2018", "OCT_R/F_2018", "NOV_R/F_2018", "DEC_R/F_2018", "JAN_R/F_2019", "FEB_R/F_2019", "MAR_R/F_2019", "APR_R/F_2019", "MAY_R/F_2019", "JUN_R/F_2019", "JUL_R/F_2019", "AUG_R/F_2019", "SEP_R/F_2019", "OCT_R/F_2019", "NOV_R/F_2019", "DEC_R/F_2019", "JAN_R/F_2020",]
target_col = "FEB_R/F_2020"


df_clean = df.dropna(subset=feature_cols + [target_col])

# Extract features and target
X_train = df_clean[feature_cols]
y_train = df_clean[target_col]

# Train linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Predict on the training data
y_train_pred = reg.predict(X_train)

# Print regression coefficients
print("Regression coefficients:")
for f, c in zip(feature_cols, reg.coef_):
    print(f"{f}: {c:.4f}")
print(f"Intercept: {reg.intercept_:.4f}")

# Evaluate performance
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)
print(f"Training MSE: {mse:.2f}")
print(f"Training R²: {r2:.2f}")

