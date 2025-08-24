import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_excel("rajasthan.xlsx", sheet_name="Sheet1")

# Features: drop non-numeric identifiers
X = df.drop(columns=["State", "District"])

# Target: average rainfall in 2022
y = df[[col for col in df.columns if "2022" in col and "R/F" in col]].mean(axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model + param grid
rf = RandomForestRegressor(random_state=42)
param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}

# RandomizedSearchCV
rf_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring="r2",
    random_state=42
)
rf_search.fit(X_train, y_train)

print("Best Parameters for Random Forest:", rf_search.best_params_)
print("Best CV Score:", rf_search.best_score_)
