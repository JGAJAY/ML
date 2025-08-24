import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_excel("rajasthan.xlsx", sheet_name="Sheet1")

# Features and target
X = df.drop(columns=["State", "District"])
y = df[[col for col in df.columns if "2022" in col and "R/F" in col]].mean(axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# TUNE EACH MODEL SEPARATELY
# -----------------------------
# Random Forest
rf = RandomForestRegressor(random_state=42)
rf_params = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None]}
rf_search = RandomizedSearchCV(rf, rf_params, n_iter=5, cv=3, scoring="r2", random_state=42)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_

# Gradient Boosting
gb = GradientBoostingRegressor(random_state=42)
gb_params = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
gb_search = RandomizedSearchCV(gb, gb_params, n_iter=5, cv=3, scoring="r2", random_state=42)
gb_search.fit(X_train, y_train)
gb_best = gb_search.best_estimator_

# XGBoost
xgb = XGBRegressor(random_state=42, verbosity=0)
xgb_params = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=5, cv=3, scoring="r2", random_state=42)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_

# SVR
svr = SVR()
svr_params = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf", "linear"]}
svr_search = RandomizedSearchCV(svr, svr_params, n_iter=5, cv=3, scoring="r2", random_state=42)
svr_search.fit(X_train, y_train)
svr_best = svr_search.best_estimator_

# -----------------------------
# EVALUATE ALL MODELS
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": rf_best,
    "Gradient Boosting": gb_best,
    "XGBoost": xgb_best,
    "SVR": svr_best
}

results = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        "Model": name,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std": cv_scores.std(),
        "Test_MSE": mean_squared_error(y_test, y_pred),
        "Test_R2": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print(results_df)
