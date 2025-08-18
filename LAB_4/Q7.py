import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset with proper encoding
df = pd.read_csv("dataset.csv", encoding='ISO-8859-1')

# Convert target columns to numeric
for col in ['Pre-monsoon_2020 (meters below ground level)', 'Post-monsoon_2020 (meters below ground level)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing values
df = df.dropna(subset=[
    'Pre-monsoon_2020 (meters below ground level)', 
    'Post-monsoon_2020 (meters below ground level)'
])

# Feature matrix (X) and target vector (y)
X = df[['Pre-monsoon_2020 (meters below ground level)', 
        'Post-monsoon_2020 (meters below ground level)']].values
y = np.where(X[:, 0] < 20, 0, 1)  # Binary class

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create kNN model
knn = KNeighborsClassifier()

# Define grid of 'k' values to try
param_grid = {'n_neighbors': list(range(1, 21))}

# Grid search with 5-fold cross-validation
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Best parameters and score
print("Best k value found:", grid.best_params_['n_neighbors'])
print("Best cross-validation accuracy:", round(grid.best_score_ * 100, 2), "%")

# Evaluate on test set
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)

# Metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
