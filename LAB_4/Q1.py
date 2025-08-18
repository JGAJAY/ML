import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load data
df = pd.read_csv('dataset.csv', encoding='ISO-8859-1')

# Column name
col_name = 'Pre-monsoon_2022 (meters below ground level)'

# Convert to numeric (invalid parsing will become NaN)
df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

# Drop rows where conversion failed (i.e., values like 'Dry', 'Filled up', etc.)
df = df.dropna(subset=[col_name])

# Classification target: 1 if >20, else 0
df['Target'] = df[col_name].apply(lambda x: 1 if x > 20 else 0)

# Features to use
features = ['Latitude', 'Longitude', 'Well_Depth (meters)']
X = df[features]
y = df['Target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Results
print("Train Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
