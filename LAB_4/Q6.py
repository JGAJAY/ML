import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset with proper encoding
df = pd.read_csv("dataset.csv", encoding='ISO-8859-1')

# Replace non-numeric values like 'Dry', 'Filled up', etc.
for col in ['Pre-monsoon_2020 (meters below ground level)', 'Post-monsoon_2020 (meters below ground level)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing data
df = df.dropna(subset=[
    'Pre-monsoon_2020 (meters below ground level)', 
    'Post-monsoon_2020 (meters below ground level)'
])

# Select two features
X = df[['Pre-monsoon_2020 (meters below ground level)', 
        'Post-monsoon_2020 (meters below ground level)']].values

# Create binary class based on Pre-monsoon_2020 values
y = np.where(X[:, 0] < 20, 0, 1)

# Scatter plot of training data
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label='Class 1')
plt.xlabel("Pre-monsoon 2020 (m)")
plt.ylabel("Post-monsoon 2020 (m)")
plt.title("Training Data (Q6)")
plt.legend()
plt.grid(True)
plt.show()

# Generate test grid
x_range = np.arange(X[:, 0].min()-1, X[:, 0].max()+1, 0.1)
y_range = np.arange(X[:, 1].min()-1, X[:, 1].max()+1, 0.1)
xx, yy = np.meshgrid(x_range, y_range)
X_test = np.c_[xx.ravel(), yy.ravel()]

# Train kNN (k=3) and classify test points
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
y_pred = knn.predict(X_test)

# Plot predicted class regions
plt.figure(figsize=(10, 8))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.coolwarm, s=10, alpha=0.2)
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label='Class 0 (Blue)')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label='Class 1 (Red)')
plt.xlabel("Pre-monsoon 2020 (m)")
plt.ylabel("Post-monsoon 2020 (m)")
plt.title("kNN Decision Regions (Project Data)")
plt.legend()
plt.grid(True)
plt.show()
