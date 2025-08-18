import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 1. Generate training data (same as Q3)
np.random.seed(42)
X_train = np.random.uniform(1, 10, (20, 2))
y_train = np.random.randint(0, 2, 20)

# 2. Generate test grid (100 x 100 = 10,000 points)
x_test_vals = np.arange(0, 10.1, 0.1)
y_test_vals = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_test_vals, y_test_vals)
X_test = np.c_[xx.ravel(), yy.ravel()]  # shape: (10000, 2)

# 3. Train kNN (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4. Predict on test data
y_pred = knn.predict(X_test)

# 5. Plot decision regions using scatter
plt.figure(figsize=(10, 8))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.coolwarm, alpha=0.4, s=10, marker='s')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor='k', s=100, label="Training Data")

plt.title("kNN Classification (k=3) with Class Boundaries")
plt.xlabel("X Feature")
plt.ylabel("Y Feature")
plt.grid(True)
plt.legend()
plt.show()
