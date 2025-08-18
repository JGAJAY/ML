import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 1. Generate training data
np.random.seed(42)
X_train = np.random.uniform(1, 10, (20, 2))
y_train = np.random.randint(0, 2, 20)

# 2. Create test grid
x_vals = np.arange(0, 10.1, 0.1)
y_vals = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_vals, y_vals)
X_test = np.c_[xx.ravel(), yy.ravel()]

# 3. Define k values to compare
k_values = [1, 3, 5, 10]

# 4. Plot for each k
plt.figure(figsize=(16, 12))

for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    plt.subplot(2, 2, i+1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.coolwarm, s=10, alpha=0.3, marker='s')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor='k', s=100)
    plt.title(f"k = {k}")
    plt.xlabel("X Feature")
    plt.ylabel("Y Feature")
    plt.grid(True)

plt.suptitle("kNN Class Boundaries for Varying k Values", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
