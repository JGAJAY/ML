import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate 20 random 2D points (X, Y) between 1 and 10
X = np.random.uniform(1, 10, (20, 2))

# Randomly assign class labels: 0 or 1
y = np.random.randint(0, 2, 20)

# Plot the points
plt.figure(figsize=(8, 6))
for i in range(20):
    color = 'blue' if y[i] == 0 else 'red'
    label = 'Class 0 (Blue)' if y[i] == 0 else 'Class 1 (Red)'
    plt.scatter(X[i, 0], X[i, 1], c=color, label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xlabel("X Feature")
plt.ylabel("Y Feature")
plt.title("Synthetic 2D Data Points (Q3)")
plt.legend()
plt.grid(True)
plt.show()
