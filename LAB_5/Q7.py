import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("rajasthan.xlsx")

# Keep only numeric columns
X_train = df.select_dtypes(include=['float64', 'int64']).fillna(0)

# Elbow method
distortions = []
K_range = range(2, 20)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    distortions.append(kmeans.inertia_)

# Plot
plt.plot(K_range, distortions, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
