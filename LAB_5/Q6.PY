import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("rajasthan.xlsx")

# Keep only numeric columns
X_train = df.select_dtypes(include=['float64', 'int64']).fillna(0)

# Lists to store metrics
k_values = range(2, 11)
inertias = []
silhouette_scores = []
ch_scores = []
db_scores = []

# Loop through k values
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)

    labels = kmeans.labels_
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_train, labels))
    ch_scores.append(calinski_harabasz_score(X_train, labels))
    db_scores.append(davies_bouldin_score(X_train, labels))

# Plot all metrics against k
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Inertia vs k')

plt.subplot(2, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', color='green')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')

plt.subplot(2, 2, 3)
plt.plot(k_values, ch_scores, marker='o', color='orange')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Calinski-Harabasz Score')
plt.title('CH Score vs k')

plt.subplot(2, 2, 4)
plt.plot(k_values, db_scores, marker='o', color='red')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.title('DB Index vs k')

plt.tight_layout()
plt.show()
