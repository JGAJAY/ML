import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_excel("rajasthan.xlsx")

# 2. Remove target column if it exists
if 'target' in df.columns:
    df = df.drop('target', axis=1)

# 3. Convert categorical/string columns to numeric (One-Hot Encoding)
df_encoded = pd.get_dummies(df)

# 4. Split into train and test (optional)
X_train, X_test = train_test_split(df_encoded, test_size=0.2, random_state=42)

# 5. Fit KMeans (k=2 as per A4)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train)

# 6. Calculate metrics
silhouette = silhouette_score(X_train, kmeans.labels_)
ch_score = calinski_harabasz_score(X_train, kmeans.labels_)
db_index = davies_bouldin_score(X_train, kmeans.labels_)

# 7. Print results
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")
