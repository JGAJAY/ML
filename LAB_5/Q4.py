import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("rajasthan.xlsx")  # Replace with your actual file name

# Select numerical columns only
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Remove target variable used in regression (example: JUN_R/F_2018)
target_col = "JUN_R/F_2018"
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

# Prepare data for clustering
X = df[numeric_cols].dropna()

# Perform K-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

# Cluster labels for each row
df["Cluster"] = kmeans.labels_

# Cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# First 10 rows with cluster assignments
print("\nFirst 10 rows with cluster labels:")
print(df.head(10)[["Cluster"] + numeric_cols])

