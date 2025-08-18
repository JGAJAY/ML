import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv", encoding='ISO-8859-1')

# Choose class and feature columns
class_column = 'District Name'
feature_column = 'GWL (in Mtr)'

# Drop rows with missing values
df = df[[class_column, feature_column]].dropna()

# Choose two districts
class1 = 'Pune'
class2 = 'Bengaluru Rural'

# Extract data
data1 = df[df[class_column] == class1][feature_column].values
data2 = df[df[class_column] == class2][feature_column].values

# Compute stats
centroid1 = np.mean(data1)
centroid2 = np.mean(data2)
spread1 = np.std(data1)
spread2 = np.std(data2)
interclass_distance = norm(centroid1 - centroid2)

# Print results
print(f"Centroid of {class1}: {centroid1:.3f}")
print(f"Centroid of {class2}: {centroid2:.3f}")
print(f"Spread of {class1}: {spread1:.3f}")
print(f"Spread of {class2}: {spread2:.3f}")
print(f"Interclass distance: {interclass_distance:.3f}")

# Plot
plt.hist(data1, alpha=0.5, label=class1)
plt.hist(data2, alpha=0.5, label=class2)
plt.xlabel("Ground Water Level (m)")
plt.ylabel("Frequency")
plt.title("District-wise Ground Water Level Distribution")
plt.legend()
plt.show()
