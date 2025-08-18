import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv")

# Select relevant feature columns
features = ['GWL (in Mtr)', 'Latitude', 'Longitude']
data = data[features].dropna().reset_index(drop=True)

# Select two data points
vec_a = data.loc[3].values
vec_b = data.loc[6].values

# Calculate Minkowski distances from r = 1 to 10
r_range = range(1, 11)
minkowski_vals = []

for p in r_range:
    distance = np.power(np.sum(np.abs(vec_a - vec_b) ** p), 1 / p)
    minkowski_vals.append(distance)

# Plot
plt.plot(r_range, minkowski_vals, marker='o', color='teal')
plt.xlabel("Minkowski Parameter r")
plt.ylabel("Distance between vec_a and vec_b")
plt.title("Minkowski Distance vs r")
plt.grid(True)
plt.show()
