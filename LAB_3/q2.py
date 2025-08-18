import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('dataset.csv', encoding='latin1')
feature = 'Pre-monsoon_2015 (meters below ground level)'  


data = pd.to_numeric(df[feature], errors='coerce').dropna()


counts, bins = np.histogram(data, bins=10)

# Show histogram info
print("Histogram counts per bucket:", counts)
print("Bin ranges:", bins)

# Mean & variance
mean_val = np.mean(data)
var_val = np.var(data)

print(f"Variance of {feature}: {var_val:.2f}")
print(f"Mean of {feature}: {mean_val:.2f}")


# Plot
plt.hist(data, bins=10, color='lightblue', edgecolor='black')
plt.title(f"Histogram of {feature}")
plt.xlabel("Meters below ground level")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
