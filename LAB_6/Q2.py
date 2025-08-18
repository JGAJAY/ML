import pandas as pd
import numpy as np

def equal_width_binning(data, num_bins=4):
    """
    Perform equal-width binning on continuous data.
    Converts numeric values into discrete categories.
    """
    return pd.cut(data, bins=num_bins, labels=False, include_lowest=True)

def calculate_gini_index(data):
    """
    Calculate Gini index for a discrete/categorical dataset.
    Formula: Gini = 1 - Σ (p_j)^2
    """
    series = pd.Series(data).dropna()
    probabilities = series.value_counts(normalize=True)
    return 1 - np.sum(probabilities ** 2)

file_path = "rajasthan.xlsx"  
df = pd.read_excel(file_path)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Store Gini results
gini_results = {}

for col in numeric_cols:
    binned_data = equal_width_binning(df[col], num_bins=4)
    gini_results[col] = calculate_gini_index(binned_data)

# Convert to DataFrame for better display
gini_df = pd.DataFrame(list(gini_results.items()), columns=["Column", "Gini_Index"])

# Show the results
print(gini_df)
