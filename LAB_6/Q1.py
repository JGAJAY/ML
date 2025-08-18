import pandas as pd
import numpy as np

def equal_width_binning(data, num_bins=4):
    """
    Perform equal-width binning on continuous data.
    Converts numeric values into discrete categories.
    """
    return pd.cut(data, bins=num_bins, labels=False, include_lowest=True)

def calculate_entropy(data):
    """
    Calculate Shannon entropy for a discrete/categorical dataset.
    Formula: H = - Σ p_i * log2(p_i)
    """
    series = pd.Series(data).dropna()
    probabilities = series.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities))

# Load the dataset
file_path = "rajasthan.xlsx"  
df = pd.read_excel(file_path)

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Store entropy results
entropy_results = {}

for col in numeric_cols:
    # Apply equal-width binning
    binned_data = equal_width_binning(df[col], num_bins=4)
    # Calculate entropy
    entropy_results[col] = calculate_entropy(binned_data)

# Convert to DataFrame for better display
entropy_df = pd.DataFrame(list(entropy_results.items()), columns=["Column", "Entropy"])

# Show the results
print(entropy_df)
