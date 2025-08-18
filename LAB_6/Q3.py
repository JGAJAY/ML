import pandas as pd
import numpy as np

# ---------- Equal Width Binning ----------
def equal_width_binning(data, num_bins=4):
    """Convert numeric values into discrete categories."""
    if pd.api.types.is_numeric_dtype(data):
        return pd.cut(data, bins=num_bins, labels=False, include_lowest=True)
    return data

# ---------- Entropy Calculation ----------
def entropy(values):
    """Calculate entropy for a categorical variable."""
    probabilities = values.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # avoid log(0)

# ---------- Information Gain Calculation ----------
def information_gain(df, feature, target):
    """Calculate information gain of a feature w.r.t target."""
    total_entropy = entropy(df[target])
    weighted_entropy = 0
    for val in df[feature].dropna().unique():
        subset = df[df[feature] == val][target]
        weight = len(subset) / len(df)
        weighted_entropy += weight * entropy(subset)
    return total_entropy - weighted_entropy

# ---------- Root Node Detection ----------
def find_root_node(df, target, num_bins=4):
    """Find the feature with maximum information gain."""
    df_binned = df.copy()
    for col in df_binned.columns:
        if col != target:
            df_binned[col] = equal_width_binning(df_binned[col], num_bins=num_bins)
    gains = {col: information_gain(df_binned, col, target)
             for col in df_binned.columns if col != target}
    root_feature = max(gains, key=gains.get)
    return root_feature, gains

# ---------- Main Script ----------
if __name__ == "__main__":
    file_path = "rajasthan.xlsx"  
    df = pd.read_excel(file_path)
    
    # Choose target column
    target_column = "JAN_R/F_2018"
    df[target_column] = equal_width_binning(df[target_column], num_bins=4)
    
    # Find root node
    root, gains = find_root_node(df, target=target_column, num_bins=4)
    
    print("Information Gain for each feature:")
    for feature, gain in gains.items():
        print(f"{feature}: {gain:.6f}")
    
    print("\nRoot Node Feature:", root)
