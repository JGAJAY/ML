import pandas as pd
import numpy as np

# ----------------- Binning Function -----------------
def binning(data, num_bins=4, method="equal_width"):
    """Convert continuous values into categorical bins."""
    if not pd.api.types.is_numeric_dtype(data):
        return data  # Return unchanged if already categorical
    
    if method == "equal_width":
        return pd.cut(data, bins=num_bins, labels=False, include_lowest=True)
    elif method == "equal_frequency":
        return pd.qcut(data, q=num_bins, labels=False, duplicates="drop")
    else:
        raise ValueError("Invalid method. Use 'equal_width' or 'equal_frequency'.")

# ----------------- Entropy Function -----------------
def entropy(column):
    """Calculate entropy for a pandas Series."""
    probs = column.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9))  # add small value to avoid log(0)

# ----------------- Information Gain -----------------
def information_gain(df, feature, target):
    """Compute Information Gain of a feature with respect to target."""
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0

    for v in values:
        subset = df[df[feature] == v]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])

    return total_entropy - weighted_entropy

# ----------------- Decision Tree Builder -----------------
def build_tree(df, target, features, binning_method="equal_width", num_bins=4, depth=0, max_depth=5):
    """
    Recursively build a decision tree.
    """
    # If all values of target are same -> leaf node
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    
    # If no features left or depth reached -> majority class
    if len(features) == 0 or depth == max_depth:
        return df[target].mode()[0]

    # Apply binning to numeric features
    processed_features = []
    for f in features:
        if pd.api.types.is_numeric_dtype(df[f]):
            df[f] = binning(df[f], num_bins=num_bins, method=binning_method)
        processed_features.append(f)

    # Select feature with highest Information Gain
    gains = {f: information_gain(df, f, target) for f in processed_features}
    best_feature = max(gains, key=gains.get)

    # Build subtree
    tree = {best_feature: {}}
    for value in df[best_feature].unique():
        subset = df[df[best_feature] == value]
        if subset.empty:
            tree[best_feature][value] = df[target].mode()[0]
        else:
            remaining_features = [f for f in processed_features if f != best_feature]
            tree[best_feature][value] = build_tree(subset, target, remaining_features,
                                                   binning_method, num_bins, depth+1, max_depth)
    return tree

# ----------------- Example Usage -----------------
if __name__ == "__main__":
    # Load dataset
    df = pd.read_excel("rajasthan.xlsx")

    # Example: Predict JAN_R/F_2018 using District and other features
    target = "JAN_R/F_2018"
    features = [col for col in df.columns if col != target]

    # Apply binning to target (if continuous)
    df[target] = binning(df[target], num_bins=4, method="equal_width")

    # Build tree
    tree = build_tree(df, target, features, binning_method="equal_width", num_bins=4, max_depth=3)

    print("\nDecision Tree:")
    print(tree)
