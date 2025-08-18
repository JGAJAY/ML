import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ----------------- Binning -----------------
def binning(data, num_bins=4, method="equal_width"):
    if not pd.api.types.is_numeric_dtype(data):
        return data
    if method == "equal_width":
        return pd.cut(data, bins=num_bins, labels=False, include_lowest=True)
    elif method == "equal_frequency":
        return pd.qcut(data, q=num_bins, labels=False, duplicates="drop")
    else:
        raise ValueError("Invalid method. Use 'equal_width' or 'equal_frequency'.")

# ----------------- Entropy -----------------
def entropy(column):
    probs = column.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9))

# ----------------- Information Gain -----------------
def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = df[df[feature] == v]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy

# ----------------- Build Tree -----------------
def build_tree(df, target, features, binning_method="equal_width", num_bins=4, depth=0, max_depth=3):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if len(features) == 0 or depth == max_depth:
        return df[target].mode()[0]

    processed_features = []
    for f in features:
        if pd.api.types.is_numeric_dtype(df[f]):
            df[f] = binning(df[f], num_bins=num_bins, method=binning_method)
        processed_features.append(f)

    gains = {f: information_gain(df, f, target) for f in processed_features}
    best_feature = max(gains, key=gains.get)

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

# ----------------- Visualize Tree -----------------
def visualize_tree(tree, parent_name='', graph=None):
    if graph is None:
        graph = nx.DiGraph()
    root = list(tree.keys())[0]
    children = tree[root]
    for key, value in children.items():
        child_name = f"{root}={key}"
        if isinstance(value, dict):
            graph.add_edge(parent_name if parent_name else root, child_name)
            visualize_tree(value, parent_name=child_name, graph=graph)
        else:
            leaf_name = f"{child_name}\n(Class {value})"
            graph.add_edge(parent_name if parent_name else root, leaf_name)
    return graph

# ----------------- Main -----------------
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_excel("rajasthan.xlsx")

    # Select target column (example: rainfall in Jan 2018)
    target = "JAN_R/F_2018"
    features = [col for col in df.columns if col != target]

    # Bin target if continuous
    df[target] = binning(df[target], num_bins=4, method="equal_width")

    # Build decision tree
    tree = build_tree(df, target, features, binning_method="equal_width", num_bins=4, max_depth=2)

    print("Decision Tree:", tree)

    # Visualize
    G = visualize_tree(tree)
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color="lightgreen", font_size=9, font_weight="bold")
    plt.title("Decision Tree Visualization", fontsize=18)
    plt.show()
