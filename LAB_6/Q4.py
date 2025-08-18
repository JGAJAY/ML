import pandas as pd

# ---------- Binning Function ----------
def binning(data, num_bins=4, method="equal_width"):
    """
    Convert continuous values into categorical bins.
    
    Parameters:
        data (array-like or Series): Continuous numeric data
        num_bins (int): Number of bins (default = 4)
        method (str): Binning method ("equal_width" or "equal_frequency")
    
    Returns:
        pd.Series: Categorical binned data
    """
    if not pd.api.types.is_numeric_dtype(data):
        raise ValueError("Binning only works for numeric data!")

    if method == "equal_width":
        return pd.cut(data, bins=num_bins, labels=False, include_lowest=True)
    elif method == "equal_frequency":
        return pd.qcut(data, q=num_bins, labels=False, duplicates="drop")
    else:
        raise ValueError("Invalid method. Use 'equal_width' or 'equal_frequency'.")

# ---------- Main Script ----------
if __name__ == "__main__":
    # Load your dataset
    file_path = "rajasthan.xlsx"   # make sure it's in the same folder
    df = pd.read_excel(file_path)

    # Choose any continuous column from dataset
    column = "JAN_R/F_2018"   # you can replace this with any numeric column
    
    print("Original data (first 10):")
    print(df[column].head(10))

    # Apply equal-width binning
    print("\nEqual Width Binning (4 bins):")
    print(binning(df[column], num_bins=4, method="equal_width").head(10))

    # Apply equal-frequency binning
    print("\nEqual Frequency Binning (4 bins):")
    print(binning(df[column], num_bins=4, method="equal_frequency").head(10))
