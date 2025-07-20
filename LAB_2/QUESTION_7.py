import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_first_20_vectors(filepath, sheet_name):
    """Load first 20 rows from the given worksheet."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df.head(20)

def get_binary_columns(df):
    """Return only binary columns (0 or 1 values)."""
    return [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]

def compute_dice_hamming(df):
    """Compute pairwise Dice coefficient and Hamming distance for binary columns."""
    binary_cols = get_binary_columns(df)
    binary_data = df[binary_cols]

    n = binary_data.shape[0]
    dice_matrix = np.zeros((n, n))
    hamming_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vec1 = binary_data.iloc[i]
            vec2 = binary_data.iloc[j]
            f11 = ((vec1 == 1) & (vec2 == 1)).sum()
            f10 = ((vec1 == 1) & (vec2 == 0)).sum()
            f01 = ((vec1 == 0) & (vec2 == 1)).sum()
            dice = (2 * f11) / (2 * f11 + f10 + f01) if (2 * f11 + f10 + f01) > 0 else 0
            hamming = (f10 + f01) / len(vec1)
            dice_matrix[i][j] = dice
            hamming_matrix[i][j] = hamming

    return dice_matrix, hamming_matrix

def plot_heatmap(matrix, title):
    """Display a heatmap for the given matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='YlGnBu', fmt=".2f", square=True)
    plt.title(title)
    plt.xlabel("Observation Index")
    plt.ylabel("Observation Index")
    plt.show()

def A7_version3():
    filepath = "Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"

    df20 = load_first_20_vectors(filepath, sheet_name)

    print(" Calculating Dice similarity and Hamming distance...")
    dice_matrix, hamming_matrix = compute_dice_hamming(df20)

    print(" Plotting heatmaps...")
    plot_heatmap(dice_matrix, "Dice Coefficient Heatmap (First 20 Observations)")
    plot_heatmap(hamming_matrix, "Hamming Distance Heatmap (First 20 Observations)")

if _name_ == "_main_":
    A7_version3()