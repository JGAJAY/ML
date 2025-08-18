import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "rajasthan.xlsx"
df = pd.read_excel(file_path)

# Select 2 features (example: Jan & Feb rainfall 2018)
X = df[["JAN_R/F_2018", "FEB_R/F_2018"]].values

# Target column (example: MAR_R/F_2018 rainfall → convert to categorical using binning)
y_continuous = df["MAR_R/F_2018"].values.reshape(-1, 1)

# Bin the target into 4 categories
binning = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="uniform")
y = binning.fit_transform(y_continuous).astype(int).ravel()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# --- Plot Decision Boundary ---
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Accent)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Accent)
plt.xlabel("JAN_R/F_2018")
plt.ylabel("FEB_R/F_2018")
plt.title("Decision Boundary of Decision Tree")
plt.show()
