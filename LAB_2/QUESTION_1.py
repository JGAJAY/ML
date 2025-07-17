import numpy as np

A = np.array([
    [20, 6, 2],
    [16, 3, 6],
    [27, 6, 2],
    [19, 1, 2],
    [24, 4, 2],
    [22, 1, 5],
    [15, 4, 2],
    [18, 4, 2],
    [21, 1, 4],
    [16, 2, 4]
])

C = np.array([
    [386],
    [289],
    [393],
    [110],
    [280],
    [167],
    [271],
    [274],
    [148],
    [198]
])

dimensionality = A.shape[1]
num_vectors = A.shape[0]

U, S, Vt = np.linalg.svd(A)
rank = sum(s > 1e-10 for s in S)

A_pinv = np.linalg.pinv(A)
X = A_pinv @ C

print("---- RESULTS ----")
print(f"Dimensionality of vector space : {dimensionality}")
print(f"Number of vectors              : {num_vectors}")
print(f"Rank of Matrix A               : {rank}")
print("\nEstimated cost of each product:")
print(f"Cost of 1 Candy       : Rs {X[0][0]:.2f}")
print(f"Cost of 1 Kg Mango    : Rs {X[1][0]:.2f}")
print(f"Cost of 1 Milk Packet : Rs {X[2][0]:.2f}")
