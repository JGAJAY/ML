import numpy as np

def matrix_power(A, m):
    A = np.array(A)
    if A.shape[0] != A.shape[1]:
        return "Input is not a square matrix"
    return np.linalg.matrix_power(A, m)

A = [[1, 2], [3, 4]]
m = 2
print("Matrix A^m:\n", matrix_power(A, m))
