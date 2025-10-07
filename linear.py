import numpy as np

B = np.array([[1, 2, -3],
    [3, 4, -1]])
print(B)

A = np.array([[2, -5, 1],
    [1, 4, 5],
    [2, -1, 6]])
print(A)

y = np.array([2, -4, 1])

z = np.array([-15, -8, -22])

print(B @ A)
print(A @ B.T)
print(A @ y)
print(y.T @ z)
print(y @ z.T)