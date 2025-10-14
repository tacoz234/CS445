import numpy as np
#1
B = np.array([[1, 2, -3],
              [3, 4, -1]])

A = np.array([[2, -5, 1],
              [1, 4, 5],
              [2, -1, 6]])

y = np.array([2, -4, 1])

z = np.array([-15, -8, -22])

# 1a
print("question 1a:", B @ A)
# 1b
print("question 1b:", A @ B.T)
# 1c
print("question 1c:", A @ y)
# 1d
print("question 1d:", y.T @ z)
# 1e
print("question 1e:", y @ z.T)
# 2
A = np.array([[1, 2],
              [3, 0]])

b = np.array([4, 6])

# 2a
print("question 2a:", np.linalg.inv(A))
# 2b
print("question 2b:", np.linalg.inv(A) @ b)
