import numpy as np
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3],
              [4, 5, 6]]) # 2d numpy array
a
b
b[1, 2] # outputs 6
# dont use [1][2] like java, use [1, 2]
a[[2, 0]] # outputs [3 1]
a < 3 # outputs [True True False]
a[a < 3] # outputs [1 2]
b.size # outputs 6
b.shape # outputs (2, 3)
for row in range(b.shape[0]):
  b[row, :] # outputs [1 2 3]\n[4 5 6]
a.sum() # outputs 6
b.sum() # outputs 21
np.sum(b, axis=1) # outputs [6 15]
b + 4 # outputs [[5 6 7]\n[8 9 10]]
b + a # outputs [[2 4 6]\n[5 7 9]]

# go to labs folder, conda activate env445, jupyter lab
