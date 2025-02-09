"""
IntroNumpyLab.py

Google the NumPy methods to find examples on how to implement each task.  Add code as you progress through this lab
and view the data with PyCharm’s debugger to verify the results.
Set a breakpoint at line 14 (where the matrix X is created) and step through the program.
"""

import numpy as np
import numpy.random as random

#1 create a 20x2 array with random values
X = random.rand(20,2)

#2 set a variable, row, to the number of rows
row = X.shape[0]

#3 multipy the first column by 20 - multipy by a scalar
X[:, 0] = X[:, 0] * 20

#3 multipy the 2nd column by 1000
X[:, 1] = X[:, 1] * 1000

#4 calculate the minimum of column 0
min_col0 = np.min(X[:, 0])

#4 calculate the max of column 1
max_col1 = np.max(X[:, 1])

#5 print the max and min:    "min of col 0: xyz,  max of col 1: abc"
print(f"min of col 0: {min_col0}, max of col 1: {max_col1}")

#6 calculate the average of the 1st column
avg_col0 = np.mean(X[:, 0])

#6 calculate the average of both columns  => array of 2 elements
avg_cols = np.mean(X, axis=0)

#7 determine the number of rows and columns in the matrix X
num_rows, num_cols = X.shape

#8 create a (rows x 1) np array of all zeros using np.zeros
zeros_column = np.zeros((num_rows, 1))

#9 add that np array of all zeros as a third column to X using np.hstack() -- make sure you specify a tuple
X = np.hstack((X, zeros_column))

#10 add column 0 and 1 of X into column 2
X[:, 2] = X[:, 0] + X[:, 1]


#11 slicing: store a section of rows or columns into a numpy array
# store rows 3, 4, and 5 into sliceRowsX
sliceRowsX = X[3:6, :]

#12 store columns 0 and 2 into sliceColsX
sliceColsX = X[:, [0, 2]]

#Use the debugger to validate your results.
print("Hello Numpy!")