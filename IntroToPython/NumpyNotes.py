import numpy as np

matrixA = np.array([1,2], [3,4])
matrixB = np.array([5,6])
matrixB.shape(2,1)

sum = matrixA+matrixB
newSum = sum+7

bias = np.zeros(sum.shape[0])
sum = np.concatenate((bias, sum), axis=1)

C = np.arange(10,50)
C = C.reshape(4,10)

C1 = C[:, 0]
C2= C[0 ,:]
dims = C2.shape()
C2 = C2.reshape(1,dims[0])

nAvg = np.average(C2)
nMean = C2.mean()

avg1 = np.average(C, axix=0)
avg2 = np.average(C, axis=1)

print("Termino")