import numpy as np
import numpy.random as random

Devious = random.rand(5, 7) * 10

popavg = np.average(Devious, axis=0)
popavg = popavg.reshape(1,7)
diffs = Devious - popavg
sqDiffs = np.square(diffs)
sumSq = np.sum(sqDiffs, axis=0)
divByN = sumSq/Devious.shape[0]
stdDev = np.sqrt(divByN)
print(stdDev)
stdDev2 = np.std(Devious, axis=0)
print(stdDev2)
print("Done")