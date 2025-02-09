"""
Name:

Please:
    - set breakpoints and walk through this lab.
    - verify the dimensions of the arrays
    - verify values with your hand calculations
    - seek to understand, not to just get it done

Lab: Gradient Descent
  - use genFirstDataSet() to get your Cost and GradientDescent functions to work
  - then use genDataSet() to get a random set of data with a given variance (sigma)

ToDo:
    - plot random data (20 values)
    - plot your predicted line given w0 and w1
    - plot the cost curve
    - answer the questions in the lab in the commented area at the end
    - congratulate yourself - you passed a major milestone!

"""
import numpy as np
import matplotlib.pyplot as plt

# Generates a simple test set of data to work with [x0,x1,y]
def genFirstDataSet():
    return np.array([[1, 0, 0],
                     [1, 1, .5],
                     [1, 2, 1],
                     [1, 3, 1.5]])

# Generates a random set of data with a set variance, slope, offset, min and max x
def genDataSet(nElements, sigma, slope, offset, minX, maxX):
    x0 = np.ones(nElements) # x0 = 1
    x1 = np.random.random(nElements) * (maxX - minX) + minX # x1 = random values between min and max
    y = x1 * slope + offset # y values (w1*x1 +w0)
    y += (np.random.random(nElements) * 2 - 1) * sigma # adds variance to y values "real data"

    data = np.column_stack((x0, x1, y)) # combines into single array
    return data

# Cost function: computes the sum of squared errors
def costFunc(x, weights, y):
    predictions = np.dot(x, weights)
    errors = predictions - y
    square = np.pow(errors, 2)
    cost = np.sum(square)
    return cost/(len(x)*2)

# Gradient Descent: computes the gradient of the cost function
def gradDesc(x, weights, y):
    predictions = np.dot(x, weights)
    errors = predictions - y
    gradient = (1 / len(y)) * np.dot(x.transpose(), errors)
    return gradient

# Call data generation functions
#data = genFirstDataSet()
data = genDataSet(20, 2, 2, 0, 0, 20)


# Split data into x (features) and y (labels)
x = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

# Initialize weights
#weights = np.random.randn(2, 1)
weights = np.array([0, 1.5]).reshape(2, 1)

# Initialize learning rate and number of iterations
LR = 0.001
maxIter = 50

# Create array to store costs per iteration
costArray = []
print(costFunc(x,weights, y))
for i in range(maxIter):
    cost = costFunc(x, weights, y)
    costArray.append(cost)

    gradient = gradDesc(x, weights, y)

    weights = weights - LR * gradient

costArray = np.array(costArray)
# todo: plot the best fit line
'''
  - use np.argmin() to find the index of the minimum value in x
  - use np.argmax() to find the index of the maximum value in x
  - create an array with the min and max x values
  - create an array with the predicted y values: 2 x values * weights 
'''
x_min_index = np.argmin(x[:, 1])
x_max_index = np.argmax(x[:, 1])
x_min = x[x_min_index, 1]
x_max = x[x_max_index, 1]
y_min = np.dot([1,x_min], weights)
y_max = np.dot([1,x_max], weights)

# plot data
fig1 = plt.figure()
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.plot(x[:, 1], y, 'rx')
ax1.plot([x_min, x_max], [y_min, y_max], color='blue')
ax1.set(title='Data with Best Fit Line', xlabel='X values', ylabel='Y values')
plt.show()


# plot cost
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(maxIter), costArray, color='blue')
ax2.set(title='Cost vs. Iterations', xlabel='iterations', ylabel='cost')
plt.show()

# todo: answer analysis questions in the lab here.
'''
Compare and Contrast

1) Increase the learning rate. At what point does the model fail to train?
Fails at 0.019
2) Decrease the learning rate.  What happens to the Cost Function graph?
The cost function turns more linear(without increasing iteration), generally look like a rectangle with increased iteration)
3) Display the final weights and compare with the expected values.  What can 
    you change to make your model more accurate?

'''
print(weights)