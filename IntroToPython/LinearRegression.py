"""
Name: Ethan Llontop

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

# Compute the cost using Sum of Squared Errors (SSE)
def costFunc(x, weights, y):

    predictions = np.dot(x, weights)
    errors = predictions - y
    square = np.pow(errors, 2)
    cost = np.sum(square)
    return cost/(len(x)*2)


# Compute the gradient for the cost function
def gradDesc(x, weights, y):
    predictions = np.dot(x, weights)
    errors = predictions - y
    gradient = (1 / len(y)) * np.dot(x.transpose(), errors)
    return gradient


# Call data generation function
#data = genFirstDataSet()
data = genDataSet(20, 2, 2, 0, 0, 20)

# Split data into x (features) and y (labels)
x = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

# Initialize weights
#weights = np.random.randn(2, 1)
weights = np.array([0, 1.5]).reshape(2, 1)

# Initialize learning rate and number of iterations
LR = 0.005  # Change to 0.011 for decent linear regression
maxIter = 100

# Create array to store costs per iteration
costArray = []

# Main loop for gradient descent
for i in range(maxIter):
    cost = costFunc(x, weights, y)
    costArray.append(cost)

    gradient = gradDesc(x, weights, y)
    weights = weights - LR * gradient

# Find the index of the min and max x values
x_min_index = np.argmin(x[:, 1])
x_max_index = np.argmax(x[:, 1])
x_min = x[x_min_index, 1]
x_max = x[x_max_index, 1]

# Calculate corresponding predicted y values for min and max x values
y_min = np.dot([1, x_min], weights)
y_max = np.dot([1, x_max], weights)

# Plot the data and the best fit line
fig1 = plt.figure()
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.plot(x[:, 1], y, 'rx', label='Data Points')  # Original data points
ax1.plot([x_min, x_max], [y_min, y_max], color='blue', label='Best Fit Line')  # Best fit line
ax1.set(title='Data with Best Fit Line', xlabel='X values', ylabel='Y values')
ax1.legend()
plt.show()

# Plot the cost function over iterations
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(maxIter), costArray, color='blue')
ax2.set(title='Cost vs. Iterations', xlabel='Iterations', ylabel='Cost')
plt.show()

'''
1) Increase the learning rate. At what point does the model fail to train?
   - As the learning rate increases beyond a certain point (consistently around 0.011), the model starts to diverge and fails to converge to the optimal weights. It results in a very high cost and fluctuating weight updates.

2) Decrease the learning rate. What happens to the Cost Function graph?
   - As the learning rate decreases, the cost function graph will show a much smoother descent toward the minimum value. However, it will take more iterations to converge to the optimal weights.

3) Display the final weights and compare with the expected values. What can you change to make your model more accurate?
   - Final weights after training: w0: 0.0886, w1: 2.0345 (learning rate 0.005)
   - Expected slope: 2, Expected bias: 0 (since we used slope and bias of 2 and 0 in data generation).
   - To make the model more accurate, ensure proper scaling of data, use more data points, or adjust the learning rate for optimal convergence.
'''

# Display final weights
print("Final Weights:")
print(f"Bias (w0): {weights[0][0]:.4f}")
print(f"Slope (w1): {weights[1][0]:.4f}")

# Expected values: I think this is the actually value...
expected_slope = 2
expected_bias = 0

# Compare final weights with expected values
print("\nComparison with Expected Values:")
print(f"Expected Bias (w0): {expected_bias}")
print(f"Expected Slope (w1): {expected_slope}")
print(f"Difference in Bias: {weights[0][0] - expected_bias:.4f}")
print(f"Difference in Slope: {weights[1][0] - expected_slope:.4f}")
