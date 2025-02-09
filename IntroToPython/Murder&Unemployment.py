import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'murdersunemployment - murdersunemployment.csv'
data = pd.read_csv(file_path)

#Dropping an outlier
data = data[data['Inhabitants'] <= 4000000]
#data = data.drop(index=19)
print(data)
# Min-max normalization for "Inhabitants" column
data['Inhabitants'] = (data['Inhabitants'] - data['Inhabitants'].min()) / (data['Inhabitants'].max() - data['Inhabitants'].min())

# Compute the cost using Sum of Squared Errors (SSE)
def costFunc(x, weights, y):
    predictions = np.dot(x, weights)
    errors = predictions - y
    return (1 / (2 * len(y))) * np.sum(errors ** 2)


# Compute the gradient for the cost function
def gradDesc(x, weights, y):
    predictions = np.dot(x, weights)
    errors = predictions - y  # Errors
    gradient = (1 / len(y)) * np.dot(x.T, errors)
    return gradient


file_path = 'murdersunemployment - murdersunemployment.csv'

x = data[['Inhabitants']].values  # Feature (normalized inhabitants)
y = data['murders per annum per 1,000,000 '].values.reshape(-1, 1)  # Label

x = np.hstack((np.ones((x.shape[0], 1)), x))

# Initialize weights
weights = np.zeros((2, 1))

# Initialize learning rate and number of iterations
LR = 0.005
maxIter = 10000

# Create array to store costs per iteration
costArray = []

for i in range(maxIter):
    cost = costFunc(x, weights, y)
    costArray.append(cost)
    gradient = gradDesc(x, weights, y)
    weights = weights - LR * gradient

x_min_index = np.argmin(x[:, 1])
x_max_index = np.argmax(x[:, 1])
x_min = x[x_min_index, 1]
x_max = x[x_max_index, 1]

y_min = np.dot([1, x_min], weights)
y_max = np.dot([1, x_max], weights)

# Plot the data and the best fit line
fig1 = plt.figure()
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.plot(x[:, 1], y, 'rx', label='Data Points')
ax1.plot([x_min, x_max], [y_min, y_max], color='blue', label='Best Fit Line')
ax1.set(title='Inhabitants vs. Murders per Annum', xlabel='Normalized Inhabitants',ylabel='Murders per Annum per 1,000,000')
ax1.legend()
plt.show()

# Plot the cost function over iterations
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(maxIter), costArray, color='blue')
ax2.set(title='Cost vs. Iterations', xlabel='Iterations', ylabel='Cost')
plt.show()

# Display final weights
print("Final Weights:")
print(f"Bias (w0): {weights[0][0]:.4f}")
print(f"Slope (w1): {weights[1][0]:.4f}")
