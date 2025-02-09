import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to read data from CSV
def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None


# Compute the cost using Sum of Squared Errors (SSE)
def costFunc(x, weights, y):
    predictions = np.dot(x, weights)
    errors = predictions - y
    return (1 / (2 * len(y))) * np.sum(errors ** 2)


# Compute the gradient for the cost function
def gradDesc(x, weights, y):
    predictions = np.dot(x, weights)
    errors = predictions - y
    gradient = (1 / len(y)) * np.dot(x.transpose(), errors)
    return gradient


# Load the dataset
file_path = 'Loudoun_Housing - Sheet1.csv'  # Replace with your actual file path
data = read_data(file_path)

if data is not None:
    x = data[['Bathrooms']].values  # Using only Bathrooms as the feature
    y = data['Price'].values.reshape(-1, 1)  # Target is Price

    # Add a bias column of ones to x
    x = np.hstack((np.ones((x.shape[0], 1)), x))

    # Initialize weights
    weights = np.array([[0], [0]])  # Starting with zeros for weights

    # Initialize learning rate and number of iterations
    LR = 0.1
    maxIter = 500

    # Store costs per iteration
    costArray = []

    # Main loop for gradient descent
    for i in range(maxIter):
        cost = costFunc(x, weights, y)
        costArray.append(cost)

        gradient = gradDesc(x, weights, y)
        weights = weights - LR * gradient

    x_min_index = np.argmin(x[:, 1])
    x_max_index = np.argmax(x[:, 1])
    x_min = x[x_min_index, 1]
    x_max = x[x_max_index, 1]

    # Calculate corresponding predicted y values for min and max x values
    y_min = np.dot([1, x_min], weights)
    y_max = np.dot([1, x_max], weights)
    plt.figure()
    plt.plot(range(maxIter), costArray, color='blue')
    plt.title('Cost vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    '''plt.figure()
    plt.scatter(data['Bathrooms'], data['Price'], color='red', label='Data Points')
    x_vals = np.array([data['Bathrooms'].min(), data['Bathrooms'].max()])
    y_vals = weights[0] + weights[1] * x_vals
    plt.plot(x_vals, y_vals, color='blue', label='Best Fit Line')
    plt.title('Bathrooms vs. Price')
    plt.xlabel('Bathrooms')
    plt.ylabel('Price')
    plt.legend()
    plt.show()'''
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.plot(x[:, 1], y, 'rx', label='Data Points')  # Original data points
    ax1.plot([x_min, x_max], [y_min, y_max], color='blue', label='Best Fit Line')  # Best fit line
    ax1.set(title='Data with Best Fit Line', xlabel='Bedrooms', ylabel='Price')
    ax1.legend()
    plt.show()

    # Display final weights
    print("Final Weights:")
    print(f"Bias (w0): {weights[0][0]:.4f}")
    print(f"Slope (w1): {weights[1][0]:.4f}")
