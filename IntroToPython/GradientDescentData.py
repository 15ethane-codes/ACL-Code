import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to read data from CSV
def read_data(file_path):
    try:
        data = pd.read_csv(file_path, delimiter=';')
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

def preprocess_data(data):
    # debug purposes
    print("Original Columns:", data.columns)

    # Handling missing values
    if data.isnull().sum().any():
        print("Missing values found. Filling with mean for numerical columns.")
        for column in data.select_dtypes(include=['float64', 'int64']).columns:
            data[column].fillna(data[column].mean(), inplace=True)

    # One-hot encode categorical columns
    categorical_cols = ['Student Country', 'Type of Answer', 'Topic']  # Update with categorical columns
    for col in categorical_cols:
        if col in data.columns:
            data = pd.get_dummies(data, columns=[col], drop_first=True)

    # Question ID is target(label)
    if 'Question ID' in data.columns:
        x = data.drop(columns=['Question ID'])  # Features
        y = data['Question ID']  # Target
    else:
        print("Target variable 'Question ID' not found in data.")
        return None, None

    # Return the preprocessed data
    print("Preprocessed Data:")
    print(x.head())  # Display the first few rows
    return x, y

# Compute the cost using Sum of Squared Errors (SSE)
def costFunc(x, weights, y):
    predictions = np.dot(x, weights)  # Predicted values
    errors = predictions - y.values.reshape(-1, 1)
    return (1 / (2 * len(y))) * np.sum(errors ** 2)

# Compute the gradient for the cost function
def gradDesc(x, weights, y):
    predictions = np.dot(x, weights)
    errors = predictions - y.values.reshape(-1, 1)
    gradient = (1 / len(y)) * np.dot(x.T, errors)
    return gradient

# Load the dataset
file_path = 'Mathematics Dataset - Sheet1.csv'
data = read_data(file_path)

if data is not None:
    # Preprocess the dataset
    x, y = preprocess_data(data)

    if x is not None and y is not None:
        # Remove any rows where target is not a number
        valid_indices = ~y.isnull()  # Check for Not a number in y
        x = x[valid_indices]  # Keep only valid rows in x
        y = y[valid_indices]  # Keep only valid rows in y

        # Ensure that x contains only numeric data before scaling
        numeric_x = x.select_dtypes(include=[np.number])

        # Normalize features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(numeric_x)

        # Initialize weights
        weights = np.zeros((x_scaled.shape[1], 1)) # Adjust based on number of features

        # Initialize learning rate and number of iterations
        LR = 0.1  # Adjust as necessary
        maxIter = 100  # Increase for better convergence

        # Create array to store costs per iteration
        costArray = []

        # Main loop for gradient descent
        for i in range(maxIter):
            cost = costFunc(x_scaled, weights, y)
            costArray.append(cost)

            gradient = gradDesc(x_scaled, weights, y)
            weights = weights - LR * gradient

        # Plotting
        plt.figure()
        plt.plot(x_scaled[:,0], y, color='red', label='Data Points')
        y_pred = np.dot(x_scaled, weights)
        plt.plot(x_scaled[:,0], y_pred, color='blue',label='Prediction', alpha=0.5)
        plt.title('Best Fit line')
        plt.xlabel('Feature 1 (scaled)')
        plt.ylabel('Target (Question ID)')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(range(maxIter), costArray, color='blue')
        plt.title('Cost vs. Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

        # Display final weights
        print("Final Weights:")
        print(weights)
