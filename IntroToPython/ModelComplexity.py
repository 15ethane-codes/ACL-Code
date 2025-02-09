from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Create sample data
data = pd.read_csv("Dataset D.csv")
X = data[['x']].values.reshape(-1,1) # x needs to be a 2D array
y = data['y']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Track performance metrics
train_errors = []
val_errors = []

# Test models with increasing complexity (degree of polynomial)
degrees = range(1, 10)  # Polynomial degrees
for degree in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict and calculate errors
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))

# Plot the complexity graph
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label="Training Error", marker="o")
plt.plot(degrees, val_errors, label="Validation Error", marker="x")
plt.title("Model Complexity Graph D")
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.show()
