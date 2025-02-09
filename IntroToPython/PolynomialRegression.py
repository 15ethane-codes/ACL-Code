import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#Create sample data
data = pd.read_csv("Dataset A.csv")
X = data[['x']].values.reshape(-1,1) # x needs to be a 2D array
y = data['y']
#Visualize sample data
'''plt.figure(figsize=(10,6))
plt.scatter(X, y)
plt.show()'''

#Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X,y)
y_pred_lin = lin_reg_model.predict(X)


#Set the degree of the polynomial
degree = 3
poly = PolynomialFeatures(degree)
#Create x^n features
X_poly = poly.fit_transform(X)
print(X_poly)
#Create linear regression model
poly_reg_model = LinearRegression()
#fit
poly_reg_model.fit(X_poly, y)
# predict
y_pred_reg = poly_reg_model.predict(X_poly)
print(y_pred_reg)

#Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data Points')
plt.plot(X,y_pred_lin, color = 'red', label = f'Linear Regression')
plt.plot(X, y_pred_reg, color='orange', label=f'Polynomial Regression (Degree {degree})')
plt.title('Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

#Determine the linear equation
lin_intercept = lin_reg_model.intercept_
lin_coefficients = lin_reg_model.coef_
print("Linear Regression Equation:")
print("y = ", f"{lin_intercept:.2f}", " + ", f"{lin_coefficients[0]:.2f}*x")
# Calculate MSE
lin_mse = mean_squared_error(y, y_pred_lin)
# Calculate R-squared
lin_r2 = r2_score(y, y_pred_lin)
print("Mean Squared Error (MSE):", lin_mse)
print("R-squared (R2) Value:", lin_r2)

print()

#Determine the polynomial equation
intercept = poly_reg_model.intercept_
coefficients = poly_reg_model.coef_
terms = [f"{coefficients[i]:.2f} * x^{i}" if i > 0 else f"{intercept:.2f}" for i in range(degree + 1)]
equation = " + ".join(terms)
print("Polynomial Regression Equation:")
print(f"y = {equation}")
# Calculate MSE
mse = mean_squared_error(y, y_pred_reg)
# Calculate R-squared
r2 = r2_score(y, y_pred_reg)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Value:", r2)
