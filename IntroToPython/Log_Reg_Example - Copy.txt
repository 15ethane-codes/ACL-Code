import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Import Data
data_set = load_breast_cancer()
data = pd.DataFrame(data_set.data, columns=data_set.feature_names)
data['target'] = data_set.target
pd.set_option("display.max_columns", None)
print(data.describe(include = 'all'))
print(data.head())

#data = pd.read_csv("Admission.csv")
#print(data.head())

scaler = StandardScaler()
numeric_columns = data.select_dtypes(include=[np.number]).columns
columns_to_scale = [col for col in numeric_columns if col != 'target']
data_scaled = data.copy()
data_scaled[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
# Select Features and Label
X = data_scaled[['worst radius']] # mean radius; 0.91 - mean texture; 0.69 - mean perimeter; 0.93 - mean area; 0.92 - mean smoothness; 0.62 - mean compactness; 0.68 - mean concavity; 0.80 - mean concave points; 0.68 - mean symmetry; 0.62 - mean fractal dimension; 0.68 - worst radius; 0.94
y = data_scaled['target']  # 0 - malignant, 1 - benign

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 42)

# Initialize Logistic Regression Model
model=LogisticRegression()

# Fit the model and predict
model.fit(X_train, y_train)

# Predicts binary class label for each data point
y_pred = model.predict(X_test)
print(y_pred)

# Predicts probability for each data point
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba)


# Visualize the model
# Create a scatter plot of the data
plt.scatter(X, y)
# Generate a range of x values to plot the decision boundary
x_range = np.linspace(X.min(), X.max(), 100)
# Predict probabilities for the x range
y_prob = model.predict_proba(x_range.reshape(-1, 1))[:, 1]
# Plot the decision boundary
plt.plot(x_range, y_prob, color='black')
# Add labels and title
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Logistic Regression Graph')
plt.show()


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
display_cm = ConfusionMatrixDisplay(conf_matrix, display_labels = ['0','1'])
display_cm.plot()
plt.show()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

train_errors = []
val_errors = []

degrees = range(1, 6)  # Polynomial degrees
for degree in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_reg = model.predict(X_train_poly)

    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))
#Best degree model is 1

# Plot the complexity graph
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label="Training Error", marker="o")
plt.plot(degrees, val_errors, label="Validation Error", marker="x")
plt.title("Model Complexity Graph")
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.show()

X1 = data_scaled[['mean radius', 'mean perimeter', 'mean area', 'worst radius']]
# 5. Ridge and Lasso Regression
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=42)

# Step 3: Fit Linear Regression (baseline)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

# Step 4: Fit Ridge Regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Step 5: Fit Lasso Regression
lasso_model = Lasso(alpha=0.001)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Step 6: Evaluate Models
print("Linear Regression:")
print("  MSE:", mean_squared_error(y_test, y_pred_lr))
print("  R^2:", r2_score(y_test, y_pred_lr))
print("Ridge Regression:")
print("  MSE:", mean_squared_error(y_test, y_pred_ridge))
print("  R^2:", r2_score(y_test, y_pred_ridge))
print("Lasso Regression:")
print("  MSE:", mean_squared_error(y_test, y_pred_lasso))
print("  R^2:", r2_score(y_test, y_pred_lasso))

# Step 7: Plot the Coefficients
plt.figure(figsize=(10, 6))
plt.plot(linear_model.coef_, label="Linear Regression Coefficients", marker="x")
plt.plot(ridge_model.coef_, label="Ridge Coefficients", marker="s")
plt.plot(lasso_model.coef_, label="Lasso Coefficients", marker="d")
plt.title("Comparison of Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.grid(True)
plt.show()

'''#Cross Validation
# For Ridge
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge_cv.fit(X_train, y_train)
print("Optimal alpha for Ridge:", ridge_cv.alpha_)

# For Lasso
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5)
lasso_cv.fit(X_train, y_train)
print("Optimal alpha for Lasso:", lasso_cv.alpha_)
'''
"""
# Apply the model
new_student_data = pd.DataFrame([[670,3.74]], columns = ["gre", "gpa"])
new_prediction = model.predict(new_student_data)
print("Prediction: ", new_prediction)

new_student_data = pd.DataFrame([[780,3.92]], columns = ["gre", "gpa"])
new_prediction = model.predict(new_student_data)
print("Prediction: ", new_prediction)
"""