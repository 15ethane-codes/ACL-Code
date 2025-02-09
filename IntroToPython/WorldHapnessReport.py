'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from PolynomialRegression import coefficients

data = pd.read_csv("Happiness_Report_2023 - Happiness_Report_2023.csv")

print(data.info())
print(data.describe())

# 2.
data = data.dropna()

# b. Calculate mean and standard deviation
#pd.set_option("display.max_columns", None)
#print(data.describe(include = 'all'))
mean_values = data.select_dtypes(include=[np.number]).mean()
std_values = data.select_dtypes(include=[np.number]).std()
print("Mean Values:\n", mean_values)
print("Standard Deviation:\n", std_values)

# c. Create graphs
for col in data.columns:
    if col != 'Life Ladder' and col != 'Country name' and col != 'year':  # Exclude the target variable
        sns.scatterplot(x=data[col], y=data['Life Ladder'])
        plt.title(f'Life Ladder vs {col}')
        plt.xlabel(col)
        plt.ylabel('Life Ladder')
        plt.show()

# d. Identify correlations
numeric_data = data.select_dtypes(include=[np.number])
correlations = numeric_data.corr()
print("Correlations with Life Ladder:\n", correlations['Life Ladder'].sort_values(ascending=False))


# e. Standardize features
scaler = StandardScaler()
numeric_columns = data.select_dtypes(include=[np.number]).columns
columns_to_scale = [col for col in numeric_columns if col != 'Life Ladder']
data_scaled = data.copy()
data_scaled[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# 3. Linear Regression
X = data_scaled[['Log GDP per capita','Social support','Healthy life expectancy at birth','Freedom to make life choices','Generosity','Perceptions of corruption','Positive affect','Negative affect']]
y = data_scaled['Life Ladder']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
# Train linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
#Make Predictions
y_pred = linear_model.predict(X_test)
#print(y_pred)
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
r_squared = r2_score(y_test, y_pred)
print("R squared ", r_squared)
# Get coefficients and intercept
slope = linear_model.coef_
intercept = linear_model.intercept_
print("Slope: ", slope, "Intercept: ", intercept)


plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Predicted vs. Actual Quality")
plt.plot([0,9],[0,9], "--k", label='Predictions')
plt.legend()
plt.show()

# 4. Polynomial Regression
# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

train_errors = []
val_errors = []

# Test models with increasing complexity (degree of polynomial)
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


# 5. Ridge and Lasso Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Fit Linear Regression (baseline)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

# Step 4: Fit Ridge Regression
ridge_model = Ridge(alpha=10)
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

#Cross Validation
# For Ridge
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge_cv.fit(X_train, y_train)
print("Optimal alpha for Ridge:", ridge_cv.alpha_)

# For Lasso
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5)
lasso_cv.fit(X_train, y_train)
print("Optimal alpha for Lasso:", lasso_cv.alpha_)
# 6. Apply Models to New Data
new_data = [[9.786, 0.4, 68.7, 0.6, 0.09, 0.2],
            [10.561, 0.8, 71.3, 0.9, 0.3, 0.1],
            [9.210, 0.6, 63.2, 0.5, 0.1, 0.2]]

#Interpret and Use the results
new_study_data = pd.DataFrame(([9.786,0.4,68.7,0.6,0.09,0.2,0.66,0.283],[10.561,0.8,71.3,0.9,0.3,0.1,0.71,0.256],[9.210,0.6,63.2,0.5,0.1,0.2,0.53,0.31]), columns=['Log GDP per capita','Social support','Healthy life expectancy at birth','Freedom to make life choices','Generosity','Perceptions of corruption','Positive affect','Negative affect'])
scaler1 = StandardScaler()
numeric_columns1 = new_study_data.select_dtypes(include=[np.number]).columns
columns_to_scale = [col for col in numeric_columns1]

data_scaled1 = new_study_data.copy()
data_scaled1[columns_to_scale] = scaler1.fit_transform(new_study_data[columns_to_scale])

print(data_scaled1)
new_study_pred = linear_model.predict(data_scaled1)
print(new_study_pred)
'''