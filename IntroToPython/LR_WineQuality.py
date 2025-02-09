#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

#Load Data
data = pd.read_csv("Wine_Quality_White - Sheet1(1).csv")
print(data.head())
print(data.info)

#Double brackets is for features
X = data[['alcohol','volatile acidity', 'fixed acidity']]
y = data['quality']
print(X)

#Examine the Data
#Feature 1
'''plt.figure(figsize=(10,10))
plt.scatter(X['alcohol'],y)
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Quality vs. Alcohol")
plt.show()'''
#Feature 2
'''plt.figure(figsize=(10,10))
plt.scatter(X['volatile acidity'],y)
plt.xlabel("Volatile Acidity")
plt.ylabel("Quality")
plt.title("Quality vs. Acidity")
plt.show()'''
#Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Create and Fit the Model
model = LinearRegression()
model.fit(X_train, y_train)

#Make Predictions
y_pred = model.predict(X_test)
print(y_pred)

#Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

r_squared = r2_score(y_test, y_pred)
print("R squared ", r_squared)

slope = model.coef_
intercept = model.intercept_

print("Slope: ", slope, "Intercept: ", intercept)

#Visualization
plt.figure(figsize=(10,10))
plt.scatter(X['alcohol'],y)
#Cretae a trendline based on slope and intercept
plt.axline(xy1=(0, model.intercept_), slope=model.coef_[0], color="red")
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Quality vs. Alcohol")
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(X['volatile acidity'],y)
#Cretae a trendline based on slope and intercept
plt.axline(xy1=(0, model.intercept_), slope=model.coef_[0], color="red")
plt.xlabel("Volatile acidity")
plt.ylabel("Quality")
plt.title("Quality vs. Volatile Acidity")
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(X['fixed acidity'],y)
#Cretae a trendline based on slope and intercept
plt.axline(xy1=(0, model.intercept_), slope=model.coef_[0], color="red")
plt.xlabel("Fixed Acidity")
plt.ylabel("Quality")
plt.title("Quality vs. Fixed Acidity")
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Predicted vs. Actual Quality")
plt.plot([0,9],[0,9], "--k", label='Predictions')
plt.legend()
plt.show()

#Interpret and Use the results
new_study_data = pd.DataFrame(([12, 0.4, 5.2],[1.2, 0.15, 7.3],[6.4, 0.84, 9.6]), columns=['alcohol','volatile acidity', 'fixed acidity'])
print(new_study_data)
new_study_pred = model.predict(new_study_data)
print(new_study_pred)