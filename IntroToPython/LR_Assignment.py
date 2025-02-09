#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

#Load Data
data = pd.read_csv("StudentPerformanceFactors - StudentPerformanceFactors.csv")
#print(data.head())
#print(data.info)

#Double brackets is for features
X = data[['Attendance']]
y = data['Exam_Score']

#Examine the Data
plt.figure(figsize=(10,10))
plt.scatter(X,y)
plt.xlabel("Attendance")
plt.ylabel("Exam Score")
plt.title("Exam Score vs. Attendance")
plt.show()
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
plt.scatter(X,y)
#Cretae a trendline based on slope and intercept
plt.axline(xy1=(0, model.intercept_), slope=model.coef_[0], color="red")
plt.xlabel("Attendance")
plt.ylabel("Exam Score")
plt.title("Exam Score vs. Hour Studied")
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Test Scores")
plt.ylabel("Predicted Test Scores")
plt.title("Predicted Test Scores vs. Actual Test Scores")
plt.plot([50,100],[50,100], "--k", label='Predictions')
plt.legend()
plt.show()

#Interpret and Use the results
new_study_data = pd.DataFrame(([90],[75],[50]), columns=["Attendance"])
print(new_study_data)
new_study_pred = model.predict(new_study_data)
print(new_study_pred)