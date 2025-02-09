# Bagging
# importing utility modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# importing machine learning models for prediction
import xgboost as xgb

# importing bagging module
from sklearn.ensemble import BaggingRegressor

# loading train data set in dataframe from train_data.csv file
df = pd.read_csv("Happiness_Report_2023 - Happiness_Report_2023.csv")
df = df.drop(labels=["Country name", "year"], axis=1)
df = df.dropna()
# getting target data from the dataframe
target = df["Life Ladder"]

# getting train data from the dataframe
train = df.drop(labels=["Life Ladder"], axis=1)

# Splitting between train data into training and validation dataset
X_train, X_test, y_train, y_test = train_test_split(
    train, target, test_size=0.20)

# initializing the bagging model using XGboost as base model with default parameters
model = BaggingRegressor(xgb.XGBRegressor())

# training model
model.fit(X_train, y_train)

# predicting the output on the test dataset
pred = model.predict(X_test)

# printing the mean squared error between real value and predicted value
print(mean_squared_error(y_test, pred))

# Boosting
# importing utility modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# importing machine learning models for prediction
from sklearn.ensemble import GradientBoostingRegressor

# loading train data set in dataframe from train_data.csv file
df = pd.read_csv("Happiness_Report_2023 - Happiness_Report_2023.csv")
df = df.drop(labels=["Country name", "year"], axis=1)
df = df.dropna()
# getting target data from the dataframe
target = df["Life Ladder"]

# getting train data from the dataframe
train = df.drop(labels=["Life Ladder"], axis=1)

# Splitting between train data into training and validation dataset
X_train, X_test, y_train, y_test = train_test_split(
    train, target, test_size=0.20)

# initializing the boosting module with default parameters
model = GradientBoostingRegressor()

# training the model on the train dataset
model.fit(X_train, y_train)

# predicting the output on the test dataset
pred_final = model.predict(X_test)

# printing the mean squared error between real value and predicted value
print(mean_squared_error(y_test, pred_final))

