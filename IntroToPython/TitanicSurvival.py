import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.datasets import load_iris

data =  pd.read_csv("TitanicSurvival_Train - TitanicSurvival_Train.csv")

# converting gender categorical to binary
data['Sex'] = data['Sex'].replace("male", 1)
data['Sex'] = data['Sex'].replace("female", 0)

# min max normalization (0-1)
data['Age'] = (data['Age'] -data['Age'].min())/ (data['Age'].max() - data['Age'].min())
data['Fare'] = (data['Fare'] -data['Fare'].min())/ (data['Fare'].max() - data['Fare'].min())
data['Pclass'] = (data['Pclass'] -data['Pclass'].min())/ (data['Pclass'].max() - data['Pclass'].min())

data = data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId', 'Embarked'], axis=1)

for i in data.columns:
    data[i].fillna(value=data[i].mean(), inplace=True)

X = data[['Fare','Age','Pclass','Sex']].values
#X = data[['Fare']].values
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Model accuracy: ", accuracy)
print("Classification Report")
print(report)

# plt.scatter(X, y)
# x_range = np.linspace(X.min(), X.max(), 100)
# y_prob = model.predict_proba(x_range.reshape(-1, 1))[:, 1]
# plt.plot(x_range, y_prob, color='black')
# plt.xlabel('X')
# plt.ylabel('Probability')
# plt.title('Logistic Regression Graph of')
# plt.show()

cm = confusion_matrix(y_test, y_pred)
disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
disp_cm.plot(cmap="Blues")
plt.show()

testData =  pd.read_csv("TitanicSurvival_Test - TitanicSurvival_Test.csv")
# sex --> 1 and 0
testData['Sex'] = testData['Sex'].replace("male", 1)
testData['Sex'] = testData['Sex'].replace("female", 0)
 # normalized
testData['Age'] = (testData['Age'] -testData['Age'].min())/ (testData['Age'].max() - testData['Age'].min())
testData['Fare'] = (testData['Fare'] -testData['Fare'].min())/ (testData['Fare'].max() - testData['Fare'].min())
testData['Pclass'] = (testData['Pclass'] -testData['Pclass'].min())/ (testData['Pclass'].max() - testData['Pclass'].min())


testData = testData.drop(['Cabin', 'Ticket', 'Name', 'PassengerId', 'Embarked'], axis = 1)

# makes it 4 args from the 6 so it can run mode.predict
testData = testData[['Fare', 'Age', 'Pclass', 'Sex']]

for i in testData.columns:
    testData[i].fillna(value=testData[i].mean(), inplace=True)

test_y_pred = model.predict(testData)
print(test_y_pred)