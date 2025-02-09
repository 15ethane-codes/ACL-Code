import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL.GimpGradientFile import linear

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_iris
iris_data_set = load_iris()
iris_data = pd.DataFrame(iris_data_set.data, columns=iris_data_set.feature_names)
iris_data['Type'] = iris_data_set.target

"""
The target data in this data set is the categorization of the type of iris
0 = Iris Setosa
1 = Iris Versicolor
2 = Iris Virginica

"""
print(iris_data.info())
print(iris_data.head())

#Visualize the dataset
X = iris_data[["petal width (cm)"]]
#sepal length (cm) : sepal width (cm) : petal length (cm) : petal width (cm)
y = iris_data["Type"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(solver= "lbfgs", C = 10, max_iter=100)
#model = LogisticRegression(multi_class='multinomial', solver= "lbfgs", C = 10)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

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

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model accuracy: ", accuracy)
print("Classification Report")
print(report)

cm = confusion_matrix(y_test, y_pred)
disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
disp_cm.plot(cmap = "Blues")
plt.show()

for i in range(5):
    print("Sample ", (i+1), ": ", y_pred_proba[i])

#Softmax Regression
X1 = iris_data[["petal length (cm)", "petal width (cm)"]].values
y1 = iris_data["Type"]
X_trains, X_tests, y_trains, y_tests = train_test_split(X1, y1, test_size=0.2, random_state=42)

softmax_reg = LogisticRegression(C = 4, random_state = 42)
softmax_reg.fit(X_trains, y_trains)

print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]).round(2))

from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                     np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X1[y1 == 2, 0], X1[y1 == 2, 1], "g^", label="Iris virginica")
plt.plot(X1[y1 == 1, 0], X1[y1 == 1, 1], "bs", label="Iris versicolor")
plt.plot(X1[y1 == 0, 0], X1[y1 == 0, 1], "yo", label="Iris setosa")

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="center left")
plt.axis([0.5, 7, 0, 3.5])
plt.grid()
plt.show()

flowers = np.array([[1.9,0.25],[6.5,2.1],[2.5,0.9]])
predictions = softmax_reg.predict(flowers)
probabilities = softmax_reg.predict_proba(flowers)

for i, (flower, prediction, prob) in enumerate(zip(flowers, predictions, probabilities), start=1):
    print(f"Flower {i}:")
    print(f" Petal length = {flower[0]}, Petal width = {flower[1]}")
    print(f" Predicted Class: {prediction}")
    print(f" Class probabilities: {prob.round(2)}")
    print()

