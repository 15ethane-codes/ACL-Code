from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #Have to download this package


from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df['species'] = iris.target
print(iris_df)


from sklearn.model_selection import train_test_split
X = iris_df.drop(['species'], axis = 1)
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Can use other parameters such as criterion (“gini”, “entropy”) and max_depth
model = DecisionTreeClassifier(criterion="gini", random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                    index = ['Setosa','Versicolor','Virginica'],
                    columns = ['Setosa','Versicolor','Virginica'])
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()
print(f'Accuracy: {accuracy}')
cm = confusion_matrix(y_test, y_pred)
fig = plt.figure(figsize = (10, 7))
tree.plot_tree(model, feature_names = iris.feature_names,  class_names = iris.target_names, filled = True)
plt.show()


