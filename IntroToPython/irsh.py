import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

print(iris)
'''fig, ax  = plt.subplots()
scatter = ax.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
ax.set(xlabel = iris.feature_names[0], ylabel= iris.feature_names[1])
fig = ax.legend(scatter.legend_elements()[0], iris.target_names, loc = "lower right")
plt.show()'''

#Create a data frame
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

iris_df['target'] = iris.target_names[iris.target]

print(iris_df)

#print(iris_df.info())
#print(iris_df.describe())

#print(iris_df.columns[1])
print(iris_df['target'].unique())

#take a look at the first 5 lines of data
print(iris_df.head())
#how many instances are in the data set
print(iris_df.index)
#what is the name of the third column
print(iris_df.columns[2])
#what is the data type of the columns
print(iris_df.dtypes)
#or
print(iris_df.info())
#size of the dataframe
print(iris_df.size)
#number of columns
print(iris_df.shape)
#dimensions
print(iris_df.ndim)

print(iris_df.describe())

print(iris_df["sepal width (cm)"].value_counts())
print(iris_df['target'].unique())

print(iris_df['petal length (cm)'].mean())
print(iris_df['petal length (cm)'].median())
print(iris_df['petal length (cm)'].sum())
