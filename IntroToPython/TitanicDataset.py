import matplotlib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("TitanicSurvival_Train - TitanicSurvival_Train.csv")
#test = pd.read_csv()

train.isnull().sum()
#test

train.info()
#test

train.describe()
#test

train.isnull().sum()
#test
#test["Survived"] = ""
#test.head()

sns.set()

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.show()



bar_chart('Sex')
print("Survived :\n", train[train['Survived']==1]['Sex'].value_counts())
print("Dead :\n", train[train['Survived']==0]['Sex'].value_counts())
#Notes: Women likely survived than Men

bar_chart('Pclass')
print("Survived :\n", train[train['Survived']==1]['Pclass'].value_counts())
print("Dead :\n", train[train['Survived']==0]['Pclass'].value_counts())
#Notes: 1st class likely survived than other classes and 3rd class are likely dead than other classes.

bar_chart('SibSp')
print("Survived :\n", train[train['Survived']==1]['SibSp'].value_counts())
print("Dead :\n", train[train['Survived']==0]['SibSp'].value_counts())
#Notes: Person aboraded with 1-2 are likely to die(50/50)-remove

bar_chart('Parch')
print("Survived :\n", train[train['Survived']==1]['Parch'].value_counts())
print("Dead :\n", train[train['Survived']==0]['Parch'].value_counts())
#Notes: Person aboraded with 1-2 are likely to die (50/50)-remove

bar_chart('Embarked')
print("Survived :\n", train[train['Survived']==1]['Embarked'].value_counts())
print("Dead :\n", train[train['Survived']==0]['Embarked'].value_counts())
#Person from C-remove, 50/50

bar_chart('Age')

