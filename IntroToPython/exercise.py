"""
The Linnerud dataset is a small but useful dataset that consists of data on three different exercises
performed by 20 middle-aged men at a fitness center.
The three exercise variables included in the dataset are chins, situps, and jumps.

chins: number of chin-ups performed by each participant
situps: number of sit-ups performed by each participant
jumps: number of jumping jacks performed by each participant

By analyzing the data, we can gain insights into which exercises are most effective at building strength and endurance
in this population.
"""

import pandas as pd
from sklearn import datasets
linnerud = datasets.load_linnerud()

"""
1.  Print the data to view the raw data
"""
print(linnerud)
"""
2. Create a data frame of the data
"""
linnerud_df = pd.DataFrame(data = linnerud.data, columns= linnerud.feature_names)
"""
3.  Determine the size of the data set
"""
print(linnerud_df.shape)
"""
4. Determine the column names and data types
"""
column_info = linnerud_df.dtypes
print(column_info)
"""
5. How many Situps did the first (index = 0) person complete?  The 10th person?
"""
print(linnerud_df.loc[0,'Situps'])
print(linnerud_df.loc[9,'Situps'])
"""
6. How many of each exercise did the last person complete?
"""
print(linnerud_df.iloc[-1])
"""
7. Print the rows for participants 10-15.
"""
print(linnerud_df.iloc[10:16,[0,1,2]])
"""
8. Print the jumps column.
"""
print(linnerud_df['Jumps'])
"""
9. Extract the rows where the Situps counts are greater than 100.
"""
print(linnerud_df[linnerud_df['Situps']>100])
"""
10.  Find the mean number of Jumps.
"""
print(linnerud_df["Jumps"].mean())
"""
11.  Find the median number of Jumps.
"""
print(linnerud_df["Jumps"].median())
"""
12.  Find the total number of Jumps.
"""
print(linnerud_df["Jumps"].sum())
