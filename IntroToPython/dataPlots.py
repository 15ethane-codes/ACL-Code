"""
Data Visualization with Matplotlib

"""
import matplotlib.pyplot as plt
import numpy as np

from MlReview import student

"""
Create a line plot of global CO2 emissions over time.
1. Plot the CO2 emissions versus years.
2. Label the x-axis as "Year" and the y-axis as "CO2 Emissions (Billions of Metric Tons)".
3. Add a title "Global CO2 Emissions Over Time".
"""
# Data
years = [2000, 2005, 2010, 2015, 2020]
co2_emissions = [26.8, 29.0, 31.5, 34.2, 33.0]  # Global CO2 emissions in billions of metric tons
fig, ax = plt.subplots()
ax.plot(years, co2_emissions)
ax.set_xlabel("Year")
ax.set_ylabel("CO2 Emissions")
ax.set_title("Global CO2 Emissions Over Time")
plt.show()

"""
Compare the average annual temperature of two major cities over five years. (multi-line plot)
1. Plot temperatures of New York and Los Angeles against the years.
2. Use different colors and line styles for each city.
3. Label each line, and add a legend.
4. Add axis labels and a title "Average Annual Temperature Comparison".
"""
# Data
years = [2016, 2017, 2018, 2019, 2020]
temperature_ny = [12.4, 12.8, 11.9, 12.3, 12.0]  # Temperatures in Celsius
temperature_la = [17.6, 17.9, 17.2, 18.1, 17.5]
fig, ax = plt.subplots()
ax.plot(years, temperature_ny, label="NY")
ax.plot(years, temperature_la, label="LA")
ax.set_xlabel("Year")
ax.set_ylabel("Temperature")
ax.set_title("Average Annual Temperature Comparison")
plt.legend()
plt.show()

"""
Create a bar chart to show the sales of different products in a quarter.
1. Create a bar chart using the sales data.
2. Label each bar with the respective product. Set the bar width so the labels can be clearly displayed.
3. Add labels for the x and y axes.
4. Include a title "Quarterly Sales by Product".
"""
# Data
products = ['Laptops', 'Tablets', 'Smartphones', 'Desktops']
sales = [150, 120, 300, 80]  # Sales units
fig, ax = plt.subplots()
ax.bar(products, sales, linewidth = 3)
ax.set_xlabel("Products")
ax.set_ylabel("Sales")
ax.set_title("Quarterly Sales by Product")
plt.show()


"""
Create a scatter plot to show the relationship between height and weight in teens.
1. Create a scatter plot of height (y) vs. weight (x).
3. Add x and y labels and a title "Height vs. Weight".

"""
patientData = {
    "ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    "Weight": [125, 101, 165, 187, 144, 202, 113, 191, 223, 303, 166, 150, 108, 217, 191],
    "Gender": ['F', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'M', 'M'],
    "Height (in)": [64, 59, 70, 72, 66, 72, 61, 71, 74, 69, 69, 67, 60, 70, 65]
}
fig, ax = plt.subplots()
ax.scatter(patientData["Weight"], patientData["Height (in)"])
ax.set_xlabel("Weight")
ax.set_ylabel("Height (in)")
ax.set_title("Weight vs Height")
plt.show()

"""
Analyze the age distribution of survey respondents using a histogram.
1. Create a histogram of the ages with appropriate bins.
2. Add labels for x (Age) and y (Frequency) axes.
3. Include a title "Age Distribution of Survey Respondents".
"""
# Data
ages = [22, 45, 30, 59, 21, 36, 28, 33, 43, 55, 40, 25, 37, 31, 39]
fig, ax = plt.subplots()
ax.hist(ages)
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
ax.set_title("Age Distribution of Survey Respondents")
plt.show()

"""
Student Performance Data
This data set contains information regarding factors that may impact student academic performance

Create 2 different plots (of your choosing) using this data to show how a factor impacts test score.  

"""
studentData = {
    "Hours_studied": [23, 19, 24, 29, 19, 19, 28, 25, 17, 23, 17, 17, 21, 9, 10, 17, 14, 22, 15, 12],
    "Attendance": [84, 64, 98, 89, 92, 88, 84, 78, 94, 98, 80, 97, 83, 82, 78, 68, 60, 70, 80, 75],
    "Sleep_hours": [7, 8, 7, 8, 6, 8, 7, 6, 6, 8, 8, 6, 8, 8, 8, 8, 10, 6, 9, 7],
    "Motivation": ['low', 'low', 'medium', 'medium', 'medium', 'medium', 'low', 'medium', 'high', 'medium', 'medium', 'low', 'low',  'medium', 'medium', 'medium', 'low',  'medium',  'high', 'low'],
    "test_score": [73, 59, 91, 98, 65, 89, 68, 50, 80, 75, 88, 87, 97, 72, 74, 70, 82, 91, 89, 99]
}
plt.subplot(3,1,1)
plt.title("Test Scores vs Sleep hours")
plt.scatter(studentData["Sleep_hours"], studentData["test_score"])
plt.xlabel("Sleep hours")
plt.ylabel("Test Scores")

plt.subplot(3,1,3)
plt.scatter(studentData["Hours_studied"],studentData["test_score"])
plt.title("Test Scores vs Hours Studied")
plt.xlabel("Hours Studied")
plt.ylabel("Test Scores")
plt.show()



