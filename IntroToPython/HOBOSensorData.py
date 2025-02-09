import pandas as pd
import matplotlib.pyplot as plt

file_path = 'S-A-0 2022-01-27 09_58_29 -0500.xlsx - DATA.csv'
data = pd.read_csv(file_path, skiprows=1)
print(data.describe())

data.columns = ['Index', 'DateTime', 'Temperature_F', 'Light_Intensity']

data['DateTime'] = pd.to_datetime(data['DateTime'])

data['Light_Intensity'] = pd.to_numeric(data['Light_Intensity'].str.replace(',', ''), errors='coerce')

data_grouped = data.groupby(data.index // 48).mean()

plt.figure(figsize=(10, 5))

plt.scatter(data_grouped['DateTime'], data_grouped['Temperature_F'], color='red', label='Temperature (°F)')
plt.scatter(data_grouped['DateTime'], data_grouped['Light_Intensity'], color='blue', label='Light Intensity (lum/ft²)')

plt.title('Temperature and Light Intensity Over Time (Averaged Every 48 Entries)')
plt.xlabel('Date Time')
plt.ylabel('Values')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

summary_stats = data[['Temperature_F', 'Light_Intensity']].describe()

missing_values = data.isnull().sum()

date_range = data['DateTime'].min(), data['DateTime'].max()

print("Basic Summary Statistics:\n", summary_stats)
print("\nMissing Values Count:\n", missing_values)
print("\nDate Range of the Data:\n", date_range)
