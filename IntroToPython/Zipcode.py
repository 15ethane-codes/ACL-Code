"""
StateZipCode Lab
  - read in a csv file with zip code info
      row: zip code, state, place(town, city), and the Latitude/Longitude
      DictReader will create a Dictionary entry for each row

  1. Create a Dictionary where:
       Key: State Name
       value: list of tuples - (latitude,longitude) for each zip code.

  2. Process the Dictionary:
       Compute the average latitude and longitude per State,
       Then display the information in the following format:
               New York, 2153, (42.224909,-75.145666)
          Massachusetts,  684, (42.235230,-71.494505)
           Rhode Island,   90, (41.708033,-71.501928)
          New Hampshire,  281, (43.388981,-71.566260)
                           ^
                           number of zip code per state

       Use the following 'formatted' print statement:
          print('%15s, %4d, (%f,%f)' % (state, cnt, avgLat, avgLong))

"""
import csv
import matplotlib.pyplot as plt

all_zip_list = []  # List of dictionaries for zip code data
state_dictionary = {}  # Dictionary for state with list of tuples


def read_postal_file():
    with open('us_postal_codes.csv') as my_file:
        all_zip_data = csv.DictReader(my_file)

        for line in all_zip_data:
            all_zip_list.append(line)

            # Get state and coordinates
            state = line['State']
            latitude = float(line['Latitude'])
            longitude = float(line['Longitude'])

            # Append lat and long
            if state not in state_dictionary:
                state_dictionary[state] = []
            state_dictionary[state].append((latitude, longitude))

    # calculate averages and print
    calculate_averages_and_print()


def calculate_averages_and_print():
    averages = {}
    #Calculates average for lat and long
    for state, coords in state_dictionary.items():
        cnt = len(coords)
        avg_lat = sum(lat for lat, long in coords) / cnt
        avg_long = sum(long for lat, long in coords) / cnt
        averages[state] = (cnt, avg_lat, avg_long)

        # Print data
        print('%15s, %4d, (%f,%f)' % (state, cnt, avg_lat, avg_long))

    # Create bar chart
    create_bar_chart()
    # Create scatter plots
    create_scatter_plots(averages)


def create_bar_chart():
    states = list(state_dictionary.keys())
    counts = [len(coords) for coords in state_dictionary.values()]
    plt.barh(states, counts)
    plt.title('Number of Lat/Long per State')
    plt.show()

def create_scatter_plots(averages):
    all_lat = []
    all_long = []
    avg_lat = []
    avg_long = []


    lat_range = (24.396308, 49.384358)
    long_range = (-125.0, -66.93457)

    for coords in state_dictionary.values():
        for lat, long in coords:
            if lat_range[0] <= lat <= lat_range[1] and long_range[0] <= long <= long_range[1]:
                all_lat.append(lat)
                all_long.append(long)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(all_long, all_lat, alpha=0.5,marker ='+', color='orange')
    plt.title('Latitude and Longitude of US States')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Scatter plot for average latitudes and longitudes
    for state, (cnt, avg_lat_val, avg_long_val) in averages.items():
        if lat_range[0] <= avg_lat_val <= lat_range[1] and long_range[0] <= avg_long_val <= long_range[1]:
            avg_lat.append(avg_lat_val)
            avg_long.append(avg_long_val)

    plt.subplot(1, 2, 2)
    plt.scatter(avg_long, avg_lat, alpha=0.5,marker='*', color='green')
    plt.title('Average Latitude and Longitude of US States')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.tight_layout()
    plt.show()




read_postal_file()
