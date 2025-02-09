# Importing the libraries
from pydoc import TextDoc

import numpy as np

######  Read in files into 2 numpy matrices
def readDataFile(fileName):
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')

    # loadtxt defaults to floats, use dtype to specify string
    # usecols chooses the columns to use, by default, all columns are used.
    # skiprows skips a header row if you need to
    data = np.loadtxt(raw_data, skiprows=1, delimiter=",", dtype='str')
    names = data[:, 0]
    stats = data[:, 1:6].astype(int)

    return names, stats


######  Display the names and stats
def printStats(desc, names, stats):
    print('\n', desc, ':')
    for index in range(len(names)):
        # print(names_1998[index], end='')
        print('{:>18}'.format(names[index]), ': ', end='')
        for col in range(len(stats[0])):
            print(stats[index][col], ' ', end='')
        print('')  # newline


# =============================================================================
# Read in the matrix1998 file
# =============================================================================
names_1998, stats_1998 = readDataFile('Football_matrices_1998.csv')

# =============================================================================
# Read in the matrix1999 file
# =============================================================================
names_1999, stats_1999 = readDataFile('Football_matrices_1999.csv')


# =============================================================================
# create a new stats matrix with the sum of stats and display
# =============================================================================
sum_stats = stats_1998 + stats_1999
# =============================================================================
# Find the Difference and print
# =============================================================================
difference_stats = stats_1998 + stats_1999

# =============================================================================
# Write a method to compute the QB ratings given a matrix of stats
#         - could be 1998, 1999, or the sum of both years
# =============================================================================
def getRatings(stats):
    ATT = stats[:, 0]
    COMP = stats[:, 1]
    YDS = stats[:,2]
    TD = stats[:,3]
    INT = stats[:,4]
    a = np.clip(((COMP/ATT)-0.3)*5,0,2.375)
    b = np.clip(((YDS/ATT)-3)*0.25, 0, 2.375)
    c = np.clip((TD/ATT)*20, 0, 2.375)
    d = np.clip(2.375-((INT/ATT)*25),0, 2.375)
    ratings = ((a + b + c + d)/6)*100
    return ratings


# =============================================================================
# Use the method above to get the QB ratings for 1998 and 1999
# then display the rating for each QB for both years
#
# expected output:
# QB Ratings:
#       Quarterback :   1998    1999
#       Troy Aikman :  88.46   81.12
#        Tony Banks :  68.62   81.20
#        Jeff Blake :  78.20   77.60
#   Steve Beuerlein :  88.25   94.58
#
# =============================================================================

ratings_1998 = getRatings(stats_1998)
ratings_1999 = getRatings(stats_1999)

print('\n', 'QB Ratings:')
print('{:>18}'.format("Quarterback"), ': ', end='')
print('{:>6}'.format('1998'), ' ', end='')
print('{:>6}'.format('1999'))

for index in range(len(names_1999)):
    print('{:>18}'.format(names_1999[index]), ': ', end='')
    print('{:6.2f}'.format(ratings_1998[index]), ' ', end='')
    print('{:6.2f}'.format(ratings_1999[index]))