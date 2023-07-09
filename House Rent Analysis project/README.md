
# House Rent Analysis

This repository contains Python code for analyzing house rent data. The code reads the dataset, performs data preprocessing, and generates various visualizations to provide insights into the rental market.
## Dataset

The dataset used for analysis is stored in the House_Rent_Dataset.csv file. It includes information about houses for rent, such as city, area locality, number of bedrooms, rent, and more.

## Usage

To run the code:

1) Clone this repository or download the code files.
2) Place the House_Rent_Dataset.csv file in the same directory as the Python code.
3) Install the necessary dependencies by running pip install pandas numpy matplotlib seaborn.
4) Execute the code using a Python interpreter or Jupyter Notebook.


## Analysis Steps

The code performs the following analysis steps:

Importing Libraries: The required libraries, including pandas, numpy, matplotlib, and seaborn, are imported to support data analysis and visualization.

Loading the Dataset: The code reads the dataset from the House_Rent_Dataset.csv file using the pandas library and stores it in a DataFrame.

Checking for Null Values: The code checks for null values in the dataset using the isnull().sum() function and prints the count of null values for each column.

Checking for Duplicate Values: The code checks for duplicate values in the dataset using the duplicated().sum() function and prints the count of duplicate rows.

Retrieving Column Names: The code retrieves the column names from the dataset using the columns attribute of the DataFrame and prints them.

Calculating Rent Statistics: The code calculates and prints the average, median, highest, and lowest rents in the dataset using the mean(), median(), max(), and min() functions.

Top 10 Highest Rent: The code selects the top 10 houses with the highest rent using the sort_values() function and prints the city, area locality, number of bedrooms, and rent for each house.

Top 10 Minimum Rent: The code selects the top 10 houses with the minimum rent using the sort_values() function and prints the city, area locality, number of bedrooms, and rent for each house.

Scatter Plot of House Sizes vs. Rents: The code generates a scatter plot to visualize the relationship between house sizes (x-axis) and rents (y-axis) and colors the points based on the city. It uses the scatterplot() function from the seaborn library and sets the size (s) and marker (marker) parameters.

Bar Plot of House Counts by City: The code generates a bar plot to visualize the count of houses in each city. It uses the value_counts() function to count the occurrences of each city and plots the bar graph using the barplot() function from the seaborn library.

Tenant Preference Analysis: The code performs an analysis of tenant preferences based on the "Tenant Preferred" column. It groups the data by tenant preferences and furnishing status using the groupby() function and calculates the count of each combination. The results are printed.

Bar Plot of Tenant Preferences: The code generates a bar plot to visualize the distribution of different tenant preferences. It uses the value_counts() function to count the occurrences of each tenant preference and plots the bar graph using the barplot() function from the seaborn library.

Bar Plot of Rent by Tenant Preference: The code generates a bar plot to compare the rent based on different tenant preferences. It uses the barplot() function from the seaborn library and sets the x and y parameters to "Tenant Preferred" and "Rent" columns, respectively.

Percentage of Furnishing Status: The code calculates the percentage of different furnishing statuses (Furnished, Semi-Furnished, Unfurnished) using the value_counts() function and prints the percentages.

Pie Chart of Furnishing Status Distribution: The code generates a pie chart to visualize the distribution of different furnishing statuses. It uses the pie() function from the matplotlib library to plot the pie chart.

Bar Plot of Rent by City: The code generates a bar plot to compare the rent based on different cities. It uses the barplot() function from the seaborn library and sets the x and y parameters to "City" and "Rent" columns, respectively.

Bar Plot of Furnishing Status Distribution: The code generates a bar plot to visualize the distribution of different furnishing statuses. It uses the value_counts() function to count the occurrences of each furnishing status and plots the bar graph using the barplot() function from the seaborn library.

Bar Plot of Bedroom Counts: The code generates a bar plot to visualize the distribution of different numbers of bedrooms. It uses the value_counts() function to count the occurrences of each number of bedrooms and plots the bar graph using the barplot() function from the seaborn library.

Heatmap of Bedroom vs. Area Type: The code generates a heatmap to visualize the relationship between the number of bedrooms and area type. It uses the crosstab() function from the pandas library to create a cross-tabulation of the two variables and the heatmap() function from the seaborn library to plot the heatmap.

Bar Plot of Rent by City and Area Type: The code generates a bar plot to compare the rent based on different cities and area types. It uses the barplot() function from the seaborn library and sets the x and y parameters to "City" and "Rent" columns, respectively, and the hue parameter to "Area Type".
## Results

The code provides various visualizations and insights into the house rental market. By running the code, you can explore the dataset, analyze rent trends, compare cities, and understand tenant preferences.

Feel free to modify the code or dataset to suit your needs and further enhance the analysis.

Note: Ensure that you have the necessary data in the House_Rent_Dataset.csv file in the correct format before running the code.