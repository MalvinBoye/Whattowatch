import pandas as pd

# Load the data from the extracted files
movies = pd.read_csv('datasets/movielens/movies.csv')
ratings = pd.read_csv('datasets/movielens/user_rating_history.csv')

# Display the first few rows of the datasets
print("Movies DataFrame:")
print(movies.head())
print("\nRatings DataFrame:")
print(ratings.head())

# Check for missing values
print("\nMissing values in Movies DataFrame:")
print(movies.isnull().sum())
print("\nMissing values in Ratings DataFrame:")
print(ratings.isnull().sum())

# Drop rows with missing values (if any)
movies.dropna(inplace=True)
ratings.dropna(inplace=True)

# Display the cleaned data
print("\nCleaned Movies DataFrame:")
print(movies.head())
print("\nCleaned Ratings DataFrame:")
print(ratings.head())
