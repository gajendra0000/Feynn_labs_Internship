import pandas as pd

# Load the dataset
df = pd.read_csv('/home/ubuntu/upload/All-timeTable-Bangalore-Wards.csv')

# Display the first few rows of the dataframe
print('First 5 rows of the dataframe:')
print(df.head())

# Display dataframe information
print('\nDataFrame Info:')
df.info()

# Display descriptive statistics
print('\nDescriptive Statistics:')
print(df.describe(include='all'))

# Check for missing values
print('\nMissing Values:')
print(df.isnull().sum())

