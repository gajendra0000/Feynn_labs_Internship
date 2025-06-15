import pandas as pd

# Load the dataset
df = pd.read_csv("/home/ubuntu/upload/All-timeTable-Bangalore-Wards.csv")

# Function to clean and convert columns to numeric
def clean_numeric(series):
    return series.astype(str).str.replace("₹", "").str.replace(",", "").astype(float)

# Columns to clean and convert
columns_to_clean = [
    "Searches",
    "Searches which got estimate",
    "Searches for Quotes",
    "Searches which got Quotes",
    "Bookings",
    "Completed Trips",
    "Cancelled Bookings",
    "Drivers' Earnings",
    "Distance Travelled (km)",
    "Average Fare per Trip"
]

for col in columns_to_clean:
    df[col] = clean_numeric(df[col])

# Convert percentage columns to float
percentage_columns = [
    "Search-to-estimate Rate",
    "Estimate-to-search for quotes Rate",
    "Quote Acceptance Rate",
    "Quote-to-booking Rate",
    "Booking Cancellation Rate",
    "Conversion Rate"
]

for col in percentage_columns:
    df[col] = df[col].astype(str).str.replace("%", "").astype(float) / 100

# Display the first few rows of the cleaned dataframe
print("First 5 rows of the cleaned dataframe:")
print(df.head())

# Display dataframe information to verify data types
print("\nDataFrame Info after cleaning:")
df.info()

# Save the cleaned data to a new CSV file
df.to_csv("/home/ubuntu/cleaned_bangalore_cab_data.csv", index=False)
print("\nCleaned data saved to /home/ubuntu/cleaned_bangalore_cab_data.csv")

