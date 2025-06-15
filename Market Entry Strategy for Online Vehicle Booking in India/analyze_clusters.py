import pandas as pd

# Load the clustered dataset
df = pd.read_csv("/home/ubuntu/clustered_bangalore_cab_data.csv")

# Define features used for clustering
features = [
    "Searches",
    "Bookings",
    "Completed Trips",
    "Conversion Rate",
    "Average Distance per Trip (km)",
    "Average Fare per Trip",
    "Distance Travelled (km)",
    "Drivers\' Earnings"
]

# Calculate the mean of each feature for each cluster
cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster Summary (Mean of Features per Cluster):")
print(cluster_summary)

# Calculate the count of wards in each cluster
cluster_counts = df["Cluster"].value_counts().sort_index()
print("\nNumber of Wards per Cluster:")
print(cluster_counts)

# Save the cluster summary to a CSV file
cluster_summary.to_csv("/home/ubuntu/cluster_summary.csv")
print("Cluster summary saved to /home/ubuntu/cluster_summary.csv")

# Save cluster counts to a CSV file
cluster_counts.to_csv("/home/ubuntu/cluster_counts.csv")
print("Cluster counts saved to /home/ubuntu/cluster_counts.csv")

