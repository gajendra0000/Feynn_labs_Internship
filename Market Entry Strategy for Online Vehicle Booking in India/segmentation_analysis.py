import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("/home/ubuntu/cleaned_bangalore_cab_data.csv")

# Select features for clustering
# We'll use numerical columns that represent booking behavior and demand
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

X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
# Sum of squared distances
ssd = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    ssd.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), ssd, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (SSD)')
plt.grid(True)
plt.savefig('/home/ubuntu/elbow_method.png')
print("Elbow method plot saved to /home/ubuntu/elbow_method.png")

# Based on the elbow method, let's assume optimal K is 3 or 4 for now. We can adjust after reviewing the plot.
# For demonstration, let's choose k=3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze the characteristics of each cluster
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
cluster_centers['Cluster'] = range(k)
print("\nCluster Centers (Original Scale):")
print(cluster_centers)

# Count of wards in each cluster
print("\nNumber of Wards per Cluster:")
print(df['Cluster'].value_counts().sort_index())

# Save the clustered data
df.to_csv("/home/ubuntu/clustered_bangalore_cab_data.csv", index=False)
print("Clustered data saved to /home/ubuntu/clustered_bangalore_cab_data.csv")

# Visualize clusters (example: scatter plot of two features)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Completed Trips', y='Drivers\' Earnings', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.7)
plt.title('Clusters of Bangalore Wards by Completed Trips and Drivers\' Earnings')
plt.xlabel('Completed Trips')
plt.ylabel('Drivers\' Earnings')
plt.grid(True)
plt.savefig('/home/ubuntu/cluster_scatter_plot.png')
print("Cluster scatter plot saved to /home/ubuntu/cluster_scatter_plot.png")

