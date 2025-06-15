import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("EVMarket-India_data.csv")

# Select numerical features for clustering
features = ["AccelSec", "TopSpeed_KmH", "Range_Km", "Efficiency_WhKm", "PriceEuro"]
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
Sum_of_squared_distances = []
K = range(1, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km = km.fit(X_scaled)
    Sum_of_squared_distances.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, Sum_of_squared_distances, "bx-")
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.title("Elbow Method For Optimal k")
plt.savefig("/home/ubuntu/elbow_method.png")


