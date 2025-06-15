import pandas as pd

# Load the clustered dataset
df = pd.read_csv("/home/ubuntu/clustered_bangalore_cab_data.csv")

# Load cluster summary
cluster_summary = pd.read_csv("/home/ubuntu/cluster_summary.csv", index_col=0)

# Display cluster summary
print("\nCluster Summary (Mean of Features per Cluster):")
print(cluster_summary)

# Identify characteristics of each cluster to determine potential early adopter segments
# Cluster 0: Low activity, low earnings
# Cluster 1: Very high activity, very high earnings (likely the 'Other Wards' which is an aggregation of many high-demand areas)
# Cluster 2: Moderate activity, moderate earnings

# For early market entry, we are looking for areas with high potential for growth and adoption of new services.
# This would typically align with areas that have a good balance of demand and willingness to try new things.
# Cluster 2 seems to represent a segment with moderate activity and earnings, which could be a good target for early adoption.
# Cluster 1, while having very high activity, might be saturated with existing players (Ola/Uber).
# Cluster 0 represents low-demand areas, not ideal for early market entry.

# Let's focus on Cluster 2 as a potential early adopter segment for a new service.
# We can further analyze the wards within Cluster 2.

print("\nWards in Cluster 2 (Potential Early Adopter Segment):")
print(df[df["Cluster"] == 2]["Ward"].tolist())

# Further analysis of these wards can be done to identify specific characteristics like demographics, tech-savviness etc.
# However, without more granular data, we will rely on the assumption that these wards represent a good balance for early adoption.

# For location analysis, Bangalore as a whole is a Tier-1 city with high smartphone penetration and a tech-savvy population.
# The analysis of wards within Bangalore helps in identifying specific micro-markets.

# Conclusion for Location Analysis:
# Bangalore, being a major tech hub and a Tier-1 city, is highly suitable for early market entry.
# Within Bangalore, the wards identified in Cluster 2 represent a promising segment for targeting early adopters due to their moderate yet significant activity levels.

# Save the analysis conclusion to a file
with open("/home/ubuntu/location_analysis_conclusion.md", "w") as f:
    f.write("## Location Analysis Conclusion\n\nBased on the Innovation Adoption Life Cycle and the market segmentation analysis of Bangalore wards:\n\n*   **Bangalore as a whole:** Is highly suitable for early market entry due to its status as a major tech hub, high smartphone penetration, and a generally tech-savvy and young population. These characteristics align with the 'Innovators' and 'Early Adopters' segments of the Innovation Adoption Life Cycle.\n\n*   **Targeted Wards (Cluster 2):** The wards grouped into Cluster 2 exhibit moderate levels of searches, bookings, and completed trips, along with moderate drivers' earnings. This suggests a healthy level of activity without the potential saturation seen in the highest-demand areas (Cluster 1). This segment represents a promising target for a new vehicle booking service to gain early traction and build a loyal user base before expanding to more competitive or less developed areas.\n\n**Recommendation:** The initial market entry should focus on Bangalore, specifically targeting the wards identified within Cluster 2. This approach allows the startup to leverage a receptive audience and optimize its service offerings in a manageable geographic area before scaling up.\n")

print("Location analysis conclusion saved to /home/ubuntu/location_analysis_conclusion.md")

