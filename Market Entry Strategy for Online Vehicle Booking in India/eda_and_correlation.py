import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("/home/ubuntu/cleaned_bangalore_cab_data.csv")

# --- EDA Graphs ---

# Set style for plots
sns.set_style("whitegrid")

# Histograms for key numerical features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Distribution of Key Numerical Features", fontsize=16)

sns.histplot(df["Searches"], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Searches")

sns.histplot(df["Bookings"], kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Bookings")

sns.histplot(df["Completed Trips"], kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Distribution of Completed Trips")

sns.histplot(df["Drivers' Earnings"], kde=True, ax=axes[1, 1]) # Corrected column name
axes[1, 1].set_title("Distribution of Drivers' Earnings")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("/home/ubuntu/eda_histograms.png")
print("EDA Histograms saved to /home/ubuntu/eda_histograms.png")

# Box plots for key numerical features (to check for outliers)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Box Plots of Key Numerical Features", fontsize=16)

sns.boxplot(y=df["Searches"], ax=axes[0, 0])
axes[0, 0].set_title("Box Plot of Searches")

sns.boxplot(y=df["Bookings"], ax=axes[0, 1])
axes[0, 1].set_title("Box Plot of Bookings")

sns.boxplot(y=df["Completed Trips"], ax=axes[1, 0])
axes[1, 0].set_title("Box Plot of Completed Trips")

sns.boxplot(y=df["Drivers' Earnings"], ax=axes[1, 1]) # Corrected column name
axes[1, 1].set_title("Box Plot of Drivers' Earnings")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("/home/ubuntu/eda_boxplots.png")
print("EDA Box Plots saved to /home/ubuntu/eda_boxplots.png")

# --- Correlation Matrix ---

# Select only numerical columns for correlation matrix
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features", fontsize=16)
plt.savefig("/home/ubuntu/correlation_matrix.png")
print("Correlation Matrix saved to /home/ubuntu/correlation_matrix.png")

