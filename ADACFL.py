import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('EVCSNB_cleaned.csv')

# Convert the 'St_Time' column to datetime format
data['St_Time'] = pd.to_datetime(data['St_Time'])

# Add a 'Week' column to the dataset for weekly aggregation
data['Week'] = data['St_Time'].dt.to_period('W').apply(lambda r: r.start_time)

# Group by station name and week, aggregating the consumption
weekly_data = data.groupby(['St_Name', 'Week']).agg({'Consumption': 'sum', 'St_Name': 'count'}).rename(columns={'St_Name': 'Transactions'}).reset_index()

# Find the top 30 charging stations with the most transactions
top_stations = weekly_data.groupby('St_Name')['Transactions'].sum().nlargest(30).index

# Filter the dataset to include only the top 30 stations
top_stations_data = weekly_data[weekly_data['St_Name'].isin(top_stations)]

# Pivot the data to get a matrix suitable for clustering
pivot_data = top_stations_data.pivot(index='St_Name', columns='Week', values='Consumption').fillna(0)

# Ensure all column names are strings
pivot_data.columns = pivot_data.columns.astype(str)

# Simulate partial weight sharing by using principal components or a subset of features
partial_weights = pivot_data.iloc[:, :10]  # Using the first 10 weeks as a proxy

# Perform hierarchical clustering
linkage_matrix = linkage(partial_weights, method='ward')

# Plot the dendrogram to visualize the clusters
plt.figure(figsize=(15, 8))
dendrogram(linkage_matrix, labels=pivot_data.index, leaf_rotation=90)
plt.title('Dendrogram for Charging Stations (Initial Clustering)')
plt.xlabel('Station Name')
plt.ylabel('Distance')
plt.show()

# Perform the initial clustering using AgglomerativeClustering based on the linkage matrix
n_initial_clusters = 5  # Start with 5 clusters based on the dendrogram
initial_clustering = AgglomerativeClustering(n_clusters=n_initial_clusters, metric='euclidean', linkage='ward')
initial_clusters = initial_clustering.fit_predict(partial_weights)

# Add the initial cluster assignments to the data
pivot_data['Initial_Cluster'] = initial_clusters

# Sample a smaller subset of the dataset
small_sample_data = pivot_data.sample(n=10)

# Function to simulate federated learning with dynamic clustering using AdaCFL on a smaller dataset
def simulate_federated_learning_small(data, n_rounds=30, n_clients=2, initial_clusters=3):
    # Initialize the number of clusters with the initial clusters
    current_clusters = initial_clusters
    cluster_counts_per_round = []
    convergence_round = None
    prev_clusters = None
    
    for round_num in range(n_rounds):
        local_clusters = []

        # Split the data into n_clients parts
        split_data = np.array_split(data, n_clients)

        # Each client performs clustering locally if it has enough data points
        for client_data in split_data:
            max_clusters = min(current_clusters, len(client_data))  # Ensure clusters <= number of samples
            if len(client_data) >= 2:  # At least 2 clusters are needed
                local_kmeans = KMeans(n_clusters=max_clusters, random_state=round_num, max_iter=50)
                client_clusters = local_kmeans.fit_predict(client_data.iloc[:, :-1])
                local_clusters.append(local_kmeans.cluster_centers_)

        # Aggregate clusters globally using the mean of cluster centers
        if local_clusters:
            global_centers = np.mean(local_clusters, axis=0)

            # Assign global clusters to each data point based on the nearest global center
            _, closest_centers = pairwise_distances_argmin_min(data.iloc[:, :-1], global_centers)
            data['Cluster'] = closest_centers

            # Adjust the number of clusters based on some logic (e.g., stability, silhouette score, etc.)
            unique_clusters = data['Cluster'].nunique()
            cluster_counts_per_round.append(unique_clusters)

            # Check for convergence (clusters stabilize)
            if prev_clusters is not None and unique_clusters == prev_clusters:
                if convergence_round is None:
                    convergence_round = round_num
            else:
                convergence_round = None

            prev_clusters = unique_clusters

            # Adjust the number of clusters for the next round
            current_clusters = unique_clusters

    return cluster_counts_per_round, convergence_round

# Run the simulation
cluster_counts_per_round, convergence_round = simulate_federated_learning_small(small_sample_data)

# Plot the number of clusters across the rounds
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cluster_counts_per_round) + 1), cluster_counts_per_round, marker='o')
if convergence_round is not None:
    plt.axvline(x=convergence_round, color='red', linestyle='--', label=f'Convergence at Round {convergence_round}')
plt.title('Number of Clusters per Round in Federated Learning (AdaCFL)')
plt.xlabel('Federated Learning Round')
plt.ylabel('Number of Clusters')
plt.grid(True)
plt.legend()
plt.show()
