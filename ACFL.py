# import numpy as np
# import pandas as pd
# from sklearn.metrics import pairwise_distances
# from scipy.cluster.hierarchy import linkage, fcluster
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split

# # Load the dataset
# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Convert 'St_Time' to datetime
# data['St_Time'] = pd.to_datetime(data['St_Time'])

# # Extract relevant features from 'St_Time'
# data['hour'] = data['St_Time'].dt.hour
# data['day'] = data['St_Time'].dt.day
# data['month'] = data['St_Time'].dt.month
# data['weekday'] = data['St_Time'].dt.weekday

# # Select the top 20 charging stations with the maximum number of transactions
# top_stations = data['St_Name'].value_counts().head(20).index
# filtered_data = data[data['St_Name'].isin(top_stations)]

# # Function to calculate model similarity matrix
# def calculate_similarity_matrix(data):
#     # Using only relevant features for similarity calculation
#     features = data[['hour', 'day', 'month', 'weekday', 'Consumption']]
#     similarity_matrix = pairwise_distances(features, metric='euclidean')
#     return similarity_matrix

# # Function to perform adaptive clustering
# def adaptive_clustering(similarity_matrix, max_d=5):
#     # Perform hierarchical clustering
#     Z = linkage(similarity_matrix, 'ward')
#     clusters = fcluster(Z, max_d, criterion='distance')
#     return clusters

# # Function to implement weighted voting mechanism
# def weighted_voting(clusters, data):
#     data['cluster'] = clusters
#     cluster_sizes = data['cluster'].value_counts().to_dict()
    
#     # Calculate voting scores
#     voting_scores = {}
#     for cluster in data['cluster'].unique():
#         cluster_data = data[data['cluster'] == cluster]
#         voting_scores[cluster] = cluster_data['Consumption'].sum() / cluster_sizes[cluster]
    
#     return voting_scores

# # Function to train a model and calculate RMSE with sufficient samples check
# def train_and_evaluate_with_check(data, min_samples=30):
#     if len(data) < min_samples:
#         print(f"Skipping cluster with {len(data)} samples (less than {min_samples})")
#         return None
    
#     X = data[['hour', 'day', 'month', 'weekday']]
#     y = data['Consumption']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     predictions = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
#     return rmse

# # Sample a larger subset of the data
# sampled_data = filtered_data.sample(n=10000, random_state=42)

# # Print sample size
# print(f"Sampled data size: {len(sampled_data)}")

# # Calculate similarity matrix
# similarity_matrix = calculate_similarity_matrix(sampled_data)
# print(f"Similarity matrix shape: {similarity_matrix.shape}")

# # Perform adaptive clustering with adjusted max_d
# clusters = adaptive_clustering(similarity_matrix, max_d=5)
# print(f"Number of clusters formed: {len(np.unique(clusters))}")
# print(f"Cluster sizes: {np.unique(clusters, return_counts=True)}")

# # Perform weighted voting and get voting scores
# voting_scores = weighted_voting(clusters, sampled_data)

# # Train and evaluate model for each cluster with sufficient samples
# rmses = []
# for cluster in np.unique(clusters):
#     cluster_data = sampled_data[sampled_data['cluster'] == cluster]
#     rmse = train_and_evaluate_with_check(cluster_data)
#     if rmse is not None:
#         rmses.append(rmse)

# # Display RMSE for each cluster and overall RMSE
# overall_rmse = np.mean(rmses) if rmses else float('nan')
# print("RMSE for each cluster: ", rmses)
# print("Overall RMSE: ", overall_rmse)

# ***************************************************************************************************************
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# import matplotlib.pyplot as plt

# # Load the dataset
# file_path = '/mnt/data/EVCSNB_cleaned.xlsx'
# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Extract the relevant features: 'St_Name', 'Consumption', 'GHG_Savings'
# # Compute the number of transactions per charging station
# transaction_counts = data['St_Name'].value_counts().head(20)

# # Filter the data to include only the top 20 charging stations
# top_stations = transaction_counts.index
# filtered_data = data[data['St_Name'].isin(top_stations)]

# # Group by 'St_Name' and compute aggregate features
# grouped_data = filtered_data.groupby('St_Name').agg({
#     'Consumption': ['mean', 'std'],
#     'GHG_Savings': ['mean', 'std']
# }).reset_index()

# # Flatten the multi-index columns
# grouped_data.columns = ['St_Name', 'Consumption_mean', 'Consumption_std', 'GHG_Savings_mean', 'GHG_Savings_std']

# # Standardize the features
# features = grouped_data[['Consumption_mean', 'Consumption_std', 'GHG_Savings_mean', 'GHG_Savings_std']]
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# # Pre-clustering using simple statistics
# # Here we skip the pre-clustering and directly use hierarchical clustering for simplicity
# # Hierarchical clustering
# linked = linkage(scaled_features, method='ward')

# # Plot dendrogram
# plt.figure(figsize=(10, 7))
# dendrogram(linked, labels=grouped_data['St_Name'].values, orientation='top', distance_sort='descending', show_leaf_counts=True)
# plt.title('Dendrogram for Hierarchical Clustering')
# plt.xlabel('Charging Stations')
# plt.ylabel('Distance')
# plt.show()

# # Determine the clusters
# # Here we set the threshold to determine clusters; you might need to adjust this based on the dendrogram
# threshold = 1.5
# clusters = fcluster(linked, t=threshold, criterion='distance')

# # Add cluster labels to the original grouped data
# grouped_data['Cluster'] = clusters

# # Print the grouped data with cluster labels
# print(grouped_data)

# # Save the clustered data to a CSV file
# grouped_data.to_csv('clustered_charging_stations.csv', index=False)

#********************************************************************************************************

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# import matplotlib.pyplot as plt

# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Extract the relevant features: 'St_Name', 'Consumption', 'GHG_Savings'
# # Compute the number of transactions per charging station
# transaction_counts = data['St_Name'].value_counts().head(20)

# # Filter the data to include only the top 20 charging stations
# top_stations = transaction_counts.index
# filtered_data = data[data['St_Name'].isin(top_stations)]

# # Group by 'St_Name' and compute aggregate features
# grouped_data = filtered_data.groupby('St_Name').agg({
#     'Consumption': ['mean', 'std'],
#     'GHG_Savings': ['mean', 'std']
# }).reset_index()

# # Flatten the multi-index columns
# grouped_data.columns = ['St_Name', 'Consumption_mean', 'Consumption_std', 'GHG_Savings_mean', 'GHG_Savings_std']

# # Standardize the features
# features = grouped_data[['Consumption_mean', 'Consumption_std', 'GHG_Savings_mean', 'GHG_Savings_std']]
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# # Number of initial clustering rounds and total rounds
# initial_clustering_rounds = 5
# total_rounds = 30
# cluster_counts = []

# # Perform initial clustering rounds
# for round_num in range(initial_clustering_rounds):
#     # Hierarchical clustering
#     linked = linkage(scaled_features, method='ward')

#     # Set a dynamic threshold to determine clusters
#     threshold = 1.5 - (round_num * 0.1)  # Adjust threshold dynamically

#     # Determine clusters
#     clusters = fcluster(linked, t=threshold, criterion='distance')
#     num_clusters = len(np.unique(clusters))
#     cluster_counts.append(num_clusters)

#     # Add cluster labels to the original grouped data
#     grouped_data['Cluster'] = clusters

#     # Optionally, you can print or log the results of each round
#     print(f'Clustering Round {round_num + 1}: {num_clusters} clusters')

# # Save the final clustering results for further training
# final_clusters = grouped_data[['St_Name', 'Cluster']]

# # Simulate training on the clustered data for the remaining rounds
# # For simplicity, we only track the number of clusters and assume that each cluster trains its model independently
# for round_num in range(initial_clustering_rounds, total_rounds):
#     # Here, we'd typically perform training for each cluster's model
#     # For demonstration, we assume the number of clusters remains the same after initial clustering
#     cluster_counts.append(num_clusters)
#     print(f'Training Round {round_num + 1}: {num_clusters} clusters')

# # Plot the number of clusters over the rounds
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, total_rounds + 1), cluster_counts, marker='*', linestyle='--', color='blue', markevery=1)
# plt.xlabel('Number of Communication Rounds')
# plt.ylabel('Number of Clusters')
# plt.title('Number of Clusters Over Rounds')
# plt.grid(True)
# plt.show()

# # Save the clustered data to a CSV file
# final_clusters.to_csv('clustered_charging_stations.csv', index=False)


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from scipy.cluster.hierarchy import linkage, fcluster
# import matplotlib.pyplot as plt

# # Load the dataset
# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Extract the relevant features: 'St_Name', 'Consumption', 'GHG_Savings'
# # Compute the number of transactions per charging station
# transaction_counts = data['St_Name'].value_counts().head(20)

# # Filter the data to include only the top 20 charging stations
# top_stations = transaction_counts.index
# filtered_data = data[data['St_Name'].isin(top_stations)]

# # Group by 'St_Name' and compute aggregate features
# grouped_data = filtered_data.groupby('St_Name').agg({
#     'Consumption': ['mean', 'std'],
#     'GHG_Savings': ['mean', 'std']
# }).reset_index()

# # Flatten the multi-index columns
# grouped_data.columns = ['St_Name', 'Consumption_mean', 'Consumption_std', 'GHG_Savings_mean', 'GHG_Savings_std']

# # Standardize the features
# features = grouped_data[['Consumption_mean', 'Consumption_std', 'GHG_Savings_mean', 'GHG_Savings_std']]
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

# # Number of total rounds
# total_rounds = 30
# cluster_counts = []

# # Perform clustering in each round
# for round_num in range(total_rounds):
#     # Hierarchical clustering
#     linked = linkage(scaled_features, method='ward')

#     # Set a dynamic threshold to determine clusters
#     threshold = 1.2 - (round_num * 0.03)  # Adjust threshold dynamically

#     # Determine clusters
#     clusters = fcluster(linked, t=threshold, criterion='distance')
#     num_clusters = len(np.unique(clusters))
#     cluster_counts.append(num_clusters)

#     # Add cluster labels to the original grouped data
#     grouped_data['Cluster'] = clusters

#     # Optionally, you can print or log the results of each round
#     print(f'Clustering Round {round_num + 1}: {num_clusters} clusters')

# # Save the final clustering results for further training
# final_clusters = grouped_data[['St_Name', 'Cluster']]

# # Plot the number of clusters over the rounds
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, total_rounds + 1), cluster_counts, marker='*', linestyle='--', color='blue', markevery=1)
# plt.xlabel('Number of Communication Rounds')
# plt.ylabel('Number of Clusters')
# plt.title('Number of Clusters Over Rounds')
# plt.grid(True)
# plt.show()

# # Save the clustered data to a CSV file
# final_clusters.to_csv('clustered_charging_stations.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('EVCSNB_cleaned.csv')

# Extract the relevant features: 'St_Name', 'Consumption', 'GHG_Savings'
# Compute the number of transactions per charging station
transaction_counts = data['St_Name'].value_counts().head(20)

# Filter the data to include only the top 20 charging stations
top_stations = transaction_counts.index
filtered_data = data[data['St_Name'].isin(top_stations)]

# Group by 'St_Name' and compute aggregate features
grouped_data = filtered_data.groupby('St_Name').agg({
    'Consumption': ['mean', 'std'],
    'GHG_Savings': ['mean', 'std']
}).reset_index()

# Flatten the multi-index columns
grouped_data.columns = ['St_Name', 'Consumption_mean', 'Consumption_std', 'GHG_Savings_mean', 'GHG_Savings_std']

# Standardize the features
features = grouped_data[['Consumption_mean', 'Consumption_std', 'GHG_Savings_mean', 'GHG_Savings_std']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Number of total rounds
total_rounds = 30
cluster_counts = []

# Initial Clustering using Ward's method (Hierarchical)
linked_initial = linkage(scaled_features, method='ward')
initial_threshold = 1.2  # Set an initial threshold for clustering
initial_clusters = fcluster(linked_initial, t=initial_threshold, criterion='distance')
num_initial_clusters = len(np.unique(initial_clusters))
cluster_counts.append(num_initial_clusters)

# Add initial cluster labels to the original grouped data
grouped_data['Cluster'] = initial_clusters

# Early stopping criteria
stable_rounds = 0  # To count stable rounds with no significant change in the number of clusters
convergence_threshold = 3  # Number of consecutive rounds with no change to stop

# Track if convergence is reached
convergence_reached = False

print(f'Initial Clustering: {num_initial_clusters} clusters')

# Perform clustering in subsequent rounds and dynamically adjust threshold
for round_num in range(1, total_rounds):
    if not convergence_reached:
        # Dynamically adjust the threshold to achieve faster convergence
        if round_num < 6:
            threshold = initial_threshold - (round_num * 0.1)  # Larger step size for initial rounds
        else:
            threshold = initial_threshold - (round_num * 0.02)  # Smaller step size for subsequent rounds

        # Perform hierarchical clustering based on the new threshold
        clusters = fcluster(linked_initial, t=threshold, criterion='distance')
        num_clusters = len(np.unique(clusters))

        # Early stopping criteria based on the stability of cluster numbers
        if num_clusters == cluster_counts[-1]:
            stable_rounds += 1
        else:
            stable_rounds = 0  # Reset stable rounds counter if the number of clusters changes

        # If stable for the required number of rounds, mark convergence
        if stable_rounds >= convergence_threshold:
            convergence_reached = True
            print(f'Converged at Round {round_num + 1} with {num_clusters} clusters.')
        
        cluster_counts.append(num_clusters)
    else:
        # After convergence, append the stable number of clusters for the remaining rounds
        cluster_counts.append(cluster_counts[-1])
    
    print(f'Clustering Round {round_num + 1}: {cluster_counts[-1]} clusters')

# Save the final clustering results for further training
final_clusters = grouped_data[['St_Name', 'Cluster']]

# Plot the number of clusters over the rounds
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cluster_counts) + 1), cluster_counts, marker='*', linestyle='--', color='blue', markevery=1)
plt.xlabel('Number of Communication Rounds')
plt.ylabel('Number of Clusters')
plt.title('Number of Clusters Over Rounds')
plt.grid(True)
plt.show()

# Save the clustered data to a CSV file
final_clusters.to_csv('optimized_clustered_charging_stations.csv', index=False)

