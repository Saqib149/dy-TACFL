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

