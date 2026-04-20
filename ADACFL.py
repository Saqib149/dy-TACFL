# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# # Load the dataset
# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Select the top 20 stations with the most transactions
# top_stations = data['St_Name'].value_counts().nlargest(20).index
# data = data[data['St_Name'].isin(top_stations)]

# # Preprocess the data: Extract more detailed time features
# data['Timestamp'] = pd.to_datetime(data['St_Time'])
# data['Hour'] = data['Timestamp'].dt.hour
# data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
# data['WeekOfYear'] = data['Timestamp'].dt.isocalendar().week
# data['DayOfYear'] = data['Timestamp'].dt.dayofyear

# # Assuming 'Consumption' is the target variable
# features = ['Hour', 'DayOfWeek', 'WeekOfYear', 'DayOfYear', 'Consumption']

# # Clean data
# data.dropna(subset=features, inplace=True)
# data[features] = data[features].apply(pd.to_numeric, errors='coerce', downcast='float')
# data.dropna(subset=features, inplace=True)

# # Prepare the features and target
# X = data[features].values
# y = data['Consumption'].values.reshape(-1, 1)

# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Define neural network model
# class EnhancedNN(nn.Module):
#     def __init__(self):
#         super(EnhancedNN, self).__init__()
#         self.fc1 = nn.Linear(len(features), 20)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(20, 10)
#         self.relu2 = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = self.relu1(self.fc1(x))
#         x = self.relu2(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # Function to train the model
# def train_model(model, data_loader):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     model.train()
#     for epoch in range(5):  # Train for 5 epochs
#         for inputs, targets in data_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#     return model

# # Function to calculate RMSE
# def calculate_rmse(model, data_loader):
#     model.eval()
#     predictions, actuals = [], []
#     with torch.no_grad():
#         for inputs, targets in data_loader:
#             outputs = model(inputs)
#             predictions.append(outputs.detach().numpy())
#             actuals.append(targets.numpy())
#     predictions = np.vstack(predictions)
#     actuals = np.vstack(actuals)
#     return np.sqrt(mean_squared_error(actuals, predictions))

# # Array to collect overall RMSE for each round
# overall_rmse_per_round = []

# # Run federated learning for 50 rounds
# for round in range(500):
#     rmse_scores = []
#     station_weights = []
#     for station in top_stations:
#         station_data = data[data['St_Name'] == station]
#         X_station = station_data[features].values.astype(np.float32)
#         y_station = station_data['Consumption'].values.reshape(-1, 1).astype(np.float32)
        
#         X_train, X_test, y_train, y_test = train_test_split(X_station, y_station, test_size=0.2, random_state=42)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
        
#         train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=10, shuffle=True)
#         test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=10, shuffle=False)
        
#         model = EnhancedNN()
#         model = train_model(model, train_loader)
#         weights = model.fc1.weight.data.numpy().flatten()
#         station_weights.append(weights)  # Collecting weights
#         rmse = calculate_rmse(model, test_loader)
#         rmse_scores.append(rmse)
    
#     # Calculate and collect the average RMSE for this round
#     average_rmse = np.mean(rmse_scores)
#     overall_rmse_per_round.append(average_rmse)
#     print(f"Round {round + 1}: Average RMSE = {average_rmse:.3f}")

# # Perform adaptive clustering with a higher threshold after all rounds
# station_weights = np.array(station_weights)
# clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=5.0)  # Increased threshold
# clusters = clustering.fit_predict(station_weights)

# # Print cluster assignments
# print("Cluster assignments:", clusters)

# # Plot RMSE over 50 rounds
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 500), overall_rmse_per_round, marker='o')
# plt.title('RMSE Over 50 Rounds of Federated Learning')
# plt.xlabel('Round Number')
# plt.ylabel('Average RMSE')
# plt.grid(True)
# plt.show()
# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# # Load the dataset
# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Select the top 20 stations with the most transactions
# top_stations = data['St_Name'].value_counts().nlargest(20).index
# data = data[data['St_Name'].isin(top_stations)]

# # Preprocess the data
# data['Timestamp'] = pd.to_datetime(data['St_Time'])
# data['Hour'] = data['Timestamp'].dt.hour
# data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
# data['WeekOfYear'] = data['Timestamp'].dt.isocalendar().week
# data['DayOfYear'] = data['Timestamp'].dt.dayofyear

# # Assuming 'Consumption' as the target variable
# features = ['Hour', 'DayOfWeek', 'WeekOfYear', 'DayOfYear', 'Consumption']
# data.dropna(subset=features, inplace=True)
# data[features] = data[features].apply(pd.to_numeric, errors='coerce', downcast='float')
# data.dropna(subset=features, inplace=True)

# # Prepare the features and target
# X = data[features].values
# y = data['Consumption'].values.reshape(-1, 1)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Define neural network model
# class EnhancedNN(nn.Module):
#     def __init__(self):
#         super(EnhancedNN, self).__init__()
#         self.fc1 = nn.Linear(len(features), 50)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(50, 10)
#         self.relu2 = nn.ReLU()
#         #self.dropout = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = self.relu1(self.fc1(x))
#         x = self.relu2(self.fc2(x))
#         #x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # Train the model
# def train_model(model, data_loader):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     model.train()
#     for epoch in range(1):
#         for inputs, targets in data_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#     return model

# # Calculate RMSE
# def calculate_rmse(model, data_loader):
#     model.eval()
#     predictions, actuals = [], []
#     with torch.no_grad():
#         for inputs, targets in data_loader:
#             outputs = model(inputs)
#             predictions.append(outputs.detach().numpy())
#             actuals.append(targets.numpy())
#     predictions = np.vstack(predictions)
#     actuals = np.vstack(actuals)
#     return np.sqrt(mean_squared_error(actuals, predictions))

# # Initialize variables to store per-round data
# overall_rmse_per_round = []
# cluster_assignments_per_round = []

# # Run federated learning for 50 rounds
# for round in range(200):
#     rmse_scores = []
#     station_weights = []
#     for station in top_stations:
#         station_data = data[data['St_Name'] == station]
#         X_station = station_data[features].values.astype(np.float32)
#         y_station = station_data['Consumption'].values.reshape(-1, 1).astype(np.float32)
        
#         X_train, X_test, y_train, y_test = train_test_split(X_station, y_station, test_size=0.2, random_state=42 + round)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
        
#         train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=False)
#         test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=32, shuffle=False)
        
#         model = EnhancedNN()
#         model = train_model(model, train_loader)
#         weights = model.fc1.weight.data.numpy().flatten()
#         station_weights.append(weights)
#         rmse = calculate_rmse(model, test_loader)
#         #print(f"Round {round + 1}: Average RMSE = {rmse:.3f}")

#         rmse_scores.append(rmse)


#     average_rmse = np.mean(rmse_scores)
#     overall_rmse_per_round.append(average_rmse)
#     print(f"Round {round + 1}: Average RMSE = {average_rmse:.3f}")

#     # Perform clustering after each round
#     station_weights = np.array(station_weights)
#     clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=5.0)
#     clusters = clustering.fit_predict(station_weights)
#     cluster_assignments_per_round.append(clusters)

# # Select 5 stations for tracking cluster transitions
# selected_stations = top_stations[:10]  # Select first 5 stations for simplicity
# transitions = {station: [] for station in selected_stations}
# for round_clusters in cluster_assignments_per_round:
#     for idx, station in enumerate(selected_stations):
#         transitions[station].append(round_clusters[idx])

# # # Plot the cluster transitions for selected stations
# # plt.figure(figsize=(15, 5))
# # for station, changes in transitions.items():
# #     plt.plot(range(1, 51), changes, label=f'Station {station}', marker='o', linestyle='-')

# # plt.title('Cluster Transitions for Selected Charging Stations Over 50 Rounds')
# # plt.xlabel('Round Number')
# # plt.ylabel('Cluster Assignment')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
# Plot RMSE over 50 rounds
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 51), overall_rmse_per_round, marker='o')
# plt.title('RMSE Over 50 Rounds of Federated Learning')
# plt.xlabel('Round Number')
# plt.ylabel('Average RMSE')
# plt.grid(True)
# plt.show()

# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# # Load the dataset
# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Select the top 20 stations with the most transactions
# top_stations = data['St_Name'].value_counts().nlargest(20).index
# data = data[data['St_Name'].isin(top_stations)]

# # Preprocess the data: Extract more detailed time features
# data['Timestamp'] = pd.to_datetime(data['St_Time'])
# data['Hour'] = data['Timestamp'].dt.hour
# data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
# data['WeekOfYear'] = data['Timestamp'].dt.isocalendar().week
# data['DayOfYear'] = data['Timestamp'].dt.dayofyear

# # Assuming 'Consumption' is the target variable
# features = ['Hour', 'DayOfWeek', 'WeekOfYear', 'DayOfYear', 'Consumption']
# data.dropna(subset=features, inplace=True)
# data[features] = data[features].apply(pd.to_numeric, errors='coerce', downcast='float')
# data.dropna(subset=features, inplace=True)

# # Prepare the features and target
# X = data[features].values
# y = data['Consumption'].values.reshape(-1, 1)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Define neural network model
# class EnhancedNN(nn.Module):
#     def __init__(self):
#         super(EnhancedNN, self).__init__()
#         self.fc1 = nn.Linear(len(features), 20)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(20, 10)
#         self.relu2 = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = self.relu1(self.fc1(x))
#         x = self.relu2(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # Function to train the model
# def train_model(model, data_loader):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     model.train()
#     for epoch in range(5):
#         for inputs, targets in data_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#     return model

# # Function to calculate RMSE
# def calculate_rmse(model, data_loader):
#     model.eval()
#     predictions, actuals = [], []
#     with torch.no_grad():
#         for inputs, targets in data_loader:
#             outputs = model(inputs)
#             predictions.append(outputs.detach().numpy())
#             actuals.append(targets.numpy())
#     predictions = np.vstack(predictions)
#     actuals = np.vstack(actuals)
#     return np.sqrt(mean_squared_error(actuals, predictions))

# # Array to collect overall RMSE for each round
# overall_rmse_per_round = []

# # Run federated learning for 500 rounds
# for round in range(500):
#     rmse_scores = []
#     station_weights = []
#     for station in top_stations:
#         station_data = data[data['St_Name'] == station]
#         X_station = station_data[features].values.astype(np.float32)
#         y_station = station_data['Consumption'].values.reshape(-1, 1).astype(np.float32)
        
#         X_train, X_test, y_train, y_test = train_test_split(X_station, y_station, test_size=0.2, random_state=42)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
        
#         train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=10, shuffle=True)
#         test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=10, shuffle=False)
        
#         model = EnhancedNN()
#         model = train_model(model, train_loader)
#         weights = model.fc1.weight.data.numpy().flatten()
#         station_weights.append(weights)
#         rmse = calculate_rmse(model, test_loader)
#         rmse_scores.append(rmse)
    
#     average_rmse = np.mean(rmse_scores)
#     overall_rmse_per_round.append(average_rmse)
#     print(f"Round {round + 1}: Average RMSE = {average_rmse:.3f}")

# # Plot RMSE over 500 rounds
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 501), overall_rmse_per_round, marker='o')
# plt.title('RMSE Over 500 Rounds of Federated Learning')
# plt.xlabel('Round Number')
# plt.ylabel('Average RMSE')
# plt.grid(True)
# plt.show()
# *****************************************************************************************************************
# *****************************************************************************************************************
# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# # Load the dataset
# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Select the top 20 stations with the most transactions
# data = data[data['St_Name'].isin(data['St_Name'].value_counts().nlargest(20).index)]

# # Prepare datetime and features
# data['Timestamp'] = pd.to_datetime(data['St_Time'])
# data['Hour'] = data['Timestamp'].dt.hour
# data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
# data['WeekOfYear'] = data['Timestamp'].dt.isocalendar().week
# data['DayOfYear'] = data['Timestamp'].dt.dayofyear
# features = ['Hour', 'DayOfWeek', 'WeekOfYear', 'DayOfYear', 'Consumption']
# data = data.dropna(subset=features)

# # Scale features
# X = data[features].values
# y = data['Consumption'].values.reshape(-1, 1)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Define a simple neural network
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(5, 10)  # Assuming 5 features
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)

# # Setup for training
# model = SimpleNN()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# # Prepare data loaders
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
# train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=10)
# test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=10)

# # Train model and collect losses
# losses = []
# for epoch in range(50):  # Run for more epochs if needed
#     model.train()
#     for inputs, targets in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#     scheduler.step()
#     model.eval()
#     with torch.no_grad():
#         total_loss = 0
#         count = 0
#         for inputs, targets in test_loader:
#             outputs = model(inputs)
#             total_loss += criterion(outputs, targets).item()
#             count += 1
#         avg_loss = total_loss / count
#         losses.append(avg_loss)

# # Smooth the losses for plotting
# def smooth_curve(points, factor=0.8):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points

# smoothed_losses = smooth_curve(losses)

# # Plot the smoothed losses
# plt.figure(figsize=(10, 6))
# plt.plot(smoothed_losses, label='Proposed Method')
# plt.title('Loss Over Training Rounds')
# plt.xlabel('Training Rounds')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()
# ***************************************************************************************************
# ******************************************************************************************************
# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# # Load the dataset
# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Select the top 20 stations with the highest number of transactions
# top_stations = data['St_Name'].value_counts().nlargest(20).index
# data = data[data['St_Name'].isin(top_stations)]

# # Prepare datetime and other features
# data['Timestamp'] = pd.to_datetime(data['St_Time'])
# data['Hour'] = data['Timestamp'].dt.hour
# data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
# data['WeekOfYear'] = data['Timestamp'].dt.isocalendar().week
# data['DayOfYear'] = data['Timestamp'].dt.dayofyear

# # Define features and target
# features = ['Hour', 'DayOfWeek', 'WeekOfYear', 'DayOfYear']
# target = 'Consumption'

# # Clean and scale data
# data.dropna(subset=features + [target], inplace=True)
# X = pd.get_dummies(data[features])
# y = data[target].values
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# class SimpleNN(nn.Module):
#     def __init__(self, input_size):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 10)  # Simple layer sizes
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(10, 1)  # Predicting consumption

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)
# def train_model(data_loader, model, epochs=10, learning_rate=0.01):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     for _ in range(epochs):
#         for inputs, targets in data_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets.view(-1, 1))
#             loss.backward()
#             optimizer.step()

# def evaluate_model(data_loader, model):
#     criterion = nn.MSELoss()
#     with torch.no_grad():
#         losses = []
#         for inputs, targets in data_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, targets.view(-1, 1))
#             losses.append(loss.item())
#         return np.sqrt(np.mean(losses))

# def federated_learning(X, y, rounds=500):
#     models = {station: SimpleNN(X.shape[1]) for station in top_stations}
#     rmse_scores = []
    
#     for round in range(rounds):
#         # Train each model
#         for station in top_stations:
#             model = models[station]
#             mask = data['St_Name'] == station
#             loader = DataLoader(TensorDataset(torch.tensor(X[mask], dtype=torch.float32),
#                                               torch.tensor(y[mask], dtype=torch.float32)), batch_size=10, shuffle=True)
#             train_model(loader, model)
        
#         # Aggregate weights and apply clustering
#         weights = np.array([models[station].fc1.weight.data.numpy().flatten() for station in top_stations])
#         clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
#         labels = clustering.fit_predict(weights)
        
#         # Evaluate models
#         total_rmse = []
#         for station in top_stations:
#             model = models[station]
#             mask = data['St_Name'] == station
#             loader = DataLoader(TensorDataset(torch.tensor(X[mask], dtype=torch.float32),
#                                               torch.tensor(y[mask], dtype=torch.float32)), batch_size=10, shuffle=False)
#             rmse = evaluate_model(loader, model)
#             total_rmse.append(rmse)
        
#         avg_rmse = np.mean(total_rmse)
#         rmse_scores.append(avg_rmse)
#         print(f'Round {round + 1}: Avg RMSE = {avg_rmse:.4f}')
    
#     return rmse_scores

# # Running the adaptive clustering and training process
# rmse_scores = federated_learning(X_scaled, y)

# # Plotting the RMSE over rounds
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 501), rmse_scores, label='Adaptive Clustered Federated Learning RMSE')
# plt.title('Average RMSE Over 500 Rounds of Federated Training')
# plt.xlabel('Round')
# plt.ylabel('Average RMSE')
# plt.legend()
# plt.grid(True)
# plt.show()
# *************************************************************************************************
#******************************
# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# # Load the dataset
# data = pd.read_csv('EVCSNB_cleaned.csv')

# # Select the top 20 stations based on transaction counts
# top_stations = data['St_Name'].value_counts().nlargest(20).index
# data = data[data['St_Name'].isin(top_stations)]

# # Extract and engineer features
# data['Timestamp'] = pd.to_datetime(data['St_Time'])
# data['Hour'] = data['Timestamp'].dt.hour
# data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
# data['WeekOfYear'] = data['Timestamp'].dt.isocalendar().week
# data['DayOfYear'] = data['Timestamp'].dt.dayofyear
# features = ['Hour', 'DayOfWeek', 'WeekOfYear', 'DayOfYear']
# target = 'Consumption'

# # Clean and scale data
# data.dropna(subset=features + [target], inplace=True)
# X = pd.get_dummies(data[features])
# y = data[target].values
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Define an enhanced neural network model
# class EnhancedNN(nn.Module):
#     def __init__(self, input_size):
#         super(EnhancedNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 20)
#         self.relu1 = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(20, 10)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
#         return self.fc3(x)

# # Train and evaluate the model
# def train_model(data_loader, model, epochs=15, learning_rate=0.005):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#     for _ in range(epochs):
#         for inputs, targets in data_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets.view(-1, 1))
#             loss.backward()
#             optimizer.step()

# def evaluate_model(data_loader, model):
#     criterion = nn.MSELoss()
#     with torch.no_grad():
#         losses = []
#         for inputs, targets in data_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, targets.view(-1, 1))
#             losses.append(loss.item())
#         return np.sqrt(np.mean(losses))

# # Federated learning with adaptive clustering
# def federated_learning(X, y, rounds=30):
#     models = {station: EnhancedNN(X.shape[1]) for station in top_stations}
#     rmse_scores = []
    
#     for round in range(rounds):
#         # Train each model
#         for station in top_stations:
#             model = models[station]
#             mask = data['St_Name'] == station
#             loader = DataLoader(TensorDataset(torch.tensor(X[mask], dtype=torch.float32),
#                                               torch.tensor(y[mask], dtype=torch.float32)), batch_size=16, shuffle=True)
#             train_model(loader, model)
        
#         # Aggregate weights and apply clustering
#         weights = np.array([models[station].fc1.weight.data.numpy().flatten() for station in top_stations])
#         clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
#         labels = clustering.fit_predict(weights)
        
#         # Evaluate models
#         total_rmse = []
#         for station in top_stations:
#             model = models[station]
#             mask = data['St_Name'] == station
#             loader = DataLoader(TensorDataset(torch.tensor(X[mask], dtype=torch.float32),
#                                               torch.tensor(y[mask], dtype=torch.float32)), batch_size=16, shuffle=False)
#             rmse = evaluate_model(loader, model)
#             total_rmse.append(rmse)
        
#         avg_rmse = np.mean(total_rmse)
#         rmse_scores.append(avg_rmse)
#         print(f'Round {round + 1}: Avg RMSE = {avg_rmse:.4f}')
    
#     return rmse_scores

# # Running the adaptive clustering and training process
# rmse_scores = federated_learning(X_scaled, y)

# # Plotting the RMSE over rounds
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 301), rmse_scores, label='Adaptive Clustered Federated Learning RMSE')
# plt.title('Average RMSE Over 500 Rounds of Federated Training')
# plt.xlabel('Round')
# plt.ylabel('Average RMSE')
# plt.legend()
# plt.grid(True)
# plt.show()

#**************************************************************************************************************************
#**************************************************************************************************************************
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
