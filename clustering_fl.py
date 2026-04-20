import gc
import pickle
import logging
import argparse
import torch
import os
import time
import datetime
import copy
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from multiprocessing import pool, cpu_count
from tqdm.auto import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from torchvision import models, transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, jaccard_score
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Option 2: Compile with TORCH_USE_CUDA_DSA
os.environ['TORCH_USE_CUDA_DSA'] = '1'


torch.set_float32_matmul_precision('medium') # low, medium, high

# arguments
# parser = argparse.ArgumentParser(description='byol-lightning-test')
# parser.add_argument('--data_path',  type=str, required = True,
#                        help='path to your folder of images for self-supervised learning')
# parser.add_argument('--num_clients', default=10, type=int, required = False, help='num_clients')
# parser.add_argument('--fraction', default=0.5, type=float, required = False, help='fraction')
# parser.add_argument('--num_rounds', default=500, type=int, required = False, help='num_rounds')
# parser.add_argument('--iid', default=True, type=bool, required = False, help='iid')
# parser.add_argument('--local_epochs', default=2, type=int, required = False, help='local_epochs')
# parser.add_argument('--batch_size', default=32, type=int, required = False, help='batch_size')
# parser.add_argument('--num_clusters', default=3, type=int, required = False, help='num_clusters')
# parser.add_argument('--loss_func', default='mse', type=str, required = False, help='loss_func')



# args = parser.parse_args()
# arguments
parser = argparse.ArgumentParser(description='byol-lightning-test')
# Set a default value for data_path
parser.add_argument('--data_path', type=str, default='data/weekly_new', required=False,
                       help='path to your folder of images for self-supervised learning')
parser.add_argument('--num_clients', default=10, type=int, required=False, help='num_clients')
parser.add_argument('--fraction', default=0.5, type=float, required=False, help='fraction')
parser.add_argument('--num_rounds', default=200, type=int, required=False, help='num_rounds')
parser.add_argument('--iid', default=True, type=bool, required=False, help='iid')
parser.add_argument('--local_epochs', default=1, type=int, required=False, help='local_epochs')
parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch_size')
parser.add_argument('--num_clusters', default=3, type=int, required=False, help='num_clusters')
parser.add_argument('--loss_func', default='mse', type=str, required=False, help='loss_func')

args = parser.parse_args()


# python train_fl_pretext.py --data_path data/weekly_new --num_clients 3 --fraction 1.0 --num_rounds 100 --batch_size 32 --local_epochs 1 --num_clusters 3 --loss_func mse

np.random.seed(42)

logger = logging.getLogger(__name__)

seed= 5959
device= "cuda"

init_type= "xavier"
init_gain= 1.0
gpu_ids= [1]

log_path= "./log/"
log_name=  "FL.log"

# constants
LR = 1e-6
GPUS = [1]
IMAGE_SIZE = 256
NUM_WORKERS = min(args.batch_size, multiprocessing.cpu_count())

feature_cols = ['St_Name', 'Postal_Code', 'week_number', 'year', 'TC_Time']
target_col = 'Consumption'
lookback = 4 # 5 hours lookback to make prediction

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, X_batch):
        hidden, carry = torch.randn(self.n_layers, len(X_batch), self.hidden_dim), torch.randn(self.n_layers, len(X_batch), self.hidden_dim)
        output, (hidden, carry) = self.lstm(X_batch, (hidden, carry))
        return self.linear(output[:,-1])

def create_datasets(data_path, num_clients, iid):
    file_list = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    data_dfs = []
    X_organized, Y_organized = [], []
    # Print the list of files
    for file in file_list:
        data_df = pd.read_csv(f"data/weekly_new/{file}")
        data_df['Postal_Code'] = data_df['Postal_Code'].astype('float64')
        data_df['week_number'] = data_df['week_number'].astype('float64')
        data_df['year'] = data_df['year'].astype('float64')
        data_df = data_df.dropna()
        X = data_df[feature_cols].values
        Y = data_df[target_col].values
    
        prev_len = len(Y_organized)
        n_features = X.shape[1]
        for i in range(0, X.shape[0]-lookback, 1):
            X_organized.append(X[i:i+lookback])
            Y_organized.append(Y[i+lookback])

        # print(f'{file}: {len(Y_organized) - prev_len}')
        data_dfs.append(data_df)
        
    X_organized, Y_organized = np.array(X_organized), np.array(Y_organized)
    X_organized, Y_organized = torch.tensor(X_organized, dtype=torch.float32), torch.tensor(Y_organized, dtype=torch.float32)


    X_train, X_test, Y_train, Y_test = train_test_split(X_organized, Y_organized, test_size=0.1, random_state=42)
    mean, std = Y_train.mean(), Y_train.std()
    print("Mean : {:.2f}, Standard Deviation : {:.2f}".format(mean, std))

    Y_train_scaled, Y_test_scaled = (Y_train - mean)/std , (Y_test-mean)/std

    print('Y_train_scaled.min()', Y_train_scaled.min(), 'Y_train_scaled.max()', Y_train_scaled.max(), 'Y_test_scaled.min()', Y_test_scaled.min(), 'Y_test_scaled.max()', Y_test_scaled.max())

    X_train_splits = torch.chunk(X_train, num_clients)
    Y_train_scaled_splits = torch.chunk(Y_train_scaled, num_clients)

    train_datasets = []
    # Print the resulting chunks
    for i, (X_chunk, Y_chunk) in enumerate(zip(X_train_splits, Y_train_scaled_splits)):
        train_dataset = TensorDataset(X_chunk, Y_chunk)
        train_datasets.append(train_dataset)
        
    test_dataset  = TensorDataset(X_test,  Y_test_scaled)
    
    return train_datasets, test_dataset

# Function to count rows in a CSV file
def count_rows(file_path):
    try:
        df = pd.read_csv(file_path)
        return len(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0
    
def create_datasets_v1(data_path, num_clients, iid):
    file_list = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    file_list.sort(key=lambda file: count_rows(os.path.join('"data/weekly_new', file)), reverse=True)

    train_datasets, test_data_x, test_data_y = [], [], []
    
    # Print the list of files
    for file in file_list[:num_clients]:
        X_train, Y_train = [], []
        data_df = pd.read_csv(f"data/weekly_new/{file}")
        data_df['Postal_Code'] = data_df['Postal_Code'].astype('float64')
        data_df['week_number'] = data_df['week_number'].astype('float64')
        data_df['year'] = data_df['year'].astype('float64')
        data_df = data_df.dropna()
        X = data_df[feature_cols].values
        Y = data_df[target_col].values
    
        for i in range(0, X.shape[0]-lookback, 1):
            X_train.append(X[i:i+lookback])
            Y_train.append(Y[i+lookback])

        
        X_train, Y_train = np.array(X_train), np.array(Y_train)

        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
        X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)

        mean, std = Y_train.mean(), Y_train.std()

        Y_train_scaled = (Y_train - mean)/std
        Y_test_scaled = (Y_test-mean.numpy())/std.numpy()
        
        train_dataset = TensorDataset(X_train, Y_train_scaled)
        train_datasets.append(train_dataset)
        test_data_x.append(X_test)
        test_data_y.append(Y_test_scaled)
    
    X_test, Y_test_scaled = torch.tensor(np.concatenate(test_data_x), dtype=torch.float32), torch.tensor(np.concatenate(test_data_y), dtype=torch.float32)
    test_dataset  = TensorDataset(X_test,  Y_test_scaled)
    
    return train_datasets, test_dataset

class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device, cluster_num):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.cluster_num = cluster_num
        self.assigned_clusters = []
        self.assigned_clusters.append(cluster_num)
        self.data = local_data
        self.device = device
        self.__model = None

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def assign_cluster(self, num):
        self.cluster_num = num
        self.assigned_clusters.append(num)
        
    def setup(self, args):
        self.args = args
        """Set up common configuration of each client; called by center server."""
        self.local_epochs = args.local_epochs
        self.dataloader = DataLoader(self.data, shuffle=False, batch_size=args.batch_size)

    def client_update(self):
        learning_rate = 1e-3
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        if self.args.loss_func == 'rmse':
            loss_fn = RMSELoss() 
        else: 
            loss_fn = nn.MSELoss()

        for i in range(1, self.local_epochs+1):
            losses = []
            for X, Y in tqdm(self.dataloader):
                Y_preds = self.model(X)

                loss = loss_fn(Y_preds.ravel(), Y)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        return torch.tensor(losses).mean().item()

class Server(object):
    
    def __init__(self, writer, args):
        self.args = args
        self.clients = None
        self._round = 0
        self.writer = writer
        
        model = LSTMRegressor(n_features=len(feature_cols), hidden_dim=10, n_layers=2)
        self.models = [copy.deepcopy(model) for _ in range(args.num_clusters)]
        
        self.seed = seed
        self.device = device
        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.models[0].parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        local_datasets, test_dataset = create_datasets_v1(self.args.data_path, self.args.num_clients, self.args.iid)
        
        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)

        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        
        self.dataloader = DataLoader(test_dataset,  shuffle=False, batch_size=self.args.batch_size)

        # configure detailed settings for client upate and 
        self.setup_clients()
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clusters = [i for i in range(self.args.num_clusters)]
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device, cluster_num=np.random.choice(clusters))
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.args.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(self.args)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.args.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.args.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.models[client.cluster_num])

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.args.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.models[self.clients[idx].cluster_num])
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.args.fraction * self.args.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.args.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_sizes = [0 for _ in self.models]
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update()
            selected_total_sizes[self.clients[idx].cluster_num] += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_sizes)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_sizes

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        cluster_averaged_weights = [OrderedDict() for _ in self.models]
        ctrs = [0 for _ in self.models]
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            client = self.clients[idx]
            local_weights = client.model.state_dict()
            c_id = client.cluster_num
            
            for key in self.models[0].state_dict().keys():
                if len(cluster_averaged_weights[c_id].keys()) != len(local_weights.keys()):
                    cluster_averaged_weights[c_id][key] = coefficients[c_id][ctrs[c_id]] * local_weights[key].detach()
                else:
                    cluster_averaged_weights[c_id][key] += coefficients[c_id][ctrs[c_id]] * local_weights[key].detach()
            ctrs[c_id] += 1
            
        for i, averaged_weights in enumerate(cluster_averaged_weights):
            self.models[i].load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        selected_total_sizes = self.update_selected_clients(sampled_client_indices)

        mixing_coefficients = [list() for _ in self.models]
        # calculate averaging coefficient of weights
        for idx in sampled_client_indices:
            client = self.clients[idx]
            mixing_coefficients[client.cluster_num].append(len(client) / selected_total_sizes[client.cluster_num])

        # print('mixing_coefficients: ', mixing_coefficients)
        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
        self.evaluate_clients_model()
        self.assign_cluster()
        
    def assign_cluster(self):
        X, Y = [], []
        
        for client in self.clients:
            weight = client.model.state_dict()['linear.weight'].detach().numpy().flatten()
            X.append(weight)
            Y.append(client.cluster_num)
        kmeans = KMeans(n_clusters=self.args.num_clusters)
        # kmeans = TimeSeriesKMeans(n_clusters=self.args.num_clusters, metric="dtw")
        kmeans.fit(X)
        print('silhouette_score:', silhouette_score(X, kmeans.labels_))
        print('jaccard_score:', jaccard_score(kmeans.labels_, Y, average='macro'))
        for i, cluster_id in enumerate(kmeans.labels_):
            self.clients[i].assign_cluster(cluster_id)
        
    def evaluate_clients_model(self):
        if self.args.loss_func == 'rmse':
            loss_fn = RMSELoss() 
        else: 
            loss_fn = nn.MSELoss()
        for client in self.clients:
            with torch.no_grad():
                losses = []
                for X, Y in self.dataloader:
                    preds = client.model(X)
                    loss = loss_fn(preds.ravel(), Y)
                    losses.append(loss.item())
                print("Client {} Loss : {:.3f}".format(client.id, torch.tensor(losses).mean()))
            
        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        if self.args.loss_func == 'rmse':
            loss_fn = RMSELoss() 
        else: 
            loss_fn = nn.MSELoss()
            
        losses = []
        for i, model in enumerate(self.models):
            with torch.no_grad():
                cluster_losses = []
                for X, Y in self.dataloader:
                    preds = model(X)
                    loss = loss_fn(preds.ravel(), Y)
                    cluster_losses.append(loss.item())
                print("Cluster {} Loss : {:.3f}".format(i, torch.tensor(cluster_losses).mean()))
                losses.append(torch.tensor(cluster_losses).mean().item())
        return torch.tensor(losses).mean().item()

    def clients_assigned_clusters(self):
        plt.figure(figsize=(18, 6))
        for i, client in enumerate(self.clients):
            print(f'Client {i} assigned clusters:', client.assigned_clusters)
            assignments = client.assigned_clusters
            plt.plot(assignments, '-o', label=client.id)            

        plt.xticks(ticks=range(self.args.num_rounds), labels=[f"{i+1}" for i in range(self.args.num_rounds)])
        plt.yticks(ticks=range(self.args.num_clusters))
        plt.title("Cluster Assignments Across Rounds")
        plt.xlabel("Rounds")
        plt.ylabel("Cluster")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('graphs/cluster_assignments_across_rounds.pdf')
    
    def plot_globel_loss(self):
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.grid()

        # Sample data
        x = [i for i in range(len(self.results['loss']))]
        y = self.results['loss']

        # Create the line graph
        plt.plot(x, y)


        # Add labels to the axes and a title
        plt.xlabel('mse loss')
        plt.ylabel('epoch')
        plt.title('Globel loss')
        plt.show()

        # Show the graph
        plt.savefig('graphs/global_validation_loss_with_clustring.pdf')
           
            
    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}
        for r in range(self.args.num_rounds):
            self._round = r + 1
            print('======================== Training started. ============================')
            self.train_federated_model()
            print('======================== Training completed. ============================')
            print('======================== Evaluation started. ============================')
            test_loss = self.evaluate_global_model()
            print('======================== Evaluation completed. ============================')
            self.results['loss'].append(test_loss)
            
            self.writer.add_scalars(
                'Loss',
                {f"clients_{self.args.num_clients}_C_{self.args.fraction}, E_{self.args.local_epochs}, B_{self.args.batch_size}, loss_func_{self.args.loss_func}, clusters_{self.args.num_clusters}": test_loss},
                self._round
                )
            
            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()
            
        self.transmit_model()
        self.clients_assigned_clusters()
        self.plot_globel_loss()


if __name__ == "__main__":
   
    # modify log_path to contain current time
    log_path = os.path.join(log_path, str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_path, filename_suffix="FL")

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(log_path, log_name), level=logging.INFO, format="[%(levelname)s](%(asctime)s) %(message)s", datefmt="%Y/%m/%d/ %I:%M:%S %p")
    
    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)

    # initialize federated learning 
    central_server = Server(writer, args)
    central_server.setup()

    # do federated learning
    central_server.fit()

    # save resulting losses and metrics
    with open(os.path.join(log_path, "result.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)
     
    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); exit()