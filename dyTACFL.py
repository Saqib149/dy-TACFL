import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── HYPERPARAMETERS ───────────────────────────────────────────────
LOOKBACK      = 23
NUM_CLUSTERS  = 3
NUM_ROUNDS    = 100
LOCAL_EPOCHS  = 3
LEARNING_RATE = 0.001
BATCH_SIZE    = 64

FEATURE_COLS  = ['St_Name', 'Postal_Code', 'week_number', 'year', 'TC_Time']
TARGET_COL    = 'Consumption'

# ── SEED ──────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# ── DATA ──────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    df['St_Time']     = pd.to_datetime(df['St_Time'])
    df['week_number'] = df['St_Time'].dt.isocalendar().week.astype(int)
    df['year']        = df['St_Time'].dt.year - df['St_Time'].dt.year.min()  # relative year
    df['TC_Time']     = pd.to_timedelta(df['TC_Time']).dt.total_seconds()
    df['St_Name']     = df['St_Name'].astype('category').cat.codes
    df['Postal_Code'] = df['Postal_Code'].astype('category').cat.codes

    # Scale ALL features — critical for LSTM convergence
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS].values)

    # Build per-station datasets for proper non-IID FL
    # Sort each station by time so sequences are temporally ordered
    station_datasets = {}
    for sid in df['St_Name'].unique():
        sdf = df[df['St_Name'] == sid].sort_values('St_Time').reset_index(drop=True)
        X = sdf[FEATURE_COLS].values.astype(float)
        y = sdf[TARGET_COL].values.astype(float)

        if len(X) <= LOOKBACK:
            continue

        X_seq, y_seq = [], []
        for i in range(len(X) - LOOKBACK):
            X_seq.append(X[i:i + LOOKBACK])
            y_seq.append(y[i + LOOKBACK])

        station_datasets[sid] = (np.array(X_seq), np.array(y_seq))

    return station_datasets

# ── MODEL ─────────────────────────────────────────────────────────
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── CLIENT ────────────────────────────────────────────────────────
class Client:
    def __init__(self, train_data, station_id):
        self.train_data = train_data
        self.station_id = station_id
        self.model      = LSTMRegressor()

    def train(self, cluster_weights):
        self.model.load_state_dict(copy.deepcopy(cluster_weights))
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        loss_fn   = nn.MSELoss()
        loader    = DataLoader(self.train_data, batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(LOCAL_EPOCHS):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                pred = self.model(X_b).squeeze()
                loss = loss_fn(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

        return copy.deepcopy(self.model.state_dict()), len(self.train_data)

    def weight_vector(self):
        # Use FC layer weights for clustering (fast: 65 dims vs 51K)
        return torch.cat([p.detach().view(-1)
                          for p in self.model.fc.parameters()]).numpy()

# ── SERVER ────────────────────────────────────────────────────────
class Server:
    def __init__(self, clients, test_data, y_std, num_clusters=NUM_CLUSTERS):
        self.clients        = clients
        self.test_data      = test_data
        self.y_std          = y_std
        self.num_clusters   = num_clusters
        self.cluster_models = [LSTMRegressor() for _ in range(num_clusters)]
        self.assignments    = np.array([i % num_clusters for i in range(len(clients))])
        self.rmse_history   = []
        self.sil_history    = []

    def _fedavg(self, updates, sizes):
        total = sum(sizes)
        agg   = {}
        for key in updates[0]:
            agg[key] = sum(w[key] * (n / total) for w, n in zip(updates, sizes))
        return agg

    def _recluster(self):
        feats = np.stack([c.weight_vector() for c in self.clients])
        km    = KMeans(n_clusters=self.num_clusters, n_init=3, random_state=42).fit(feats)
        self.assignments = km.labels_
        return silhouette_score(feats, km.labels_)

    def _evaluate(self):
        loss_fn  = nn.MSELoss(reduction='sum')
        loader   = DataLoader(self.test_data, batch_size=128)
        best_mse = float('inf')

        for model in self.cluster_models:
            model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for X_b, y_b in loader:
                    total += loss_fn(model(X_b).squeeze(), y_b).item()
                    n     += len(y_b)
            mse = total / n
            if mse < best_mse:
                best_mse = mse

        rmse_norm   = np.sqrt(best_mse)
        rmse_actual = rmse_norm * self.y_std
        return rmse_norm, rmse_actual

    def train(self, rounds=NUM_ROUNDS):
        for r in range(rounds):
            updates = [[] for _ in range(self.num_clusters)]
            sizes   = [[] for _ in range(self.num_clusters)]

            for i, client in enumerate(self.clients):
                cid = self.assignments[i]
                w, n = client.train(self.cluster_models[cid].state_dict())
                updates[cid].append(w)
                sizes[cid].append(n)

            for c in range(self.num_clusters):
                if updates[c]:
                    self.cluster_models[c].load_state_dict(
                        self._fedavg(updates[c], sizes[c])
                    )

            sil = self._recluster()
            rmse_norm, rmse_actual = self._evaluate()

            self.rmse_history.append(rmse_actual)
            self.sil_history.append(sil)
            print(f"Round {r+1:3d} | RMSE: {rmse_actual:.4f} kWh | "
                  f"RMSE (norm): {rmse_norm:.4f} | Silhouette: {sil:.4f}")

# ── MAIN ──────────────────────────────────────────────────────────
station_datasets = load_data("EVCSBO.csv")
print(f"Loaded {len(station_datasets)} stations as clients\n")

# Split each station into train/test, collect all test data centrally
clients   = []
all_X_test, all_y_test = [], []
all_y_train = []

for sid, (X, y) in station_datasets.items():
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42)
    all_y_train.append(y_tr)
    all_X_test.append(X_te)
    all_y_test.append(y_te)

# Compute global normalisation stats from all training targets
y_train_all = np.concatenate(all_y_train)
y_mean, y_std = y_train_all.mean(), y_train_all.std()
print(f"Target normalisation — mean: {y_mean:.3f}, std: {y_std:.3f}\n")

# Build client objects with normalised targets
for i, (sid, (X, y)) in enumerate(station_datasets.items()):
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.1, random_state=42)
    y_tr_norm = (y_tr - y_mean) / y_std
    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr_norm, dtype=torch.float32)
    )
    clients.append(Client(ds, station_id=sid))

# Central test set (normalised)
X_test_all = np.concatenate(all_X_test)
y_test_all = (np.concatenate(all_y_test) - y_mean) / y_std
test_ds = TensorDataset(
    torch.tensor(X_test_all, dtype=torch.float32),
    torch.tensor(y_test_all, dtype=torch.float32)
)

print(f"Clients: {len(clients)} | Test samples: {len(test_ds)}\n")

server = Server(clients, test_ds, y_std=y_std)
server.train(rounds=NUM_ROUNDS)

# ── PLOT ──────────────────────────────────────────────────────────
def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode='valid')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(smooth(server.rmse_history), color='tab:blue', linewidth=2)
ax1.set_title("dyTACFL — RMSE Convergence")
ax1.set_xlabel("Communication Rounds")
ax1.set_ylabel("RMSE (kWh)")
ax1.grid(True)

ax2.plot(server.sil_history, color='tab:orange', alpha=0.8, linewidth=1.5)
ax2.set_title("Cluster Quality (Silhouette Score)")
ax2.set_xlabel("Communication Rounds")
ax2.set_ylabel("Silhouette Score")
ax2.grid(True)

plt.tight_layout()
plt.savefig("dyTACFL_results.png", dpi=150)
plt.show()

print(f"\nFinal RMSE : {server.rmse_history[-1]:.4f} kWh")
print(f"Best  RMSE : {min(server.rmse_history):.4f} kWh")
