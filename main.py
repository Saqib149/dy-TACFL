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
NUM_CLUSTERS  = 3
NUM_ROUNDS    = 100
LOCAL_EPOCHS  = 2
LEARNING_RATE = 0.001
BATCH_SIZE    = 64
SIL_THRESHOLD = 0.6
MU            = 0.01   # FedProx proximal coefficient

FEATURE_COLS = ['St_Name', 'Postal_Code', 'hour', 'dayofweek', 'week_number', 'TC_Time']
TARGET_COL   = 'Consumption'
INPUT_SIZE   = len(FEATURE_COLS)

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
    df['hour']        = df['St_Time'].dt.hour
    df['dayofweek']   = df['St_Time'].dt.dayofweek
    df['week_number'] = df['St_Time'].dt.isocalendar().week.astype(int)
    df['TC_Time']     = pd.to_timedelta(df['TC_Time']).dt.total_seconds()
    df['St_Name']     = df['St_Name'].astype('category').cat.codes
    df['Postal_Code'] = df['Postal_Code'].astype('category').cat.codes

    station_raw = {}
    all_X_train = []

    for sid in sorted(df['St_Name'].unique()):
        sdf = df[df['St_Name'] == sid].reset_index(drop=True)
        if len(sdf) < 50:
            continue
        X = sdf[FEATURE_COLS].values.astype(float)
        y = sdf[TARGET_COL].values.astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.1, random_state=42)
        all_X_train.append(X_tr)
        station_raw[sid] = (X_tr, X_te, y_tr, y_te)

    # Fit scaler on all training data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_X_train))

    station_data = {}
    for sid, (X_tr, X_te, y_tr, y_te) in station_raw.items():
        station_data[sid] = (scaler.transform(X_tr), scaler.transform(X_te),
                             y_tr, y_te)
    return station_data

# ── MODEL: simple MLP — no sequences, learns TC_Time→Consumption ──
class MLPRegressor(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ── CLIENT ────────────────────────────────────────────────────────
class Client:
    def __init__(self, train_ds, test_ds, station_id, y_std):
        self.train_ds   = train_ds
        self.test_ds    = test_ds
        self.station_id = station_id
        self.y_std      = y_std
        self.model      = MLPRegressor()

    def train(self, cluster_weights):
        self.model.load_state_dict(copy.deepcopy(cluster_weights))
        global_params = [p.detach().clone() for p in self.model.parameters()]
        self.model.train()

        opt     = torch.optim.Adam(self.model.parameters(),
                                   lr=LEARNING_RATE, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        loader  = DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(LOCAL_EPOCHS):
            for X_b, y_b in loader:
                opt.zero_grad()
                pred = self.model(X_b)
                loss = loss_fn(pred, y_b)
                # FedProx: keep local model close to cluster model
                prox = sum((p - g).pow(2).sum()
                           for p, g in zip(self.model.parameters(), global_params))
                loss = loss + (MU / 2) * prox
                loss.backward()
                opt.step()

        return copy.deepcopy(self.model.state_dict()), len(self.train_ds)

    def weight_vector(self):
        # Last-layer weights for clustering (65 dims)
        return torch.cat([p.detach().view(-1)
                          for p in self.model.net[-1].parameters()]).numpy()

# ── SERVER ────────────────────────────────────────────────────────
class Server:
    def __init__(self, clients, num_clusters=NUM_CLUSTERS):
        self.clients          = clients
        self.num_clusters     = num_clusters
        self.cluster_models   = [MLPRegressor() for _ in range(num_clusters)]
        self.assignments      = np.array([i % num_clusters
                                          for i in range(len(clients))])
        self.rmse_history     = []
        self.sil_history      = []
        self.recluster_rounds = []

    def _fedavg(self, updates, sizes):
        total = sum(sizes)
        agg   = {}
        for key in updates[0]:
            agg[key] = sum(w[key] * (n / total) for w, n in zip(updates, sizes))
        return agg

    def _check_and_recluster(self, r):
        """TACFL: re-cluster only when silhouette drops below threshold."""
        feats = np.stack([c.weight_vector() for c in self.clients])
        try:
            cur_sil = silhouette_score(feats, self.assignments)
        except Exception:
            cur_sil = 0.0

        if cur_sil < SIL_THRESHOLD:
            km = KMeans(n_clusters=self.num_clusters, n_init=3,
                        random_state=r).fit(feats)
            self.assignments = km.labels_
            sil = silhouette_score(feats, self.assignments)
            self.recluster_rounds.append(r + 1)
            print(f"         [Re-clustered at round {r+1}: sil {cur_sil:.3f} < {SIL_THRESHOLD}]")
        else:
            sil = cur_sil

        return sil

    def _evaluate_cluster(self):
        """Cluster model on each client's test data (initial baseline)."""
        loss_fn = nn.MSELoss(reduction='sum')
        total, n = 0.0, 0
        for i, client in enumerate(self.clients):
            model = self.cluster_models[self.assignments[i]]
            model.eval()
            loader = DataLoader(client.test_ds, batch_size=256)
            with torch.no_grad():
                for X_b, y_b in loader:
                    total += loss_fn(model(X_b), y_b).item()
                    n     += len(y_b)
        return np.sqrt(total / n)

    def _evaluate_local(self):
        """Each client's LOCAL model on its own test data — matches paper's metric."""
        loss_fn = nn.MSELoss(reduction='sum')
        total, n = 0.0, 0
        for client in self.clients:
            client.model.eval()
            loader = DataLoader(client.test_ds, batch_size=256)
            with torch.no_grad():
                for X_b, y_b in loader:
                    total += loss_fn(client.model(X_b), y_b).item()
                    n     += len(y_b)
        return np.sqrt(total / n)

    def train(self, rounds=NUM_ROUNDS):
        # Record initial RMSE (random model, before any training)
        rmse0 = self._evaluate_cluster()
        self.rmse_history.append(rmse0)
        self.sil_history.append(0.0)
        avg_std = np.mean([c.y_std for c in self.clients])
        print(f"Round   0 | RMSE (norm): {rmse0:.4f} | "
              f"RMSE (kWh): {rmse0 * avg_std:.4f}  [initial]")

        for r in range(rounds):
            updates = [[] for _ in range(self.num_clusters)]
            sizes   = [[] for _ in range(self.num_clusters)]

            for i, client in enumerate(self.clients):
                cid = self.assignments[i]
                w, n = client.train(self.cluster_models[cid].state_dict())
                updates[cid].append(w)
                sizes[cid].append(n)

            # FedAvg aggregation per cluster
            for c in range(self.num_clusters):
                if updates[c]:
                    self.cluster_models[c].load_state_dict(
                        self._fedavg(updates[c], sizes[c])
                    )

            # Evaluate CLUSTER model after aggregation (the federated model)
            rmse_cluster = self._evaluate_cluster()
            sil = self._check_and_recluster(r)
            self.rmse_history.append(rmse_cluster)
            self.sil_history.append(sil)

            print(f"Round {r+1:3d} | RMSE (norm): {rmse_cluster:.4f} | "
                  f"RMSE (kWh): {rmse_cluster * avg_std:.4f} | "
                  f"Silhouette: {sil:.4f}")

# ── MAIN ──────────────────────────────────────────────────────────
station_data = load_data("EVCSBO.csv")
print(f"Stations (clients): {len(station_data)}\n")

all_y_train = np.concatenate([y_tr for _, _, y_tr, _ in station_data.values()])
y_mean, y_std = all_y_train.mean(), all_y_train.std()
print(f"Target — mean: {y_mean:.3f} kWh  std: {y_std:.3f} kWh\n")

clients = []
for sid, (X_tr, X_te, y_tr, y_te) in station_data.items():
    y_tr_n = (y_tr - y_mean) / y_std
    y_te_n = (y_te - y_mean) / y_std

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                              torch.tensor(y_tr_n, dtype=torch.float32))
    test_ds  = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                              torch.tensor(y_te_n, dtype=torch.float32))
    clients.append(Client(train_ds, test_ds, station_id=sid, y_std=y_std))

server = Server(clients)
server.train(rounds=NUM_ROUNDS)

# ── PLOT ──────────────────────────────────────────────────────────
def smooth(data, w=5):
    arr = np.array(data)
    w   = min(w, len(arr))
    if w <= 1:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode='valid')

rounds_axis   = list(range(0, NUM_ROUNDS + 1))
smoothed_rmse = smooth(server.rmse_history)
smooth_rounds = list(range(0, len(smoothed_rmse)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(smooth_rounds, smoothed_rmse, color='tab:blue', linewidth=2)
for rr in server.recluster_rounds:
    ax1.axvline(x=rr, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_title("dyTACFL — RMSE Convergence")
ax1.set_xlabel("Communication Rounds")
ax1.set_ylabel("RMSE")
ax1.grid(True)

ax2.plot(rounds_axis, server.sil_history, color='tab:orange', linewidth=1.5)
ax2.axhline(y=SIL_THRESHOLD, color='red', linestyle='--',
            linewidth=1, label=f'Threshold ({SIL_THRESHOLD})')
ax2.set_title("TACFL — Silhouette Score")
ax2.set_xlabel("Communication Rounds")
ax2.set_ylabel("Silhouette Score")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig("dyTACFL_results.png", dpi=150)
plt.show()

print(f"\nInitial RMSE (round 0) : {server.rmse_history[0]:.4f}")
print(f"Final   RMSE (round {NUM_ROUNDS}) : {server.rmse_history[-1]:.4f}")
print(f"Best    RMSE           : {min(server.rmse_history):.4f}")
print(f"Re-clustered at rounds : {server.recluster_rounds}")
