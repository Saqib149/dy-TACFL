# dy-TACFL: Dynamic Temporal Adaptive Clustered Federated Learning for Heterogeneous Clients

> **Published in:** *Electronics* (MDPI), 2025  
> **DOI:** [10.3390/electronics14010152](https://doi.org/10.3390/electronics14010152)

---

## Abstract

Federated learning is a potential solution for training secure machine learning models on a decentralized network of clients, with an emphasis on privacy. However, the management of system/data heterogeneity and the handling of time-varying client interests still pose challenges to traditional federated learning (FL) approaches. Therefore, we propose a **dynamic temporal adaptive clustered federated learning (dy-TACFL)** to tackle the issue of client heterogeneity in time-varying environments.

By continuously analyzing and assigning appropriate clusters to the clients with similar behavior, the proposed federated clustering approach increases both prediction accuracy and clustering efficiency.

- A **silhouette coefficient-based threshold** is used in the Temporal Adaptive Clustering Federated Learning (**TACFL**) algorithm to evaluate cluster stability in each round of federated training.
- An **affinity propagation-based dynamic clustering** (**APD-CFL**) algorithm is proposed to adaptively organize clients into an appropriate number of clusters, taking into account the complex underlying pattern.

The experimental findings indicate that the proposed time-based adaptive clustered federated learning algorithms can significantly improve prediction accuracy compared to existing clustered federated learning algorithms.

---

## Repository Structure

```
dy-TACFL/
├── dyTACFL.py                                  # Main dy-TACFL implementation
├── ACFL.py                                     # ACFL baseline
├── ADACFL.py                                   # APD-CFL / ADACFL implementation
├── clustering_fl.py                            # Clustered FL utilities
├── EVCSBO.csv                                  # EV Charging Station Boulder dataset
├── clustered_charging_stations.csv             # Pre-clustered station assignments
├── clustered_charging_stations_with_dynamic_rounds.csv  # Dynamic round allocations
├── results/
│   ├── dyTACFL_results.png                     # RMSE convergence & silhouette plots
│   ├── AP_TACFL.png                            # APD-CFL architecture diagram
│   ├── STPA.png                                # TACFL silhouette threshold diagram
│   ├── APDCD.png                               # APD clustering diagram
│   ├── CLTFL1.png                              # CL-TFL comparison
│   └── PCOA.png                                # PCOA results
└── README.md
```

---

## Method Overview

### dy-TACFL Pipeline

1. **Data**: EV charging session records from 15 stations — each station is one FL client (non-IID)
2. **Feature engineering**: `St_Name`, `Postal_Code`, `week_number`, `year`, `TC_Time` → StandardScaler normalized
3. **LSTM model**: 2-layer LSTM (hidden=64) + FC regression head per cluster
4. **Per-round training loop**:
   - Each client trains locally on its assigned cluster model (FedAvg aggregation)
   - KMeans re-clustering on client model weights after each round
   - Silhouette coefficient monitors cluster stability
5. **Dynamic aspect**: Cluster assignments update every round based on model-weight similarity

### Key Algorithms

| Algorithm | Description |
|-----------|-------------|
| **TACFL** | Silhouette-threshold based cluster stability check |
| **APD-CFL** | Affinity Propagation dynamic cluster number selection |
| **dy-TACFL** | Combines both — dynamic, temporally-aware clustered FL |

---

## Dataset

**EV Charging Station Boulder (EVCSBO)**  
- 38 charging stations, ~105,577 sessions  
- Features: station ID, postal code, start time, end time, charging duration, energy consumption  
- Target: `Consumption` (kWh)

---

## Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Usage

```bash
python dyTACFL.py
```

Trains for 100 communication rounds with 15 clients (one per station), 3 clusters, and outputs:
- Per-round RMSE (kWh) and silhouette score to console
- `dyTACFL_results.png` — convergence plots

---

## Citation

If you use this code, please cite:

```bibtex
@article{dytacfl2025,
  title   = {dy-TACFL: Dynamic Temporal Adaptive Clustered Federated Learning for Heterogeneous Clients},
  journal = {Electronics},
  volume  = {14},
  number  = {1},
  pages   = {152},
  year    = {2025},
  doi     = {10.3390/electronics14010152}
}
```
