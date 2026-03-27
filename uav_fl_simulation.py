"""
====================================================================
Federated Learning in UAV-Assisted IoT Networks — Simulation
====================================================================
Compares three FL strategies:
  1. Baseline 1 – Standard FL   (all devices → UAV directly)
  2. Baseline 2 – Clustered FL  (random cluster-head selection)
  3. Proposed   – Clustered FL  (smart head: computation + CC score)

Stack: NumPy + scikit-learn + Matplotlib  (no PyTorch required)

Model: logistic regression implemented from scratch with SGD so that
       federated averaging operates directly on raw weight arrays.

Metrics evaluated
  • Model accuracy   vs. communication rounds
  • Communication cost (cumulative MB transmitted)
  • Training latency  (seconds, simulated)

Run:
    pip install numpy matplotlib scikit-learn
    python uav_fl_simulation.py
====================================================================
"""

import random
import math
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / file-only rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1.  SYSTEM CONFIGURATION                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

class Config:
    """Central place for all hyper-parameters."""
    N_DEVICES       = 30
    AREA_SIZE       = 1000.0    # metres
    UAV_POSITION    = (500.0, 500.0)

    N_CLUSTERS      = 5
    CLUSTER_HEAD_W_COMPUTE = 0.6
    CLUSTER_HEAD_W_CC      = 0.4

    N_ROUNDS        = 25
    LOCAL_EPOCHS    = 5
    LOCAL_LR        = 0.05

    N_SAMPLES       = 3000
    N_FEATURES      = 20
    N_CLASSES       = 2

    PATH_LOSS_EXP   = 2.5
    BASE_TX_RATE    = 10.0      # Mbps
    MODEL_SIZE_MB   = 0.1       # MB per gradient upload
    CC_RADIUS       = 220.0     # metres for clustering-coefficient


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2.  IOT DEVICE & UAV                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

class IoTDevice:
    def __init__(self, device_id: int, cfg: Config):
        self.id = device_id
        self.cfg = cfg
        self.x = random.uniform(0, cfg.AREA_SIZE)
        self.y = random.uniform(0, cfg.AREA_SIZE)
        self.compute_power = random.uniform(0.5, 5.0)   # GFLOPS

        dx = self.x - cfg.UAV_POSITION[0]
        dy = self.y - cfg.UAV_POSITION[1]
        self.distance_to_uav = math.sqrt(dx**2 + dy**2) + 1e-6
        self.channel_quality = self._path_loss(self.distance_to_uav)

        self.cluster_id      = None
        self.is_cluster_head = False
        self.cc              = 0.0

        self.X_local  = None
        self.y_local  = None
        self.n_samples = 0

    def _path_loss(self, dist: float) -> float:
        loss = 1.0 / (1.0 + (dist / 100.0) ** self.cfg.PATH_LOSS_EXP)
        return float(np.clip(loss, 0.01, 1.0))

    def distance_to(self, other: "IoTDevice") -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def tx_latency(self, dest_dist: float) -> float:
        """Transmission latency [s] for one MODEL_SIZE_MB upload."""
        rate_mbps = self.cfg.BASE_TX_RATE * self._path_loss(dest_dist)
        return self.cfg.MODEL_SIZE_MB / rate_mbps

    def compute_latency(self) -> float:
        """Simulated local-training latency [s]."""
        flops = self.n_samples * self.cfg.LOCAL_EPOCHS * 1e6
        return flops / (self.compute_power * 1e9)


class UAV:
    def __init__(self, cfg: Config):
        self.x, self.y = cfg.UAV_POSITION

    def distance_to(self, device: IoTDevice) -> float:
        return math.sqrt((self.x - device.x)**2 + (self.y - device.y)**2)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3.  DATASET                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def build_dataset(cfg: Config):
    X, y = make_classification(
        n_samples=cfg.N_SAMPLES, n_features=cfg.N_FEATURES,
        n_informative=10, n_redundant=5, random_state=SEED
    )
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def assign_data_to_devices(devices, X_train, y_train):
    n    = len(X_train)
    idx  = np.random.permutation(n)
    size = n // len(devices)
    for i, dev in enumerate(devices):
        s = i * size
        e = s + size if i < len(devices) - 1 else n
        dev.X_local   = X_train[idx[s:e]]
        dev.y_local   = y_train[idx[s:e]]
        dev.n_samples = len(dev.X_local)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4.  LOGISTIC-REGRESSION MODEL (pure NumPy, mini-batch SGD)    ║
# ╚══════════════════════════════════════════════════════════════════╝

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticModel:
    """Binary logistic regression with L2 regularisation."""

    def __init__(self, n_features: int, lam: float = 1e-4):
        self.W   = np.random.randn(n_features) * 0.01
        self.b   = 0.0
        self.lam = lam

    # ── Parameter I/O (mimic torch state-dict interface) ──────────
    def get_params(self) -> dict:
        return {"W": self.W.copy(), "b": float(self.b)}

    def set_params(self, params: dict):
        self.W = params["W"].copy()
        self.b = float(params["b"])

    # ── Inference ─────────────────────────────────────────────────
    def predict_proba(self, X):
        return _sigmoid(X @ self.W + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X, y) -> float:
        return float(np.mean(self.predict(X) == y))

    # ── Local training (mini-batch SGD) ───────────────────────────
    def train(self, X, y, lr: float, epochs: int, batch_size: int = 32):
        n = len(X)
        for _ in range(epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx   = perm[start:start + batch_size]
                Xb    = X[idx]; yb = y[idx]
                p     = self.predict_proba(Xb)
                err   = p - yb
                gW    = (Xb.T @ err) / len(yb) + self.lam * self.W
                gb    = float(np.mean(err))
                self.W -= lr * gW
                self.b -= lr * gb


def make_model(cfg: Config) -> LogisticModel:
    return LogisticModel(n_features=cfg.N_FEATURES)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5.  FEDAVG                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def fedavg(param_list: list, weights: list) -> dict:
    total = float(sum(weights))
    agg   = {}
    for key in param_list[0]:
        agg[key] = sum(
            (w / total) * p[key]
            for w, p in zip(weights, param_list)
        )
    return agg


# ╔══════════════════════════════════════════════════════════════════╗
# ║  6.  CLUSTERING                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def cluster_devices(devices, cfg: Config) -> dict:
    pos    = np.array([[d.x, d.y] for d in devices])
    km     = KMeans(n_clusters=cfg.N_CLUSTERS, random_state=SEED, n_init=10)
    labels = km.fit_predict(pos)
    for dev, lbl in zip(devices, labels):
        dev.cluster_id = int(lbl)
    clusters = defaultdict(list)
    for dev in devices:
        clusters[dev.cluster_id].append(dev)
    return dict(clusters)


def _cc_one_device(device, members, radius):
    nbrs = [d for d in members
            if d.id != device.id and device.distance_to(d) <= radius]
    k = len(nbrs)
    if k < 2:
        return 0.0
    edges = sum(
        1 for i in range(k) for j in range(i + 1, k)
        if nbrs[i].distance_to(nbrs[j]) <= radius
    )
    return edges / (k * (k - 1) / 2)


def assign_clustering_coefficients(clusters: dict, cfg: Config):
    for members in clusters.values():
        for dev in members:
            dev.cc = _cc_one_device(dev, members, cfg.CC_RADIUS)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  7.  CLUSTER-HEAD SELECTION                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def reset_heads(devices):
    for d in devices:
        d.is_cluster_head = False


def select_heads_random(clusters: dict) -> dict:
    heads = {}
    for cid, members in clusters.items():
        h = random.choice(members)
        h.is_cluster_head = True
        heads[cid] = h
    return heads


def select_heads_proposed(clusters: dict, cfg: Config) -> dict:
    """Score = w_compute * norm(compute) + w_cc * norm(CC)."""
    heads = {}
    for cid, members in clusters.items():
        cp  = np.array([d.compute_power for d in members])
        ccs = np.array([d.cc            for d in members])
        nc  = (cp  - cp.min())  / (np.ptp(cp)  + 1e-9)
        ncc = (ccs - ccs.min()) / (np.ptp(ccs) + 1e-9)
        scores = cfg.CLUSTER_HEAD_W_COMPUTE * nc + cfg.CLUSTER_HEAD_W_CC * ncc
        best   = members[int(np.argmax(scores))]
        best.is_cluster_head = True
        heads[cid] = best
    return heads


# ╔══════════════════════════════════════════════════════════════════╗
# ║  8.  LOCAL TRAINING HELPER                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def local_train(global_params: dict, dev: IoTDevice, cfg: Config):
    model = make_model(cfg)
    model.set_params(global_params)
    model.train(dev.X_local, dev.y_local,
                lr=cfg.LOCAL_LR, epochs=cfg.LOCAL_EPOCHS)
    return model.get_params(), dev.compute_latency()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  9.  FL STRATEGIES                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_standard_fl(devices, uav, global_params, cfg, X_test, y_test):
    """Baseline 1 — every device uploads directly to UAV."""
    print("\n[Baseline 1] Standard FL …")
    history = {"accuracy": [], "comm_cost_mb": [], "latency_s": []}
    params  = copy.deepcopy(global_params)
    total_comm = total_latency = 0.0
    evaluator  = make_model(cfg)

    for rnd in range(cfg.N_ROUNDS):
        lparams, lns, rlats = [], [], []
        for dev in devices:
            lp, clat = local_train(params, dev, cfg)
            tx        = dev.tx_latency(uav.distance_to(dev))
            lparams.append(lp); lns.append(dev.n_samples)
            rlats.append(clat + tx)
            total_comm += cfg.MODEL_SIZE_MB

        params         = fedavg(lparams, lns)
        total_latency += max(rlats)

        evaluator.set_params(params)
        acc = evaluator.accuracy(X_test, y_test)
        history["accuracy"].append(acc)
        history["comm_cost_mb"].append(total_comm)
        history["latency_s"].append(total_latency)
        print(f"  Round {rnd+1:2d}/{cfg.N_ROUNDS}  acc={acc:.4f}  "
              f"comm={total_comm:.1f} MB  latency={total_latency:.2f} s")

    return history


def run_clustered_fl(devices, clusters, uav, global_params, cfg,
                     X_test, y_test, head_selection="random"):
    """Clustered FL (Baseline 2 or Proposed)."""
    label = ("Baseline 2 (random head)"
             if head_selection == "random"
             else "Proposed  (smart head)")
    print(f"\n[{label}] Clustered FL …")

    reset_heads(devices)
    heads = (select_heads_random(clusters)
             if head_selection == "random"
             else select_heads_proposed(clusters, cfg))

    history = {"accuracy": [], "comm_cost_mb": [], "latency_s": []}
    params  = copy.deepcopy(global_params)
    total_comm = total_latency = 0.0
    evaluator  = make_model(cfg)

    for rnd in range(cfg.N_ROUNDS):
        cparams, cns, rlats = [], [], []

        for cid, members in clusters.items():
            head = heads[cid]
            mparams, mns, mlats = [], [], []

            for dev in members:
                lp, clat = local_train(params, dev, cfg)
                d2h = dev.distance_to(head) if dev.id != head.id else 1.0
                tx  = dev.tx_latency(d2h)
                mparams.append(lp); mns.append(dev.n_samples)
                mlats.append(clat + tx)
                total_comm += cfg.MODEL_SIZE_MB     # device → head

            head_p     = fedavg(mparams, mns)
            clat_total = max(mlats)
            clat_total += head.tx_latency(uav.distance_to(head))
            total_comm += cfg.MODEL_SIZE_MB         # head → UAV

            cparams.append(head_p); cns.append(sum(mns))
            rlats.append(clat_total)

        params         = fedavg(cparams, cns)
        total_latency += max(rlats)

        evaluator.set_params(params)
        acc = evaluator.accuracy(X_test, y_test)
        history["accuracy"].append(acc)
        history["comm_cost_mb"].append(total_comm)
        history["latency_s"].append(total_latency)
        print(f"  Round {rnd+1:2d}/{cfg.N_ROUNDS}  acc={acc:.4f}  "
              f"comm={total_comm:.1f} MB  latency={total_latency:.2f} s")

    return history, heads


# ╔══════════════════════════════════════════════════════════════════╗
# ║  10. VISUALISATION                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

PALETTE = {"baseline1": "#E84855", "baseline2": "#F4A261", "proposed": "#2EC4B6"}
LABELS  = {
    "baseline1": "Baseline 1: Standard FL",
    "baseline2": "Baseline 2: Clustered (random head)",
    "proposed":  "Proposed: Clustered (smart head)",
}


def _ann(ax, rounds, series, color):
    ax.annotate(f"{series[-1]:.3f}",
                xy=(rounds[-1], series[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=7, color=color, va="center")


def plot_topology(devices, clusters, heads, uav, cfg, ax):
    cmap = plt.cm.get_cmap("tab10", cfg.N_CLUSTERS)
    for cid, members in clusters.items():
        h = heads[cid]
        for d in members:
            ax.plot([d.x, h.x], [d.y, h.y],
                    color=cmap(cid), lw=0.4, alpha=0.35)
        xs = [d.x for d in members]; ys = [d.y for d in members]
        ax.scatter(xs, ys, color=cmap(cid), s=35, alpha=0.75,
                   label=f"Cluster {cid}")
        ax.scatter(h.x, h.y, color=cmap(cid), s=200,
                   marker="*", edgecolors="k", linewidths=0.6, zorder=5)
    for h in heads.values():
        ax.plot([h.x, uav.x], [h.y, uav.y], "k--", lw=0.55, alpha=0.35)
    ax.scatter(uav.x, uav.y, marker="^", s=250,
               color="#1a1a2e", zorder=6, label="UAV")
    ax.set_xlim(0, cfg.AREA_SIZE); ax.set_ylim(0, cfg.AREA_SIZE)
    ax.set_title("Network Topology\n(★ = cluster head  ▲ = UAV)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=9); ax.set_ylabel("Y (m)", fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="upper left")


def plot_results(h1, h2, hp, devices, clusters, heads, uav, cfg, out_path):
    rounds = list(range(1, cfg.N_ROUNDS + 1))
    markers = {"baseline1": "o", "baseline2": "s", "proposed": "D"}

    fig = plt.figure(figsize=(16, 10), facecolor="#f5f6fa")
    fig.suptitle(
        "Federated Learning in UAV-Assisted IoT Networks\n"
        "Performance Comparison: Standard FL vs Clustered FL",
        fontsize=14, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.set_facecolor("#eef2f7")
    plot_topology(devices, clusters, heads, uav, cfg, ax0)

    datasets = [("baseline1", h1), ("baseline2", h2), ("proposed", hp)]

    for ax_idx, (metric, title, ylabel, ylim) in enumerate([
        ("accuracy",     "Model Accuracy vs Rounds",           "Test Accuracy",          (0.5, 1.02)),
        ("comm_cost_mb", "Cumulative Communication Cost",       "Total Data (MB)",        None),
        ("latency_s",    "Cumulative Training Latency",         "Total Latency (s)",      None),
    ]):
        row = ax_idx // 2 + (0 if ax_idx == 0 else 1)
        col = ax_idx % 2  + (1 if ax_idx == 0 else 0)
        # manual grid positions
        positions = [(0, 1), (1, 0), (1, 1)]
        r, c = positions[ax_idx]
        ax = fig.add_subplot(gs[r, c]); ax.set_facecolor("#eef2f7")
        for key, hist in datasets:
            ax.plot(rounds, hist[metric],
                    color=PALETTE[key], lw=2.0,
                    marker=markers[key], ms=4, label=LABELS[key])
            _ann(ax, rounds, hist[metric], PALETTE[key])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Round", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim: ax.set_ylim(*ylim)
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.45)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")


def print_summary(h1, h2, hp):
    print("\n" + "=" * 65)
    print(f"{'Metric':<32} {'B1-StdFL':>9} {'B2-RandH':>9} {'Proposed':>9}")
    print("-" * 65)
    rows = [
        ("Final accuracy",       "accuracy",     ".4f"),
        ("Total comm cost (MB)", "comm_cost_mb", ".2f"),
        ("Total latency (s)",    "latency_s",    ".2f"),
    ]
    for label, key, fmt in rows:
        v1 = h1[key][-1]; v2 = h2[key][-1]; vp = hp[key][-1]
        print(f"  {label:<30} {format(v1, fmt):>9} {format(v2, fmt):>9} {format(vp, fmt):>9}")
    print("=" * 65)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  11. MAIN                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    cfg = Config()
    print("=" * 62)
    print("  UAV-Assisted IoT Federated Learning Simulation")
    print("=" * 62)
    print(f"  Devices={cfg.N_DEVICES}  Clusters={cfg.N_CLUSTERS}  "
          f"Rounds={cfg.N_ROUNDS}  LocalEpochs={cfg.LOCAL_EPOCHS}")
    print("=" * 62)

    uav     = UAV(cfg)
    devices = [IoTDevice(i, cfg) for i in range(cfg.N_DEVICES)]

    X_train, X_test, y_train, y_test = build_dataset(cfg)
    assign_data_to_devices(devices, X_train, y_train)

    clusters = cluster_devices(devices, cfg)
    assign_clustering_coefficients(clusters, cfg)

    global_params = make_model(cfg).get_params()

    h1      = run_standard_fl(devices, uav, global_params, cfg, X_test, y_test)
    h2, _   = run_clustered_fl(devices, clusters, uav, global_params, cfg,
                                X_test, y_test, head_selection="random")
    hp, heads = run_clustered_fl(devices, clusters, uav, global_params, cfg,
                                  X_test, y_test, head_selection="proposed")

    print_summary(h1, h2, hp)
    out = "/mnt/user-data/outputs/uav_fl_results.png"
    plot_results(h1, h2, hp, devices, clusters, heads, uav, cfg, out)
    print("\n  Done.")


if __name__ == "__main__":
    main()
