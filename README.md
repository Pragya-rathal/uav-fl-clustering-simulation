Overview

This project simulates a UAV-assisted federated learning (FL) system with IoT devices and evaluates the impact of clustering and cluster-head selection strategies on performance.

The goal is to test whether simple clustering with intelligent cluster-head selection improves training efficiency compared to standard FL.

Methods Compared
Standard FL
All devices communicate directly with the UAV
Clustered FL (Random Heads)
Devices form clusters
Cluster heads selected randomly
Clustered FL (Proposed Method)
Cluster heads selected based on:
Computation power
Clustering coefficient (local connectivity)
Features
Pure NumPy implementation (no PyTorch dependency)
K-Means clustering (scikit-learn)
Logistic regression with mini-batch SGD
Free-space path loss communication model
Modular design for easy extension
Metrics Evaluated
Model accuracy vs communication rounds
Total communication cost
Training latency
Project Structure
.
├── main.py              # Main simulation script
├── outputs/
│   ├── results.png     # Generated plots
│   └── simulation.py   # Copy of final script
└── README.md
How to Run (Local)

Install dependencies:

pip install numpy matplotlib scikit-learn

Run:

python main.py

Results will be saved in the outputs/ folder.

Running on Google Colab
Upload the script (main.py)
Install dependencies:
!pip install numpy matplotlib scikit-learn
Run:
!python main.py
Running on Kaggle
Add the script as a dataset or upload manually
Use a Python notebook
Run:
!pip install numpy matplotlib scikit-learn
!python main.py
Key Findings
Standard FL achieves lower communication cost and latency
Clustered approaches introduce overhead due to intra-cluster communication
Smart cluster-head selection shows improved early convergence behavior
Benefits are expected to increase under:
Non-IID data
Larger models
Higher device heterogeneity
Future Work
Incorporate model compression
Test non-IID data distributions
Extend to multi-UAV systems
Explore adaptive clustering strategies
