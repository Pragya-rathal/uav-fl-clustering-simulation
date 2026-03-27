# 🚁 UAV-Assisted Federated Learning with Cluster-Based Client Selection

## 📌 Description

A modular simulation framework for **federated learning in UAV-assisted IoT networks**, designed to evaluate how clustering and intelligent cluster-head selection impact system efficiency.

The project compares three approaches:

* **Standard FL**
  All devices communicate directly with the UAV

* **Clustered FL (Random Heads)**
  Devices are grouped into clusters with randomly selected heads

* **Clustered FL (Proposed Method)**
  Cluster heads are selected based on:

  * Computation power
  * Clustering coefficient (local connectivity)

## 🎯 Objective

To analyze whether simple clustering combined with **computation-aware and topology-aware head selection** improves:

* Model convergence
* Communication cost
* Training latency

## 🧪 Key Highlights

* Pure **NumPy-based implementation** (no heavy frameworks)
* **K-Means clustering** for device grouping
* **Logistic regression with mini-batch SGD**
* Realistic **communication model (path loss + latency)**
* Fully modular design for easy experimentation

## 📊 Results Summary

* Standard FL achieves **lower communication cost and latency**
* Clustered methods introduce **overhead due to relay communication**
* Smart cluster-head selection shows **better early convergence trends**
* Performance gains are more evident under:

  * Non-IID data
  * Higher device heterogeneity
  * Larger models

---
