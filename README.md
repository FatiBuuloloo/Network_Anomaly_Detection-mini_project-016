# Network Traffic Anomaly Detection via Unsupervised Isolation Forest

An enterprise-grade network security solution leveraging an unsupervised Isolation Forest architecture to detect anomalous traffic patterns. The system is engineered for high-throughput network environments where labeled attack data is scarce or dynamically changing, focusing on learning structural boundaries strictly from baseline normal telemetry. Experiment tracking, hyperparameter lineage, and performance artifacts are fully instrumented using MLflow.

## Project Architecture & Core Decisions

* **Semi-Supervised / Unsupervised Paradigm:** The training pipeline enforces strict architectural hygiene. The Isolation Forest model is calibrated using purely normal network behaviors (85% of baseline normal traffic), ensuring the model learns intrinsic topology without target leakage.
* **Operational Separation of Concerns:** Core components are decoupled into clean modular scripts (`preprocessing_data.py`, `train.py`) to facilitate automation, enterprise scalability, and reproducible CI/CD execution.
* **MLOps Instrumentation:** Every training iteration programmatically registers hyperparameters, artifact lineage (such as serialized evaluation plots), and per-class performance matrices into a localized MLflow tracking server.

---

## Directory Structure

```text
.
├── .gitignore                          # Excludes heavy binaries, local DBs, and datasets
├── network-traffic-anomaly-detection.ipynb  # Exploratory Data Analysis and prototyping research
├── preprocessing_data.py               # Feature scaling, transformations, and encoding pipelines
├── train.py                            # Production execution script integrated with MLflow
└── README.md                           # Technical documentation and executive summary