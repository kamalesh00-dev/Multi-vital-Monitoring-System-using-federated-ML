@'
# Federated Multivital Monitoring System

A privacy-preserving machine learning framework designed to monitor and analyze patient vital signs (ECG, Heart Rate, Arrhythmia) using **Federated Learning**. This system allows local nodes (hospitals or wearable devices) to train models on sensitive medical data without ever sharing the raw data with a central server.

## Key Features
* **Decentralized Training:** Uses Federated Learning (FedAvg) to aggregate model weights from multiple clients.
* **Multivital Support:** Capable of processing ECG data and Arrhythmia patterns.
* **Real-time Dashboard:** Includes web-based plotting and performance tracing for monitoring global model performance and data distribution.
* **Data Privacy:** Ensures HIPAA/GDPR-compliant data handling by keeping data local.

---

## Project Structure
* `train_model.py` / `train_finetune.py`: Main entry points for local and fine-tuned training operations.
* `federated_train_combined.py`: Core logic for federated aggregation and cross-domain model updates.
* `test_global_model.py` / `test_federated_model.py`: Scripts used to evaluate the aggregated global model performance.
* `test_synthetic.py`: Script to generate and test models against synthetic data distributions.
* `assets/`: Dedicated directory containing performance tracking plots, including ROC curves and confusion matrices (e.g., `assets/bilstm_roc.png`, `assets/confusion_matrix.png`).

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/kamalesh00-dev/Multi-vital-Monitoring-System-using-federated-ML](https://github.com/kamalesh00-dev/Multi-vital-Monitoring-System-using-federated-ML)
   cd Multi-vital-Monitoring-System-using-federated-ML

   
