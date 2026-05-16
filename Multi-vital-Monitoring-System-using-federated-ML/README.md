# Federated Multivital Monitoring System

A privacy-preserving machine learning framework designed to monitor and analyze patient vital signs (ECG, Heart Rate, Arrhythmia) using **Federated Learning**. This system allows local nodes (hospitals or wearable devices) to train models on sensitive medical data without ever sharing the raw data with a central server.

## Key Features
* **Decentralized Training:** Uses Federated Learning (FedAvg) to aggregate model weights from multiple clients.
* **Multivital Support:** Capable of processing ECG data and Arrhythmia patterns.
* **Real-time Dashboard:** Includes a web-based interface (`dashboard.py`) for monitoring global model performance and data distribution.
* **Data Privacy:** Ensures HIPAA/GDPR-compliant data handling by keeping data local.

---

## Project Structure
* `app.py`: Main entry point for the application.
* `federated_train.py`: Core logic for federated aggregation and global model updates.
* `global_model.pth`: The latest aggregated global model weights.
* `src/`: Contains core source code for model architectures (CNN, BiLSTM).
* `utils/`: Helper scripts for data preprocessing and plotting.
* `requirements.txt`: List of dependencies required to run the project.

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/kamalesh00-dev/Multi-vital-Monitoring-System-using-federated-ML](https://github.com/kamalesh00-dev/Multi-vital-Monitoring-System-using-federated-ML)
   cd Multi-vital-Monitoring-System-using-federated-ML