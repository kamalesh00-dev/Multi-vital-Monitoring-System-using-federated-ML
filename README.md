
# Federated Multivital Monitoring System

A privacy-preserving machine learning framework designed to monitor and analyze patient vital signs (ECG, Heart Rate, Arrhythmia) using **Federated Learning**. This system allows local nodes (hospitals or wearable devices) to train models on sensitive medical data without ever sharing the raw data with a central server.

## Key Features
* **Decentralized Training:** Uses Federated Learning (FedAvg) to aggregate model weights from multiple clients.
* **Multivital Support:** Capable of processing ECG data and Arrhythmia patterns.
* **Real-time Dashboard:** Includes web-based plotting and performance tracing for monitoring global model performance and data distribution.
* **Data Privacy:** Ensures HIPAA/GDPR-compliant data handling by keeping data local.

---

## Project Structure
* 	rain_model.py / 	rain_finetune.py: Main entry points for local and fine-tuned training operations.
* ederated_train_combined.py: Core logic for federated aggregation and cross-domain model updates.
* 	est_global_model.py / 	est_federated_model.py: Scripts used to evaluate the aggregated global model performance.
* 	est_synthetic.py: Script to generate and test models against synthetic data distributions.
* ssets/: Dedicated directory containing performance tracking plots, including ROC curves and confusion matrices (e.g., assets/bilstm_roc.png, ssets/confusion_matrix.png).

---

## Installation & Setup

1. **Clone the Repository:**
   \\\bash
   git clone https://github.com/kamalesh00-dev/Multi-vital-Monitoring-System-using-federated-ML
   cd Multi-vital-Monitoring-System-using-federated-ML
   \\\

2. **Set Up the Virtual Environment:**
   \\\bash
   python -m venv .venv
   # On Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # On macOS/Linux
   source .venv/bin/activate
   \\\

3. **Install Dependencies:**
   \\\ash
   pip install -r requirements.txt
   \\\

4. **Dataset Setup:**
   * To ensure compliance with data privacy standards and keep the repository lightweight, **raw dataset files are excluded** from version control.
   * Download the MIT-BIH Arrhythmia Dataset (or your respective multivital source dataset).
   * Create a local directory named \datasets/\ in the project root folder.
   * Place your structured source files (e.g., \rrhythmia_clean.csv\, train/test splits) inside the \datasets/\ folder before running execution scripts.

---

## Usage

* **To train the federated model across simulated clients:**
   \\\bash
   python train_model.py
   \\\

* **To evaluate the aggregated global model's performance metrics:**
   \\\bash
   python test_global_model.py
   \\\

* **To run tests with generated synthetic data distributions:**
   \\\bash
   python test_synthetic.py
   \\\

