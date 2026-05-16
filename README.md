# Federated Multivital Monitoring System

A privacy-preserving machine learning framework designed to monitor and analyze patient vital signs such as ECG, Heart Rate, and Arrhythmia using Federated Learning.

This system enables hospitals, wearable devices, and healthcare nodes to collaboratively train machine learning models without sharing sensitive patient data with a centralized server.

---

## Overview

Traditional healthcare AI systems require centralized storage of patient records, creating privacy and security concerns. This project solves that issue using Federated Learning (FedAvg), where local clients train models independently and only model parameters are shared with the central aggregator.

The framework supports:
- ECG signal analysis
- Heart rate monitoring
- Arrhythmia classification
- Distributed federated training
- Synthetic data testing
- Real-time dashboard visualization

---

# Key Features

- Decentralized Federated Learning using FedAvg
- Privacy-preserving healthcare AI
- ECG and Arrhythmia classification
- Multi-client distributed training
- Synthetic healthcare data testing
- Real-time monitoring dashboard
- Confusion matrix and ROC visualization
- Global model aggregation and evaluation

---

# Project Structure

```text
Multi-vital-Monitoring-System-using-federated-ML/
│
├── train_model.py
├── train_finetune.py
├── federated_train_combined.py
├── test_global_model.py
├── test_federated_model.py
├── test_synthetic.py
│
├── datasets/
│   └── health_status/
│
├── saved_models/
│   ├── global_simpleNN.pt
│   └── training_history.json
│
├── assets/
│   ├── confusion_matrix.png
│   ├── bilstm_roc.png
│   └── training_plots/
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python
- PyTorch
- Federated Learning (FedAvg)
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Installation & Setup

## Clone the Repository

```bash
git clone https://github.com/kamalesh00-dev/Multi-vital-Monitoring-System-using-federated-ML

cd Multi-vital-Monitoring-System-using-federated-ML
```

---

## Create Virtual Environment

### Windows (PowerShell)

```bash
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

To maintain repository size and ensure HIPAA/GDPR-compliant privacy handling, raw datasets are excluded from version control.

## Download Dataset

Recommended datasets:
- MIT-BIH Arrhythmia Dataset
- ECG datasets
- Multivital healthcare datasets

---

## Create Dataset Directory

```bash
mkdir datasets
```

Place dataset files inside:

```text
datasets/
```

Example:

```text
datasets/
├── arrhythmia_clean.csv
├── train.csv
├── test.csv
└── ecg_data.csv
```

---

# Usage

## Train Federated Model

```bash
python train_model.py
```

---

## Fine-Tune Existing Model

```bash
python train_finetune.py
```

---

## Evaluate Global Model

```bash
python test_global_model.py
```

---

## Test Federated Clients

```bash
python test_federated_model.py
```

---

## Run Synthetic Data Testing

```bash
python test_synthetic.py
```

---

# Dashboard Execution

Launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Dashboard Features:
- Accuracy tracking
- Loss monitoring
- Confusion matrix visualization
- ROC curve plotting
- Federated round updates
- Client-wise analytics

---

# Federated Learning Workflow

1. Each hospital/client trains locally on private healthcare data.
2. Local model weights are shared with the federated server.
3. The server performs FedAvg aggregation.
4. Updated global weights are redistributed to all clients.
5. Training continues for multiple federated rounds.

This ensures:
- Data privacy
- Reduced risk of data leakage
- Distributed healthcare AI training

---

# Model Outputs

Generated outputs:

```text
saved_models/
├── global_simpleNN.pt
└── training_history.json
```

Visualization outputs:

```text
assets/
├── confusion_matrix.png
├── bilstm_roc.png
└── training_plots/
```

---

# Results

- High federated classification accuracy achieved
- Multi-client distributed training successfully implemented
- Real-time monitoring dashboard integrated
- Privacy-preserving AI workflow validated

---

# Future Improvements

- Differential Privacy integration
- Blockchain-assisted federated learning
- Secure encrypted aggregation
- Real-time wearable device integration
- Edge AI deployment optimization
- Mobile healthcare deployment

---

# Applications

- Smart hospitals
- Remote patient monitoring
- Wearable healthcare systems
- ICU analytics
- AI-assisted arrhythmia detection
- Edge healthcare intelligence

---

# License

This project is intended for educational and research purposes.

---

# Author

## Kamalesh S

GitHub:
https://github.com/kamalesh00-dev

---
