import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Dict, Tuple, List

from data_prep_uci_heart import make_clients, xy_from_df

# ======================================================
# CONFIG
# ======================================================
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# MODEL DEFINITION
# ======================================================
class SimpleNN(nn.Module):
    def __init__(self, in_dim=13, hidden=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# TRAIN / EVAL HELPERS
# ======================================================
def train_local(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs=5, lr=1e-3, batch=64) -> float:
    if X.shape[0] == 0:
        return 0.0
    model = model.to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

    total_loss = 0.0
    steps = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            steps += 1
    return total_loss / max(1, steps)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    if X.shape[0] == 0:
        return 0.0, np.zeros((2, 2), dtype=int)
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    return acc, cm


def get_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]):
    model.load_state_dict(state)


def average_state_dicts(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    avg = {}
    for k in states[0].keys():
        avg[k] = sum(s[k] for s in states) / float(len(states))
    return avg


# ======================================================
# MAIN FEDERATED TRAINING LOOP
# ======================================================
def main():
    clients_df, test_df = make_clients(test_frac=0.2, seed=42)
    if len(clients_df) == 0:
        raise RuntimeError("No client data found. Check dataset paths and files.")

    X_test, y_test = xy_from_df(test_df)
    global_model = SimpleNN(in_dim=13, hidden=32, out_dim=2).to(DEVICE)

    ROUNDS = 30
    LOCAL_EPOCHS = 5
    LR = 1e-3
    BATCH = 64

    history = {
        "rounds": [],
        "accuracy": [],
        "loss": [],
        "clients": {site: {"accuracy": []} for site in clients_df.keys()},
    }

    # prepare client metrics dataframe
    client_metrics_path = os.path.join(SAVE_DIR, "client_metrics.csv")
    metrics_records = []

    last_cm = np.zeros((2, 2), dtype=int)

    print("\n🏥 Federated Learning Begins...\n")

    for r in range(1, ROUNDS + 1):
        local_states = []
        local_losses = []

        print(f"------------------- Round {r} -------------------")

        # Track per-client performance this round
        for site, df in clients_df.items():
            X_tr, y_tr = xy_from_df(df)
            local_model = SimpleNN(in_dim=13, hidden=32, out_dim=2).to(DEVICE)
            set_state_dict(local_model, get_state_dict(global_model))

            loss = train_local(local_model, X_tr, y_tr, epochs=LOCAL_EPOCHS, lr=LR, batch=BATCH)
            local_losses.append(loss)

            acc_site, _ = evaluate(local_model, X_tr, y_tr)
            history["clients"][site]["accuracy"].append(float(acc_site))
            local_states.append(get_state_dict(local_model))

            # add to metrics log
            metrics_records.append({
                "Round": r,
                "Client": site,
                "TrainSamples": len(df),
                "LocalLoss": float(loss),
                "LocalAcc": float(acc_site),
            })

            print(f"🏥 {site.title():<12} | Samples: {len(df):<4} | Acc: {acc_site:.4f} | Loss: {loss:.4f}")

        # FedAvg aggregation
        global_state = average_state_dicts(local_states)
        set_state_dict(global_model, global_state)

        acc_global, cm = evaluate(global_model, X_test, y_test)
        last_cm = cm
        history["rounds"].append(r)
        history["accuracy"].append(float(acc_global))
        history["loss"].append(float(np.mean(local_losses)))

        print(f"🌍 Global Round {r:02d} | Acc: {acc_global:.4f} | Avg Local Loss: {np.mean(local_losses):.4f}")

    # Save artifacts
    torch.save(global_model.state_dict(), os.path.join(SAVE_DIR, "global_simpleNN.pt"))
    np.save(os.path.join(SAVE_DIR, "confusion_matrix.npy"), last_cm)
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # save client metrics CSV
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(client_metrics_path, index=False)

    print(f"\n📊 Client metrics saved to: {client_metrics_path}")
    print(f"🧠 Global model + history saved in: {SAVE_DIR}")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    main()
