import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import sys
from model import get_resnet
from dataset import PneumoniaDataset
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import atexit
from torchvision import transforms

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Load model
model = get_resnet().to(DEVICE)

# Dataset loading
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_path = sys.argv[1]
dataset = PneumoniaDataset(data_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Loss function with class balancing
labels = [label for _, label in dataset]
class_counts = torch.bincount(torch.tensor(labels))
weights = 1.0 / class_counts.float()
weights = weights / weights.sum()

loss_fn = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Privacy engine
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Tracking parameter norms
param_norms = []
client_id = os.path.basename(data_path.rstrip("/"))

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        total_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.parameters()]))
        param_norms.append(total_norm.item())
        print(f"[{client_id}] 🚀 Total Param Norm: {total_norm.item():.4f}")
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(model.state_dict().keys())
        model.load_state_dict(dict(zip(keys, [torch.tensor(p) for p in parameters])))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        acc = correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"[{client_id}] 🔐 DP epsilon after training: {epsilon:.2f}")

        return self.get_parameters(config), len(train_loader.dataset), {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                total_loss += loss.item() * x.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / total
        acc = correct / total
        precision = precision_score(all_preds, all_labels, zero_division=0)
        recall = recall_score(all_preds, all_labels, zero_division=0)
        f1 = f1_score(all_preds, all_labels, zero_division=0)

        return avg_loss, total, {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

def plot_weight_norms():
    os.makedirs("outputs", exist_ok=True)
    rounds = list(range(1, len(param_norms) + 1))
    plt.plot(rounds, param_norms, marker="o")
    plt.xlabel("Round")
    plt.ylabel("L2 Norm")
    plt.title(f"Param Norms - {client_id}")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"outputs/weight_norms_{client_id}.png")
    print(f"[{client_id}] 📉 Saved weight norms plot.")

atexit.register(plot_weight_norms)

# Start client
fl.client.start_client(server_address="federated-xray-server:8080", client=FLClient().to_client())
