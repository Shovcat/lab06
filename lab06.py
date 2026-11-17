import os
import time
import argparse
import yaml
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from thop import profile
import numpy as np


parser = argparse.ArgumentParser(description="Lab 6 PyTorch MLflow Pipeline")
parser.add_argument("--model", type=str, default=None, choices=["MLP", "CNN", "Attention"], help="Override model architecture")
args = parser.parse_args()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Override config if command-line argument is passed
if args.model:
    config["model"]["name"] = args.model

torch.manual_seed(config["experiment"]["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)



def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    full_train = torchvision.datasets.FashionMNIST(
        root=config["data"]["root_dir"], train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=config["data"]["root_dir"], train=False, download=True, transform=transform
    )

    val_size = config["data"]["val_split"]
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(config["experiment"]["seed"])
    )

    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader



class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )
    def forward(self, x): return self.classifier(self.features(x))


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Linear(49, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 16, 49)
        x = self.patch_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


def measure_latency(model, batch_size, num_runs=50):
    model.eval()
    dummy = torch.randn(batch_size, 1, 28, 28).to(device)
    with torch.no_grad():
        for _ in range(10): model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda": torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
    return float(np.median(latencies))


def main():
    mlflow.set_tracking_uri(config["logging"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment"]["name"])

    train_loader, val_loader, test_loader = get_dataloaders()

    model_name = config["model"]["name"]
    if model_name == "MLP": model = MLPModel()
    elif model_name == "CNN": model = CNNModel()
    elif model_name == "Attention": model = AttentionModel()
    else: raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops = 2 * macs

    best_val_acc = 0.0
    checkpoint_path = os.path.join(config["logging"]["checkpoint_dir"], f"best_{model_name.lower()}_model.pth")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_name": model_name,
            "learning_rate": config["model"]["learning_rate"],
            "batch_size": config["data"]["batch_size"],
            "total_params": info.total_params,
            "flops": flops
        })

        epochs = config["model"]["epochs"]
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images).argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = 100.0 * val_correct / val_total
            epoch_loss = running_loss / len(train_loader)
            
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(f"[{model_name}] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"   --> Saved new best checkpoint with Val Acc: {val_acc:.2f}%")

        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            mlflow.log_artifact(checkpoint_path, artifact_path=config["logging"]["artifact_path"])

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = 100.0 * test_correct / test_total
        lat_b1 = measure_latency(model, batch_size=1)
        lat_b128 = measure_latency(model, batch_size=128)

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("latency_b1_ms", lat_b1)
        mlflow.log_metric("latency_b128_ms", lat_b128)

        print(f"\n--> [{model_name}] Final Test Accuracy: {test_acc:.2f}%")
        print(f"--> Latency B=1: {lat_b1:.3f} ms | B=128: {lat_b128:.3f} ms")
        print(f"MLflow Run ID: {run.info.run_id}\n")


if __name__ == "__main__":
    main()