import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataset import get_dataloader

# Configuration matching the PyTorch Transfer Learning Tutorial & repo structure
SPLITS_PATH = "splits.json"
DATA_DIR = "NWPU-RESISC45"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_baseline_frozen.pth")

NUM_CLASSES = 45
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1
SEED = 42


def set_seed(seed: int = 42) -> None:
    """Ensure deterministic training behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_frozen_resnet18(num_classes: int = 45) -> nn.Module:
    """Load pretrained ResNet-18, freeze all backbone layers, and replace the final FC layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    # Newly created layers default to requires_grad=True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train() -> None:
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training frozen ResNet-18 baseline on device: {device}")

    # Load clean train and validation splits from dataset.py
    train_loader = get_dataloader(
        SPLITS_PATH, DATA_DIR, split="train", batch_size=BATCH_SIZE
    )
    val_loader = get_dataloader(
        SPLITS_PATH, DATA_DIR, split="val", batch_size=BATCH_SIZE
    )

    model = build_frozen_resnet18(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    # Only optimize the unfrozen final classification head
    optimizer = optim.SGD(
        model.fc.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    total_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = running_corrects / total_train

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= total_val
        val_acc = val_corrects / total_val

        # Save best checkpoint based on validation accuracy
        saved_msg = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_model_wts,
                    "val_acc": best_val_acc,
                    "val_loss": val_loss,
                    "mode": "frozen",
                },
                CHECKPOINT_PATH,
            )
            saved_msg = " [Saved Best]"

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}{saved_msg}"
        )

    total_time = time.time() - total_start
    print(f"\nTraining complete in {total_time / 60:.2f} mins.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
