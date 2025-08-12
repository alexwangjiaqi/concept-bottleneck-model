import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models

from tqdm import tqdm
import os
import sys

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cub_loader import load_data, SELECTED_CONCEPTS, N_CLASSES
from models.label_model import LabelModel


# ==== Hyperparameters ====
num_epochs = 100              # Number of times to iterate over training set
batch_size = 64              # Samples per gradient update
learning_rate = 1e-3         # Optimizer step size
weight_decay = 1e-5          # L2 regularization strength
num_concepts = len(SELECTED_CONCEPTS)           # Number of input concepts (CUB: 312)
num_classes = N_CLASSES           # Number of output labels (CUB: 200 classes)
save_path = 'checkpoints/label_model.pth'  # Model save path


# ==== Device Configuration ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==== Model Initialization ====
model = LabelModel(num_concepts=num_concepts, num_classes=num_classes)
model = model.to(device)
print(model)

# ==== Load Data ====
train_loader = load_data(
    pkl_paths=['data/CUB_processed/train.pkl'],
    batch_size=64,
    use_attr=True,
    no_img=True,
    uncertain_label=False,
    reduced_attr = SELECTED_CONCEPTS
)
val_loader = load_data(
    pkl_paths=['data/CUB_processed/val.pkl'],
    batch_size=64,
    use_attr=True,
    no_img=True,
    uncertain_label=False,
    reduced_attr = SELECTED_CONCEPTS
)



# ==== Loss and Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path, patience=5):
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for concepts, labels in train_loader:
            concepts = concepts.to(device)
            labels = labels.to(device).long()

            outputs = model(concepts)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Evaluate on validation set (accuracy)
        val_acc = evaluate_model_acc(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Early stopping on accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print("Accuracy improved — saving model.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best validation accuracy: {best_acc:.4f}")



def evaluate_model_acc(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for concepts, labels in val_loader:
            concepts = concepts.to(device)
            labels = labels.to(device).long()

            logits = model(concepts)               # [B, num_classes]
            preds = torch.argmax(logits, dim=1)    # class indices
            correct += (preds == labels).sum().item()
            total   += labels.numel()

    return correct / max(total, 1)