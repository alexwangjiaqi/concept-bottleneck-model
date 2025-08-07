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

from dataset import load_data
from models.label_model import LabelModel


# ==== Hyperparameters ====
num_epochs = 100              # Number of times to iterate over training set
batch_size = 64              # Samples per gradient update
learning_rate = 1e-3         # Optimizer step size
weight_decay = 1e-5          # L2 regularization strength
num_concepts = 312           # Number of input concepts (CUB: 312)
num_classes = 200            # Number of output labels (CUB: 200 classes)
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
    batch_size=batch_size,
    use_attr=True,
    no_img=True,     # No image needed, just concept → label
    uncertain_label=True
)
val_loader = load_data(
    pkl_paths=['data/CUB_processed/val.pkl'],
    batch_size=batch_size,
    use_attr=True,
    no_img=True,
    uncertain_label=True
)

# ==== Loss and Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path, patience=3):
    best_auc = 0.0                      # Best AUC so far
    epochs_no_improve = 0              # Counter for early stopping
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for concepts, labels in train_loader:
            concepts = concepts.to(device)
            labels = labels.to(device)

            outputs = model(concepts)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)


        # Evaluate on validation set
        val_auc = evaluate_model_auc(model, val_loader, device)  # Use your AUC function here

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Check if validation AUC improved
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print("AUC improved, saving model")
        else:
            epochs_no_improve += 1
            print(f"No improvement in AUC for {epochs_no_improve} epochs")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            early_stop = True
            break

    print(f"Best validation AUC: {best_auc:.4f}")


def evaluate_model_auc(model, val_loader, device):
    model.eval()  # Set to evaluation mode

    all_logits = []  # Store raw model outputs (logits)
    all_labels = []  # Store ground truth class labels

    with torch.no_grad():
        for concepts, labels in val_loader:
            concepts = concepts.to(device)
            labels = labels.to(device)

            logits = model(concepts)  # Shape: [batch_size, num_classes]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Stack into full tensors
    all_logits = torch.cat(all_logits, dim=0)  # Shape: [N, C]
    all_labels = torch.cat(all_labels, dim=0)  # Shape: [N]

    probs = F.softmax(all_logits, dim=1).numpy()  # Convert logits to softmax probabilities
    labels = all_labels.numpy()                  # Ground truth labels

    num_classes = probs.shape[1]
    aucs = []

    for class_idx in range(num_classes):
        # Convert labels to one-vs-rest (binary): 1 for current class, 0 otherwise
        y_true = (labels == class_idx).astype(int)
        y_score = probs[:, class_idx]

        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_score)
            aucs.append(auc)
        else:
            print(f"Skipping class {class_idx}: only one class present in validation set (label={np.unique(y_true)[0]})")

    if aucs:
        return np.mean(aucs)
    else:
        print("No valid classes for AUC. Returning NaN.")
        return float('nan')

train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path)