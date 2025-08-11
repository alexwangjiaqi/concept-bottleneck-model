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

from cub_loader import load_data, find_class_imbalance, SELECTED_CONCEPTS, N_CLASSES
from models.concept_model import ConceptModel



# ==== Hyperparameters ====
num_epochs = 40              # Number of times the model sees the entire training dataset
batch_size = 64              # Number of samples processed before updating model weights
learning_rate = 1e-3         # Step size used by the optimizer to update weights
weight_decay = 1e-5          # L2 regularization strength to prevent overfitting (0 = no regularization)
num_concepts = 312           # Number of binary attributes/concepts to predict (CUB dataset has 312)
log_every = 10               # Print training info (e.g., loss) every N batches
save_path = 'checkpoints/concept_model.pth'  # File path to save the trained model
num_concepts = len(SELECTED_CONCEPTS)
imbalance_ratio = find_class_imbalance('data/CUB_processed/train.pkl', SELECTED_CONCEPTS)


# ==== Device Configuration ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==== Model Initialization ====
model = ConceptModel(num_concepts=num_concepts)
model = model.to(device)
print(model)


# ==== Data Loaders ====
train_loader = load_data(
    pkl_paths=['data/CUB_processed/train.pkl'],
    batch_size=64,
    use_attr=True,
    no_img=False,
    uncertain_label=False,
    reduced_attr = SELECTED_CONCEPTS
)
val_loader = load_data(
    pkl_paths=['data/CUB_processed/val.pkl'],
    batch_size=64,
    use_attr=True,
    no_img=False,
    uncertain_label=False,
    reduced_attr = SELECTED_CONCEPTS
)

# ==== Loss and Optimizer ====
weights_tensor = torch.tensor(imbalance_ratio, dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ==== Training Function without early stopping ====
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, concepts, _ in train_loader:  # We only care about concepts here (x, c)
            images = images.to(device)
            concepts = concepts.to(device)

            # Forward pass
            outputs = model(images)  # outputs are logits
            loss = criterion(outputs, concepts)  # BCEWithLogitsLoss expects raw logits and soft labels

             # Backpropagation and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Average loss over all batches
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

def evaluate_model_loss(model, val_loader, criterion, device):
    model.eval()  # Set to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, concepts, _ in val_loader: 
            images = images.to(device)
            concepts = concepts.to(device)

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, concepts)  # Compare logits to soft labels

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    average_loss = total_loss / total_samples
    return average_loss

def evaluate_model_auc(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode (important for layers like dropout or batchnorm)

    all_outputs = []  # To store model predictions for the whole validation set
    all_labels = []   # To store ground truth concept labels for the whole validation set

    with torch.no_grad():  # Disable gradient computation for memory and speed efficiency
        for images, concepts, _ in val_loader:  
            images = images.to(device)
            concepts = concepts.to(device)

            logits = model(images)              # Raw model outputs (logits)
            probs = torch.sigmoid(logits)       # Convert logits to probabilities using sigmoid

            all_outputs.append(probs.cpu())     # Move to CPU and store for AUC calculation
            all_labels.append(concepts.cpu())   # Move to CPU and store true labels

    # Concatenate all stored batches into full validation set tensors
    all_outputs = torch.cat(all_outputs, dim=0).numpy()  # Shape: [num_samples, num_concepts]
    all_labels = torch.cat(all_labels, dim=0).numpy() 



    aucs = []

    for concept_idx in range(all_labels.shape[1]):
        y_true = all_labels[:, concept_idx]
        y_score = all_outputs[:, concept_idx]

        # Only compute AUC if both 0s and 1s are present in the labels
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_score)
            aucs.append(auc)
        else:
            # Skip this concept (do not append anything)
            print(f"Skipping concept {concept_idx} due to only one class in validation set (unique: {np.unique(y_true)[0]})")

    # Compute macro AUC over valid concepts only
    if aucs:
        return np.mean(aucs)
    else:
        print("No valid concepts for AUC.")
        return float('nan')

    """num_concepts = all_labels.shape[1]
    for i in range(num_concepts):
        unique_vals = np.unique(all_labels[:, i])
        if len(unique_vals) == 1:
            print(f"concept {i} has only one unique value in labels: {unique_vals[0]}")

    # Compute AUC across all 312 concepts
    try:
        auc_score = roc_auc_score(all_labels, all_outputs, average='macro')
        # 'macro' = compute AUC independently for each concept, then average the results
    except ValueError:
        # AUC can fail if a concept has only one class in the ground truth (e.g., all 1s or all 0s)
        auc_score = float('nan')

    return auc_score"""

def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path, patience=3):
    best_auc = 0.0                      # Best AUC so far
    epochs_no_improve = 0              # Counter for early stopping
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, concepts, _ in train_loader:
            images = images.to(device)
            concepts = concepts.to(device)

            outputs = model(images)
            loss = criterion(outputs, concepts)

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


with torch.no_grad():
    for images, concepts, _ in val_loader:
        print("Batch concept shape:", concepts.shape)
        print("Min value:", concepts.min().item())
        print("Max value:", concepts.max().item())
        break  # just check the first batch


train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path)

'''train_model(model, train_loader, criterion, optimizer, device, num_epochs)
torch.save(model.state_dict(), save_path)'''