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

from cub_loader import load_data, find_class_imbalance, SELECTED_CONCEPTS
from models.concept_model import ConceptModel
from training.train_concept import train_model_with_early_stopping



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

# ==== Training Loop ====
with torch.no_grad():
    for images, concepts, _ in val_loader:
        print("Batch concept shape:", concepts.shape)
        print("Min value:", concepts.min().item())
        print("Max value:", concepts.max().item())
        break  # just check the first batch


train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path)
