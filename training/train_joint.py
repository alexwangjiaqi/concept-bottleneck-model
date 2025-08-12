
print("hello")
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
from models.label_model import LabelModel
from models.concept_model import ConceptModel
from models.full_model import FullModel
from training.train_concept import evaluate_model_auc


# ==== Hyperparameters ====
num_epochs = 100              # Number of times to iterate over training set
batch_size = 64              # Samples per gradient update
learning_rate = 1e-3         # Optimizer step size
weight_decay = 1e-5          # L2 regularization strength
num_concepts = len(SELECTED_CONCEPTS)           # Number of input concepts (CUB: 312)
num_classes = N_CLASSES           # Number of output labels (CUB: 200 classes)
save_path = 'checkpoints/joint_model.pth'  # Model save path
imbalance_ratio = find_class_imbalance('data/CUB_processed/train.pkl', SELECTED_CONCEPTS)

print("hello")
# ==== Device Configuration ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==== Model Initialization ====
concept_model = ConceptModel(num_concepts=num_concepts) 
label_model   = LabelModel(num_concepts=num_concepts, num_classes=num_classes)
full_model = FullModel(concept_model, label_model)

# ==== Load Data ====
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

loss_c = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
loss_y = nn.CrossEntropyLoss()
optimizer = optim.Adam(full_model.parameters(), lr=learning_rate)

# ==== Training Loop ====
def train_joint_with_early_stopping(
    model,                      # FullModel (returns c_logits, y_logits)
    train_loader, val_loader,
    loss_c, loss_y,             # BCEWithLogitsLoss for concepts, CrossEntropyLoss for labels
    optimizer,
    device,
    num_epochs,
    save_path,
    patience=3,
    weight_task=0.5             # single weight w in [0,1]: task emphasis
):
    """
    Joint CBM training with a single weight used for:
      - Training loss:     total = w * y_loss + (1-w) * c_loss
      - Val early-stop metric: score = w * acc + (1-w) * auc

    Args:
        weight_task: float in [0,1], higher = prioritize end-to-end task.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.to(device)

    best_score = -float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        run_c, run_y, run_total = 0.0, 0.0, 0.0

        for images, c_targets, y in train_loader:
            images    = images.to(device)
            c_targets = c_targets.to(device).float()   # BCE targets (0/1)
            y         = y.to(device).long()            # CE targets (0..K-1)

            c_logits, y_logits = model(images)

            c_loss = loss_c(c_logits, c_targets)
            y_loss = loss_y(y_logits, y)

            total = weight_task * y_loss + (1.0 - weight_task) * c_loss

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            run_c    += c_loss.item()
            run_y    += y_loss.item()
            run_total+= total.item()

        # ===== Validation =====

        # Task accuracy
        val_acc = evaluate_model_acc(model, val_loader, device)

        # Concept AUC (reuse your concept eval)
        #    We call it on the concept submodel (X->C). It expects (images, concepts, _) batches.
        val_auc = evaluate_model_auc(model.concept_model, val_loader, device)

        # Single weighted validation score
        val_score = weight_task * val_acc + (1.0 - weight_task) * val_auc

        print(
            f"Epoch {epoch:03d} | "
            f"Train Total {run_total/len(train_loader):.4f}  "
            f"C {run_c/len(train_loader):.4f}  "
            f"Y {run_y/len(train_loader):.4f} | "
            f"Val Acc {val_acc:.4f}  Val AUC {val_auc:.4f}  "
            f"Weighted Score {val_score:.4f}"
        )

        # Early stopping on the single weighted score
        if val_score > best_score:
            best_score = val_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print("↑ weighted score improved — model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    print(f"Best weighted val score: {best_score:.4f}")
    return best_score

 
def evaluate_model_acc(model, val_loader, device):
    model.eval()
    correct, total_n = 0, 0
    with torch.no_grad():
        for images, _, y in val_loader:
            images = images.to(device)
            y      = y.to(device).long()
            _, y_logits = model(images)
            preds = torch.argmax(y_logits, dim=1)
            correct += (preds == y).sum().item()
            total_n += y.numel()
    return correct / max(total_n, 1)

train_joint_with_early_stopping(
    model=full_model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_c=loss_c,
    loss_y=loss_y,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    save_path=save_path,
    patience=3,
    weight_task=0.6   # e.g., emphasize task a bit more than concepts
)