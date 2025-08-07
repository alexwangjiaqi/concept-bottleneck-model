import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

# Define output directory
output_dir = 'results/plots/independent'
os.makedirs(output_dir, exist_ok=True)

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Add project root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import load_data
from models.concept_model import ConceptModel
from models.label_model import LabelModel  # assuming label model is defined there

# ==== Load Test Data ====
test_loader = load_data(
    pkl_paths=['data/CUB_processed/test.pkl'],
    batch_size=64,
    use_attr=True, 
    no_img=False,
    uncertain_label=True
)

# ==== Load Models ====
#  Load Concept Predictor
concept_model = ConceptModel(num_concepts=312)
concept_model.load_state_dict(torch.load('checkpoints/concept_model.pth', map_location=device))
concept_model.to(device)
concept_model.eval()

# Load Label Predictor
label_model = LabelModel(num_concepts=312, num_classes=200)
label_model.load_state_dict(torch.load('checkpoints/label_model.pth', map_location=device))
label_model.to(device)
label_model.eval()

"""# ==== Run X → C → Y on Test Set ====
all_preds = []
all_labels = []

with torch.no_grad():
    for images, _, labels in test_loader:  # We're ignoring concepts, using only image and class labels
        images = images.to(device)
        labels = labels.to(device)

        # Step 1: Predict Concepts
        concept_logits = concept_model(images)
        concept_probs = torch.sigmoid(concept_logits)  # BCE model gives logits

        # Step 2: Predict Class from Concepts
        class_logits = label_model(concept_probs)
        predicted_classes = torch.argmax(class_logits, dim=1)

        all_preds.append(predicted_classes.cpu())
        all_labels.append(labels.cpu())

# ==== Evaluate ====
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy (X → C → Y): {accuracy:.4f}")
"""


### ==== Visualize Predictions ====
# This part is optional, but can help understand model predictions visually

import matplotlib.pyplot as plt
import random

# Load class names (you may need to adjust this path or format)
with open('data/CUB_200_2011/classes.txt') as f:
    class_names = [line.strip().split(' ', 1)[1].replace('_', ' ') for line in f]

# Load test dataset
test_dataset = test_loader.dataset

# Set random seed (optional, for reproducibility)
random.seed(42)

# Select 10 random indices
num_samples = 10
random_indices = random.sample(range(len(test_dataset)), num_samples)

# Create output directory
output_dir = 'results/plots/independent'
os.makedirs(output_dir, exist_ok=True)

# Loop through each randomly selected sample
for i, idx in enumerate(random_indices):
    image, concepts, label = test_dataset[idx]

    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    label = label.item()                   # Convert tensor to scalar if needed

    # Get concept predictions
    with torch.no_grad():
        concept_logits = concept_model(image)
        concept_probs = torch.sigmoid(concept_logits)

        # Predict class
        label_logits = label_model(concept_probs.to(device))
        label_probs = torch.softmax(label_logits, dim=1).cpu().squeeze()

    # Get top-5 predicted labels
    topk = torch.topk(label_probs, 5)
    top5_indices = topk.indices.tolist()
    top5_probs = topk.values.tolist()
    top5_str = "\n".join([f"{class_names[i]}: {p:.2f}" for i, p in zip(top5_indices, top5_probs)])

    # True label name
    true_class_name = class_names[label]
    predicted_class_name = class_names[top5_indices[0]]

    # Plot
    plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu())
    plt.title(f"True: {true_class_name}\nPredicted:\n{top5_str}")
    plt.axis('off')

    # Save plot
    filename = f"{i:02d}_true_{true_class_name}_pred_{predicted_class_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
            
    print(f"True label: {class_names[label]}")

    print("Top 5 predictions:")
    for j in range(5):
            print(f"  {class_names[top5_indices[j]]}: {top5_probs[j]:.4f}")

    print("-" * 50)