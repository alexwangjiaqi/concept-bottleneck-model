import torch
import torch.nn as nn

class LabelModel(nn.Module):
    def __init__(self, num_concepts=312, num_classes=200):
        super().__init__()

        # Simple linear classifier from concepts to labels
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, concepts):
        """
        Forward pass.

        Args:
            concepts (Tensor): shape (batch_size, num_concepts), typically soft concept predictions

        Returns:
            logits (Tensor): shape (batch_size, num_classes)
        """
        return self.classifier(concepts)