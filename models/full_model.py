import torch
import torch.nn as nn

class FullModel(nn.Module):
    def __init__(self, concept_model, label_model):
        super().__init__()
        self.concept_model = concept_model     # X -> C (logits)
        self.label_model   = label_model       # C -> Y  (logits)

    def forward(self, x):
        c_logits = self.concept_model(x)               # [B, n_concepts] (logits)
        c_probs  = torch.sigmoid(c_logits)             # pass probs to label head
        y_logits = self.label_model(c_probs)           # [B, n_classes]
        return c_logits, y_logits