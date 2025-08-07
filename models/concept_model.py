import torch
import torch.nn as nn
import torchvision.models as models

class ConceptModel(nn.Module):
    def __init__(self, num_concepts=312):
        super().__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)

        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features  # typically 512
        self.resnet.fc =  nn.Linear(in_features, num_concepts)

    def forward(self, x):
        return self.resnet(x)