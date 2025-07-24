import torch
import torch.nn as nn
from torchvision import models

# Define number of classes in your dataset
num_classes = 11  # e.g., ['checked', 'floral', 'striped', 'zigzag', 'plain']

# Build the model
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes)
)

# Initialize weights (random)
for param in model.parameters():
    if param.requires_grad:
        if param.data.ndimension() >= 2:
            nn.init.kaiming_uniform_(param)
        else:
            nn.init.zeros_(param)

# Save the model
torch.save(model.state_dict(), "fabric_pattern_model.pt")
print("✅ Model saved as fabric_pattern_model.pt")
