import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AgeGenderModel(nn.Module):
    def __init__(self):
        super(AgeGenderModel, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.age_head = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(num_ftrs, 2)
        )
    
    def forward(self, x):
        features = self.base_model(x)
        age_out = self.age_head(features)
        gender_out = self.gender_head(features)
        return age_out, gender_out