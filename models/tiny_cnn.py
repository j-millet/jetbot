import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    def __init__(self, num_outputs=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 224→112

            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 112→56

            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 56→28

            nn.Conv2d(256, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),         # 28→14

            nn.AdaptiveAvgPool2d(1), # 28→1
        )

        self.classifier = nn.Conv2d(1024, num_outputs, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)