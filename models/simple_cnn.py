import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_outputs=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.relu  = nn.ReLU()

        self.fc1   = nn.Linear(32*56*56, 128)
        self.fc2   = nn.Linear(128, num_outputs)

        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)    # [batch, 2]