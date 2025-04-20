import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class JetBotDataset(Dataset):
    def __init__(self, images, forward_signals, left_signals, transform=None):
        self.images = images
        self.forward_signals = forward_signals
        self.left_signals = left_signals
        self.transforms = transform

    
    def __len__(self):
        return len(self.images)

   
    def __getitem__(self, idx):
        image = Image.open(os.path.join('dataset', self.images[idx])).convert('RGB')
        image = self.transforms(image)
        forward_signal = self.forward_signals[idx]
        left_signal = self.left_signals[idx]
        label = torch.stack([forward_signal, left_signal], dim=0)
        return image, label
