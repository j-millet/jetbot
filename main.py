from utils import *
from dataset import JetBotDataset
from models import SimpleCNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


data = get_data()
train_transforms, val_transforms = get_transorms()

paths = data['image_path'].to_numpy()
fw = torch.tensor(
    data['forward_signal'].to_numpy(), 
    dtype=torch.float32
)
lt = torch.tensor(
    data['left_signal'].to_numpy(), 
    dtype=torch.float32
)

train_paths, val_paths, train_fw, val_fw, train_lt, val_lt = train_test_split(
    paths, fw, lt,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_ds = JetBotDataset(
    train_paths,
    train_fw,
    train_lt,
    transform=train_transforms
)
val_ds = JetBotDataset(
    val_paths,
    val_fw,
    val_lt,
    transform=val_transforms
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_outputs=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)   # labels: [B,2]

        optimizer.zero_grad()
        outputs = model(images)        # [B,2]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} — "
          f"Train MSE: {avg_train_loss:.4f}, "
          f"Val MSE:   {avg_val_loss:.4f}")

save_onnx_model(model, "simple_cnn")