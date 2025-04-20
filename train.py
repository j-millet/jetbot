import argparse
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import get_data, get_transforms, save_onnx_model, load_model_from_path
from dataset import JetBotDataset


def main(args):
    data = get_data()
    train_t, val_t = get_transforms()

    paths = data['image_path'].to_numpy()
    fw = torch.tensor(data['forward_signal'].to_numpy(), dtype=torch.float32)
    lt = torch.tensor(data['left_signal'].to_numpy(),   dtype=torch.float32)

    tr_p, va_p, tr_fw, va_fw, tr_lt, va_lt = train_test_split(
        paths, fw, lt,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True
    )

    train_ds = JetBotDataset(tr_p, tr_fw, tr_lt, transform=train_t)
    val_ds = JetBotDataset(va_p, va_fw, va_lt, transform=val_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model = load_model_from_path(args.model_file, num_outputs=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_pbar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{args.epochs} — "
            f"Train MSE: {avg_train_loss:.4f}, "
            f"Val MSE:   {avg_val_loss:.4f}")

        save_onnx_model(model, os.path.splitext(os.path.basename(args.model_file))[0] + ".onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a JetBot steering model."
    )
    parser.add_argument(
        "model_file",
        help="Path to your model definition (e.g. models/simple_cnn.py)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="batch size for training/validation"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="number of epochs to train"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="learning rate"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="validation split fraction"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed"
    )
    args = parser.parse_args()
    main(args)