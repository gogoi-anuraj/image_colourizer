import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import ColorizationDataset
from models import Autoencoder


def main():

    data_dirs = [
        "data/coco_subset",
        "data/places365"
    ]

    save_path = "models/autoencoder.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    dataset = ColorizationDataset(data_dirs, image_size=256)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, _ = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Model
    model = Autoencoder().to(device)

    # Loss
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    EPOCHS = 20

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for L, ab in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            L = torch.tensor(L).to(device)
            ab = torch.tensor(ab).to(device)

            pred_ab = model(L)

            loss = criterion(pred_ab, ab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    main()