import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import ColorizationDataset
from models import ResNetUNet

import torchvision.models as models
import kornia.color as kc


#  Perceptual Loss (VGG)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(vgg.children())[:16]).eval()

        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return nn.functional.l1_loss(self.vgg(pred), self.vgg(target))


# LAB -> RGB

def lab_to_rgb(L, ab):
    lab = torch.cat([L, ab], dim=1)
    return kc.lab_to_rgb(lab)


# Training Setup

def main():

    data_dirs = [
        "data/coco_subset",
        "data/places365"
    ]

    save_path = "models/resnet_unet.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- Dataset --------
    dataset = ColorizationDataset(data_dirs, image_size=256)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Model 
    model = ResNetUNet().to(device)

    # Loss
    l1_loss = nn.L1Loss()
    perc_loss = VGGPerceptualLoss().to(device)

    lambda_perc = 0.1

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # TRAINING LOOP

    EPOCHS = 20

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for L, ab in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            L = torch.tensor(L).to(device)
            ab = torch.tensor(ab).to(device)

            # Forward
            pred_ab = model(L)

            pred_ab = F.interpolate(pred_ab, size=ab.shape[2:], mode='bilinear', align_corners=False)

            # L1 loss
            loss_l1 = l1_loss(pred_ab, ab)

            # Convert to RGB
            pred_rgb = lab_to_rgb(L, pred_ab)
            gt_rgb = lab_to_rgb(L, ab)

            # Perceptual loss
            loss_perc = perc_loss(pred_rgb, gt_rgb)

            # Total loss
            loss = loss_l1 + lambda_perc * loss_perc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}  Loss: {train_loss/len(train_loader):.4f}")

    # -------- Save Model --------
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


#  RUN
if __name__ == "__main__":
    main()
