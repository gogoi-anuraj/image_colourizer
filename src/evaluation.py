import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import numpy as np
import cv2

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import ColorizationDataset
from models import Autoencoder, UNet, ResNetUNet


#  LAB → RGB

def lab_to_rgb_numpy(L, ab):
    """
    L: (1, H, W)
    ab: (2, H, W)
    """
    L_img = (L[0] * 255).astype("uint8")

    ab_img = (ab + 1) * 128
    ab_img = ab_img.transpose(1, 2, 0).astype("uint8")

    h, w = L_img.shape

    lab = np.zeros((h, w, 3), dtype="uint8")
    lab[:, :, 0] = L_img
    lab[:, :, 1:] = ab_img

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb


# Load Model

def load_model(model_name, model_path, device):

    if model_name == "autoencoder":
        model = Autoencoder()
    elif model_name == "unet":
        model = UNet()
    elif model_name == "resnet":
        model = ResNetUNet()
    else:
        raise ValueError("Invalid model name")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


# Evaluation

def evaluate(model, test_dataset, device):

    psnr_scores = []
    ssim_scores = []

    for L, ab in tqdm(test_dataset, desc="Evaluating"):

        L_tensor = torch.tensor(L).unsqueeze(0).to(device)
        ab_tensor = torch.tensor(ab).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_ab = model(L_tensor)

            pred_ab = F.interpolate(
                pred_ab,
                size=ab_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        pred_ab = pred_ab.squeeze(0).cpu().numpy()

        # Convert to RGB
        rgb_pred = lab_to_rgb_numpy(L, pred_ab)
        rgb_gt = lab_to_rgb_numpy(L, ab)

        # Compute metrics
        psnr = peak_signal_noise_ratio(rgb_gt, rgb_pred)
        ssim = structural_similarity(rgb_gt, rgb_pred, channel_axis=2)

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)

    return avg_psnr, avg_ssim


# MAIN

def main():

    # CONFIG
    model_name = "resnet"  # autoencoder / unet / resnet
    model_path = "models/resnet_unet_perc.pth"

    data_dirs = [
        "data/coco_subset",
        "data/places365"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = ColorizationDataset(data_dirs, image_size=256)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    _, test_dataset = random_split(dataset, [train_size, test_size])

    # -------- Model --------
    model = load_model(model_name, model_path, device)

    # -------- Evaluate --------
    psnr, ssim = evaluate(model, test_dataset, device)

    print("\n===== Evaluation Results =====")
    print(f"Model: {model_name}")
    print(f"PSNR: {psnr:.4f}")
    print(f"SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()