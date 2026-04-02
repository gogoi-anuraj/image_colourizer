import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from src.models import ResNetUNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetUNet().to(DEVICE)
model.load_state_dict(torch.load("model_pth/resnet_unet.pth", map_location=DEVICE))
model.eval()


def predict(L):
    L_tensor = torch.tensor(L).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_ab = model(L_tensor)

    return pred_ab