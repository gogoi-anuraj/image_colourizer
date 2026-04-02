import os
import cv2
import numpy as np
import random
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    def __init__(self, root_dirs, image_size=256):
        """
        Args:
            root_dirs (list): list of dataset directories
            image_size (int): resize images (default 256)
        """
        self.image_paths = []
        self.image_size = image_size

        # Collect images from all directories
        for root_dir in root_dirs:
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.image_paths.append(os.path.join(root, file))

        if len(self.image_paths) == 0:
            raise ValueError("No images found. Check dataset path.")

        # Shuffle dataset
        random.shuffle(self.image_paths)

        print(f"Total images loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Error loading image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size))

        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # Split channels
        L = lab[:, :, 0]
        ab = lab[:, :, 1:]

        # Normalize
        L = L / 255.0                # [0,1]
        ab = ab / 128.0 - 1          # [-1,1]

        # Reshape for PyTorch
        L = np.expand_dims(L, axis=0)      # (1, H, W)
        ab = ab.transpose(2, 0, 1)         # (2, H, W)

        return L.astype("float32"), ab.astype("float32")