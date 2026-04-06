import cv2
import numpy as np
import torch.nn.functional as F


def preprocess_image(image, size=256):
    h, w, _ = image.shape

    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    image_resized = cv2.resize(image, (new_w, new_h))

    padded = np.zeros((size, size, 3), dtype=np.uint8)

    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2

    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_resized

    lab = cv2.cvtColor(padded, cv2.COLOR_RGB2LAB)

    L = lab[:, :, 0] / 255.0
    L = np.expand_dims(L, axis=0)

    # 🔥 RETURN everything needed
    meta = {
        "orig_h": h,
        "orig_w": w,
        "new_h": new_h,
        "new_w": new_w,
        "y_offset": y_offset,
        "x_offset": x_offset
    }

    return L.astype("float32"), meta




def postprocess_image(L, pred_ab, meta):
    import torch.nn.functional as F

    L_img = (L[0] * 255).astype("uint8")

    pred_ab = F.interpolate(
        pred_ab,
        size=(256, 256),
        mode='bilinear',
        align_corners=False
    )

    pred_ab = pred_ab.squeeze(0).cpu().numpy()

    ab = (pred_ab + 1) * 128
    ab = ab.transpose(1, 2, 0).astype("uint8")

    lab = np.zeros((256, 256, 3), dtype="uint8")
    lab[:, :, 0] = L_img
    lab[:, :, 1:] = ab

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    y, x = meta["y_offset"], meta["x_offset"]
    new_h, new_w = meta["new_h"], meta["new_w"]

    cropped = rgb[y:y+new_h, x:x+new_w]

    # Resize back to original
    final = cv2.resize(cropped, (meta["orig_w"], meta["orig_h"]))

    return final

