"""
Preprocessing and normalization transforms for ROP Tri-Modal Retinal Segmentation.
Supports dual-resolution pipelines: 512x512 (Ridge) and 384x384 (Optic Disc & Blood Vessels).
"""

import io
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_deployment_transform(img_size=512):
    """
    Returns deployment preprocessing transform.
    Standardizes image size, enhances contrast via CLAHE, and normalizes using ImageNet statistics.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def prepare_image(image_input):
    """
    Converts various input formats (file path, bytes, PIL Image, NumPy array)
    into a standardized uint8 RGB NumPy ndarray (H, W, 3).
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
        return np.array(pil_img)
    elif isinstance(image_input, bytes):
        npimg = np.frombuffer(image_input, np.uint8)
        bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image from bytes.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3 and image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        elif image_input.ndim == 3 and image_input.shape[2] == 3:
            return image_input
        else:
            raise ValueError(f"Unexpected image array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")
