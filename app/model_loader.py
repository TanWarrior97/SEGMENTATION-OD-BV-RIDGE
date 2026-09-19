"""
Model loading and lifecycle management for ROP Tri-Modal Inference.
Supports:
- Demarcation Ridge: MAnet + EfficientNet-B4
- Optic Disc (OD): UNet++ + EfficientNet-B4
- Blood Vessels (BV): UNet++ + EfficientNet-B4
"""

import os
import torch
import segmentation_models_pytorch as smp

def get_model_path(model_dir, filename):
    """
    Returns the absolute path to a model weights file.
    Checks model_dir, outputs/, and parent directories.
    """
    candidates = [
        os.path.join(model_dir, filename),
        os.path.join("outputs", filename),
        os.path.join("..", "model", filename),
        os.path.join("..", "outputs", filename),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 1000000:
            return c
    return os.path.join(model_dir, filename)

def load_ridge_model(model_dir="model", device=None):
    """
    Loads the MAnet + EfficientNet-B4 ROP Ridge Demarcation model checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_model_path(model_dir, "best_model_RIDGE_manet.pth")
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 1000000:
        raise FileNotFoundError(f"Missing Demarcation Ridge checkpoint at {checkpoint_path}. Run 'python download_models.py' to fetch weights.")

    model = smp.MAnet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_od_model(model_dir="model", device=None):
    """
    Loads the UNet++ + EfficientNet-B4 Optic Disc (OD) segmentation model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_model_path(model_dir, "best_model_OD.pth")
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 1000000:
        raise FileNotFoundError(f"Missing Optic Disc checkpoint at {checkpoint_path}. Run 'python download_models.py' to fetch weights.")

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_bv_model(model_dir="model", device=None):
    """
    Loads the UNet++ + EfficientNet-B4 Retinal Blood Vessel (BV) segmentation model.
    Checks for optimized weights first, then falls back to standard weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = get_model_path(model_dir, "optimized_best_model_BV.pth")
    if not (os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1000000):
        checkpoint_path = get_model_path(model_dir, "best_model_BV.pth")

    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 1000000:
        raise FileNotFoundError(f"Missing Blood Vessel checkpoint at {checkpoint_path}. Run 'python download_models.py' to fetch weights.")

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
