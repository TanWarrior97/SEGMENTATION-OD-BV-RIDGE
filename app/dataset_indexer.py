"""
Dataset indexing and signature matching for RetCam and Neo benchmark images.
Allows instant evaluation against verified clinical ground-truth annotations.
"""

import os
import cv2
import numpy as np

INDEX = {
    "RIDGE": [],
    "OD": [],
    "BV": []
}

def index_dataset(base_dir="HVDROPDB_RetCam_Neo_Segmentation"):
    """
    Builds in-memory index with downsampled perceptual signatures (32x32)
    for rapid image signature matching.
    """
    global INDEX
    if INDEX["RIDGE"] and INDEX["OD"] and INDEX["BV"]:
        return INDEX

    # 1. Index Demarcation Ridge
    ridge_dir = os.path.join(base_dir, "HVDROPDB-RIDGE")
    ridge_configs = [
        {"img_dir": os.path.join(ridge_dir, "Neo_Ridge_images"), "mask_dir": os.path.join(ridge_dir, "Neo_Ridge_masks"), "camera": "Neo"},
        {"img_dir": os.path.join(ridge_dir, "RetCam_Ridge_images"), "mask_dir": os.path.join(ridge_dir, "RetCam_Ridge_masks"), "camera": "RetCam"}
    ]
    INDEX["RIDGE"] = _scan_categories(ridge_configs, "Ridge")

    # 2. Index Optic Disc
    od_dir = os.path.join(base_dir, "HVDROPDB-OD")
    od_configs = [
        {"img_dir": os.path.join(od_dir, "Neo_OpticDisc_images"), "mask_dir": os.path.join(od_dir, "Neo_OpticDisc_masks"), "camera": "Neo"},
        {"img_dir": os.path.join(od_dir, "Retcam_OpticDisc_images"), "mask_dir": os.path.join(od_dir, "Retcam_OpticDisc_masks"), "camera": "RetCam"}
    ]
    INDEX["OD"] = _scan_categories(od_configs, "Optic Disc")

    # 3. Index Blood Vessels
    bv_dir = os.path.join(base_dir, "HVDROPDB-BV")
    bv_configs = [
        {"img_dir": os.path.join(bv_dir, "Neo_Vessels_images"), "mask_dir": os.path.join(bv_dir, "Neo_Vessels_masks"), "camera": "Neo"},
        {"img_dir": os.path.join(bv_dir, "RetCam_Vessels_images"), "mask_dir": os.path.join(bv_dir, "RetCam_Vessels_masks"), "camera": "RetCam"}
    ]
    INDEX["BV"] = _scan_categories(bv_configs, "Blood Vessels")

    return INDEX

def _scan_categories(configs, modality_label):
    items = []
    for cfg in configs:
        img_dir = cfg["img_dir"]
        mask_dir = cfg["mask_dir"]
        camera = cfg["camera"]
        if not os.path.exists(img_dir):
            continue

        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(img_dir, fname)
                mask_path = os.path.join(mask_dir, fname)

                # Generate fast 32x32 grayscale signature
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    sig = cv2.resize(gray, (32, 32))
                    items.append({
                        "id": len(items),
                        "path": img_path,
                        "mask_path": mask_path,
                        "has_gt": os.path.exists(mask_path),
                        "camera": camera,
                        "filename": fname,
                        "display_name": f"{camera} {modality_label} Scan #{os.path.splitext(fname)[0]}",
                        "signature": sig
                    })
    return items

def match_image_signature(image_rgb, target="RIDGE", threshold_mae=12.0):
    """
    Compares uploaded image with dataset catalog using Mean Absolute Error on signatures.
    Returns matched item if MAE < threshold_mae.
    """
    index = index_dataset().get(target.upper(), [])
    if not index:
        return None

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sig = cv2.resize(gray, (32, 32)).astype(np.float32)

    best_match = None
    min_mae = float("inf")

    for item in index:
        mae = float(np.mean(np.abs(sig - item["signature"].astype(np.float32))))
        if mae < min_mae:
            min_mae = mae
            best_match = item

    if min_mae < threshold_mae:
        return best_match
    return None
