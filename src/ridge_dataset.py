import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _list_images(directory):
    valid = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted(
        f for f in os.listdir(directory) if f.lower().endswith(valid)
    )


def _resolve_mask_path(masks_dir, img_name):
    base = os.path.splitext(img_name)[0]
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = os.path.join(masks_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(masks_dir, img_name)


def _load_binary_mask(path, target_shape=None):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found/readable: {path}")
    if target_shape is not None and mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return mask.astype(np.float32)


class CombinedRidgeDataset(Dataset):
    """Neo + RetCam ridge dataset with optional OD/BV context channels."""

    def __init__(
        self,
        ridge_root,
        od_root=None,
        bv_root=None,
        transform=None,
        use_auxiliary=True,
        devices=("neo", "retcam"),
    ):
        self.transform = transform
        self.use_auxiliary = use_auxiliary
        self.samples = []

        device_map = {
            "neo": {
                "images": ("Neo_Ridge_images",),
                "masks": ("Neo_Ridge_masks",),
                "od_images": ("Neo_OpticDisc_images",),
                "od_masks": ("Neo_OpticDisc_masks",),
                "bv_images": ("Neo_Vessels_images",),
                "bv_masks": ("Neo_Vessels_masks",),
            },
            "retcam": {
                "images": ("RetCam_Ridge_images",),
                "masks": ("RetCam_Ridge_masks",),
                "od_images": ("Retcam_OpticDisc_images",),
                "od_masks": ("Retcam_OpticDisc_masks",),
                "bv_images": ("RetCam_Vessels_images",),
                "bv_masks": ("RetCam_Vessels_masks",),
            },
        }

        for device in devices:
            cfg = device_map[device]
            img_dir = os.path.join(ridge_root, cfg["images"][0])
            mask_dir = os.path.join(ridge_root, cfg["masks"][0])
            if not os.path.isdir(img_dir):
                continue

            od_img_dir = od_root and os.path.join(od_root, cfg["od_images"][0])
            od_mask_dir = od_root and os.path.join(od_root, cfg["od_masks"][0])
            bv_img_dir = bv_root and os.path.join(bv_root, cfg["bv_images"][0])
            bv_mask_dir = bv_root and os.path.join(bv_root, cfg["bv_masks"][0])

            for img_name in _list_images(img_dir):
                sample = {
                    "image_path": os.path.join(img_dir, img_name),
                    "mask_path": _resolve_mask_path(mask_dir, img_name),
                    "device": device,
                }
                if use_auxiliary and od_mask_dir and os.path.isdir(od_mask_dir):
                    sample["od_mask_path"] = _resolve_mask_path(od_mask_dir, img_name)
                if use_auxiliary and bv_mask_dir and os.path.isdir(bv_mask_dir):
                    sample["bv_mask_path"] = _resolve_mask_path(bv_mask_dir, img_name)
                self.samples.append(sample)

        if not self.samples:
            raise RuntimeError("No ridge samples found. Check dataset paths.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise FileNotFoundError(f"Image not found/readable: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ridge_mask = _load_binary_mask(sample["mask_path"], target_shape=image.shape[:2])

        auxiliary = []
        if self.use_auxiliary:
            h, w = image.shape[:2]
            od_mask = np.zeros((h, w), dtype=np.float32)
            bv_mask = np.zeros((h, w), dtype=np.float32)
            if "od_mask_path" in sample and os.path.exists(sample["od_mask_path"]):
                od_mask = _load_binary_mask(sample["od_mask_path"], target_shape=(h, w))
            if "bv_mask_path" in sample and os.path.exists(sample["bv_mask_path"]):
                bv_mask = _load_binary_mask(sample["bv_mask_path"], target_shape=(h, w))
            auxiliary = [od_mask, bv_mask]

        if self.transform:
            if auxiliary:
                augmented = self.transform(
                    image=image,
                    masks=[ridge_mask, auxiliary[0], auxiliary[1]],
                )
                image = augmented["image"]
                ridge_mask, auxiliary[0], auxiliary[1] = augmented["masks"]
            else:
                augmented = self.transform(image=image, mask=ridge_mask)
                image = augmented["image"]
                ridge_mask = augmented["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if not isinstance(ridge_mask, torch.Tensor):
            ridge_mask = torch.from_numpy(ridge_mask).float()
        else:
            ridge_mask = ridge_mask.float()

        if auxiliary:
            if not isinstance(auxiliary[0], torch.Tensor):
                auxiliary = [
                    torch.from_numpy(auxiliary[0]).float(),
                    torch.from_numpy(auxiliary[1]).float(),
                ]
            else:
                auxiliary = [auxiliary[0].float(), auxiliary[1].float()]
            context = torch.stack(auxiliary, dim=0)
            image = torch.cat([image, context], dim=0)

        return image, ridge_mask.unsqueeze(0)


def get_ridge_transforms(img_size=(512, 512), phase="train"):
    if phase == "train":
        return A.Compose(
            [
                A.Resize(img_size[0], img_size[1]),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.25, p=0.5),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            additional_targets={"masks": "masks"},
            is_check_shapes=False,
        )
    return A.Compose(
        [
            A.Resize(img_size[0], img_size[1]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
