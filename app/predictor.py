"""
Unified ROP Tri-Modal Predictor Engine.
Provides end-to-end clinical inference for Demarcation Ridge, Optic Disc, and Blood Vessels,
supporting both isolated single-modality queries and combined multi-modal analysis.
"""

import os
import time
import json
import cv2
import numpy as np
import torch

from .model_loader import load_ridge_model, load_od_model, load_bv_model
from .preprocessing import get_deployment_transform, prepare_image
from .postprocessing import (
    clean_ridge_mask,
    clean_od_mask,
    extract_od_geometry,
    clean_and_skeletonize_bv,
    create_single_overlay,
    create_zones_overlay,
    create_combined_overlay,
    COLOR_RIDGE,
    COLOR_OD,
    COLOR_BV
)

class ROPPredictor:
    """
    Unified clinical prediction engine for ROP diagnostic assessment.
    """
    def __init__(self, model_dir="model", device=None, lazy_load=False):
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load metadata
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Standard thresholds
        modalities = self.metadata.get("modalities", {})
        self.ridge_threshold = float(modalities.get("ridge", {}).get("decision_threshold", 0.69))
        self.od_threshold = float(modalities.get("optic_disc", {}).get("decision_threshold", 0.50))
        self.bv_threshold = float(modalities.get("blood_vessels", {}).get("decision_threshold", 0.72))

        # Standard transforms
        self.transform_512 = get_deployment_transform(512)
        self.transform_384 = get_deployment_transform(384)

        # Models storage
        self._ridge_model = None
        self._od_model = None
        self._bv_model = None

        if not lazy_load:
            self.warmup_all_models()

    def get_ridge_model(self):
        if self._ridge_model is None:
            self._ridge_model = load_ridge_model(self.model_dir, self.device)
        return self._ridge_model

    def get_od_model(self):
        if self._od_model is None:
            self._od_model = load_od_model(self.model_dir, self.device)
        return self._od_model

    def get_bv_model(self):
        if self._bv_model is None:
            self._bv_model = load_bv_model(self.model_dir, self.device)
        return self._bv_model

    def warmup_all_models(self):
        """Pre-loads all models into memory."""
        try:
            self.get_ridge_model()
            self.get_od_model()
            self.get_bv_model()
        except Exception as e:
            print(f"[Warning] Failed to pre-load all models: {e}")

    # =========================================================================
    # Individual Segmentation Methods
    # =========================================================================

    def predict_ridge(self, image_input, threshold=None):
        """
        Executes Demarcation Ridge segmentation (MAnet 512px).
        """
        start_time = time.time()
        image_rgb = prepare_image(image_input)
        orig_h, orig_w = image_rgb.shape[:2]
        thresh = threshold if threshold is not None else self.ridge_threshold

        # 1. Transform & Tensor
        tensor = self.transform_512(image=image_rgb)["image"].unsqueeze(0).to(self.device)

        # 2. Inference
        with torch.inference_mode():
            out = self.get_ridge_model()(tensor)
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()

        # 3. Resize to original dimensions
        prob_resized = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # 4. Clean morphological mask
        mask = clean_ridge_mask(prob_resized, image_rgb, threshold=thresh)
        detected = bool(mask.any())

        # 5. Metrics & Overlay
        ridge_pixels = int(np.sum(mask))
        coverage_pct = float((ridge_pixels / (orig_h * orig_w)) * 100.0)
        mean_conf = float(np.mean(prob_resized[mask == 1])) if detected else 0.0
        overlay_rgb = create_single_overlay(image_rgb, mask, COLOR_RIDGE, alpha=0.65)
        elapsed = float(time.time() - start_time)

        return {
            "status": "SUCCESS",
            "modality": "RIDGE",
            "detected": detected,
            "prediction": "RIDGE_DETECTED" if detected else "NO_RIDGE",
            "mask": mask,
            "probability_map": prob_resized,
            "overlay": overlay_rgb,
            "metrics": {
                "ridge_pixels": ridge_pixels,
                "coverage_percent": round(coverage_pct, 4),
                "mean_confidence": round(mean_conf, 4),
                "threshold_used": thresh,
            },
            "elapsed_seconds": round(elapsed, 4)
        }

    def predict_od(self, image_input, threshold=None):
        """
        Executes Optic Disc segmentation and geometric analysis (UNet++ 384px).
        """
        start_time = time.time()
        image_rgb = prepare_image(image_input)
        orig_h, orig_w = image_rgb.shape[:2]
        thresh = threshold if threshold is not None else self.od_threshold

        tensor = self.transform_384(image=image_rgb)["image"].unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out = self.get_od_model()(tensor)
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()

        prob_resized = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask = clean_od_mask(prob_resized, threshold=thresh)
        detected = bool(mask.any())

        geometry = extract_od_geometry(mask, image_width=orig_w)
        overlay_rgb = create_single_overlay(image_rgb, mask, COLOR_OD, alpha=0.65)
        elapsed = float(time.time() - start_time)

        return {
            "status": "SUCCESS",
            "modality": "OD",
            "detected": detected,
            "prediction": "OD_DETECTED" if detected else "NO_OD",
            "mask": mask,
            "probability_map": prob_resized,
            "overlay": overlay_rgb,
            "geometry": geometry,
            "metrics": {
                "od_pixels": int(np.sum(mask)),
                "center": geometry["center"] if geometry else None,
                "diameter": geometry["diameter"] if geometry else None,
                "threshold_used": thresh
            },
            "elapsed_seconds": round(elapsed, 4)
        }

    def predict_bv(self, image_input, threshold=None):
        """
        Executes Blood Vessel segmentation and skeletonization (UNet++ 384px).
        """
        start_time = time.time()
        image_rgb = prepare_image(image_input)
        orig_h, orig_w = image_rgb.shape[:2]
        thresh = threshold if threshold is not None else self.bv_threshold

        tensor = self.transform_384(image=image_rgb)["image"].unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out = self.get_bv_model()(tensor)
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()

        prob_resized = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask, dilated_vessels = clean_and_skeletonize_bv(prob_resized, threshold=thresh)
        detected = bool(dilated_vessels.any())

        vessel_pixels = int(np.sum(dilated_vessels))
        density_pct = float((vessel_pixels / (orig_h * orig_w)) * 100.0)
        overlay_rgb = create_single_overlay(image_rgb, dilated_vessels, COLOR_BV, alpha=0.65)
        elapsed = float(time.time() - start_time)

        return {
            "status": "SUCCESS",
            "modality": "BV",
            "detected": detected,
            "prediction": "BV_DETECTED" if detected else "NO_BV",
            "mask": dilated_vessels,
            "raw_binary_mask": binary_mask,
            "probability_map": prob_resized,
            "overlay": overlay_rgb,
            "metrics": {
                "vessel_pixels": vessel_pixels,
                "vessel_density_percent": round(density_pct, 4),
                "threshold_used": thresh
            },
            "elapsed_seconds": round(elapsed, 4)
        }

    def predict_zones(self, image_input):
        """
        Calculates ROP Zone I and Zone II clinical boundary rings based on Optic Disc detection.
        """
        image_rgb = prepare_image(image_input)
        od_result = self.predict_od(image_rgb)
        geometry = od_result.get("geometry")
        zones_overlay = create_zones_overlay(image_rgb, geometry, alpha=0.85)

        return {
            "status": "SUCCESS",
            "modality": "ZONES",
            "detected": bool(geometry is not None),
            "geometry": geometry,
            "overlay": zones_overlay,
            "od_mask": od_result["mask"],
        }

    # =========================================================================
    # Unified Combined Multi-Modal Prediction
    # =========================================================================

    def predict_all(self, image_input, save_dir=None, base_name=None):
        """
        Runs complete Tri-Modal diagnostic inference on a single retinal image:
        - Segments Demarcation Ridge
        - Segments Optic Disc and computes Zone boundaries
        - Segments Blood Vessel network
        - Produces unified multimodal overlay and individual diagnostic maps
        """
        start_time = time.time()
        image_rgb = prepare_image(image_input)
        orig_h, orig_w = image_rgb.shape[:2]

        # 1. Run all 3 modalities
        ridge_res = self.predict_ridge(image_rgb)
        od_res = self.predict_od(image_rgb)
        bv_res = self.predict_bv(image_rgb)

        # 2. Extract masks and geometry
        ridge_mask = ridge_res["mask"]
        od_mask = od_res["mask"]
        bv_mask = bv_res["mask"]
        od_geom = od_res["geometry"]

        # 3. Create unified combined overlay
        combined_overlay = create_combined_overlay(
            image_rgb,
            ridge_mask=ridge_mask,
            od_mask=od_mask,
            bv_mask=bv_mask,
            od_geometry=od_geom,
            show_ridge=True,
            show_od=True,
            show_bv=True,
            show_zones=True,
            alpha=0.65
        )

        elapsed = float(time.time() - start_time)

        # 4. Optional disk exports
        saved_paths = {}
        if save_dir and base_name:
            os.makedirs(save_dir, exist_ok=True)
            stem = os.path.splitext(base_name)[0]

            def _save_img(filename, rgb_data):
                p = os.path.join(save_dir, filename)
                cv2.imwrite(p, cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))
                return p

            def _save_mask(filename, mask_data):
                p = os.path.join(save_dir, filename)
                cv2.imwrite(p, (mask_data * 255).astype(np.uint8))
                return p

            saved_paths["combined_overlay"] = _save_img(f"{stem}_combined_overlay.png", combined_overlay)
            saved_paths["ridge_overlay"] = _save_img(f"{stem}_ridge_overlay.png", ridge_res["overlay"])
            saved_paths["ridge_mask"] = _save_mask(f"{stem}_ridge_mask.png", ridge_mask)
            saved_paths["od_overlay"] = _save_img(f"{stem}_od_overlay.png", od_res["overlay"])
            saved_paths["od_mask"] = _save_mask(f"{stem}_od_mask.png", od_mask)
            saved_paths["bv_overlay"] = _save_img(f"{stem}_bv_overlay.png", bv_res["overlay"])
            saved_paths["bv_mask"] = _save_mask(f"{stem}_bv_mask.png", bv_mask)

        return {
            "status": "SUCCESS",
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "device": "CUDA" if self.device.type == "cuda" else "CPU",
            "total_inference_time_sec": round(elapsed, 4),
            "results": {
                "ridge": ridge_res,
                "optic_disc": od_res,
                "blood_vessels": bv_res,
                "zones": {
                    "detected": bool(od_geom is not None),
                    "geometry": od_geom,
                }
            },
            "overlays": {
                "combined": combined_overlay,
                "ridge": ridge_res["overlay"],
                "optic_disc": od_res["overlay"],
                "blood_vessels": bv_res["overlay"],
                "zones": create_zones_overlay(image_rgb, od_geom),
            },
            "masks": {
                "ridge": ridge_mask,
                "optic_disc": od_mask,
                "blood_vessels": bv_mask,
            },
            "saved_paths": saved_paths
        }
