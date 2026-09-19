"""
Unit and Integration Tests for ROP Tri-Modal Diagnostic Pipeline.
Tests individual and combined segmentation across:
- Demarcation Ridge (MAnet)
- Optic Disc (UNet++)
- Blood Vessels (UNet++)
- ROP Zone Mapping
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predictor import ROPPredictor
from app.preprocessing import prepare_image, get_deployment_transform
from app.postprocessing import extract_od_geometry, clean_and_skeletonize_bv

class TestROPTriModalPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_image_path = os.path.join(
            os.path.dirname(__file__), "data", "dummy_retina.jpg"
        )
        if not os.path.exists(cls.test_image_path):
            raise FileNotFoundError(f"Missing test image at {cls.test_image_path}")
        cls.predictor = ROPPredictor(model_dir="model")

    def test_image_preparation(self):
        """Tests image loader handles filepaths and arrays properly."""
        img_rgb = prepare_image(self.test_image_path)
        self.assertIsInstance(img_rgb, np.ndarray)
        self.assertEqual(img_rgb.ndim, 3)
        self.assertEqual(img_rgb.shape[2], 3)
        self.assertEqual(img_rgb.dtype, np.uint8)

    def test_transforms_shapes(self):
        """Tests deployment transforms output expected PyTorch tensor dimensions."""
        img_rgb = prepare_image(self.test_image_path)
        t512 = get_deployment_transform(512)(image=img_rgb)["image"]
        t384 = get_deployment_transform(384)(image=img_rgb)["image"]

        self.assertEqual(t512.shape, (3, 512, 512))
        self.assertEqual(t384.shape, (3, 384, 384))

    def test_ridge_prediction(self):
        """Tests Demarcation Ridge inference and mask output."""
        res = self.predictor.predict_ridge(self.test_image_path)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["modality"], "RIDGE")
        self.assertIn("detected", res)
        self.assertIn("mask", res)
        self.assertIn("overlay", res)
        self.assertIn("metrics", res)

        orig = prepare_image(self.test_image_path)
        self.assertEqual(res["mask"].shape, orig.shape[:2])
        self.assertEqual(res["overlay"].shape, orig.shape)

    def test_optic_disc_prediction(self):
        """Tests Optic Disc inference and geometric parameter extraction."""
        res = self.predictor.predict_od(self.test_image_path)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["modality"], "OD")
        self.assertIn("detected", res)
        self.assertIn("mask", res)
        self.assertIn("overlay", res)

        orig = prepare_image(self.test_image_path)
        self.assertEqual(res["mask"].shape, orig.shape[:2])

    def test_blood_vessels_prediction(self):
        """Tests Blood Vessels inference and skeletonization."""
        res = self.predictor.predict_bv(self.test_image_path)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["modality"], "BV")
        self.assertIn("detected", res)
        self.assertIn("mask", res)
        self.assertIn("overlay", res)
        self.assertIn("vessel_density_percent", res["metrics"])

        orig = prepare_image(self.test_image_path)
        self.assertEqual(res["mask"].shape, orig.shape[:2])

    def test_predict_all_combined(self):
        """Tests complete unified Tri-Modal inference and combined overlay generation."""
        res = self.predictor.predict_all(self.test_image_path)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("image_dimensions", res)
        self.assertIn("total_inference_time_sec", res)

        results = res["results"]
        self.assertIn("ridge", results)
        self.assertIn("optic_disc", results)
        self.assertIn("blood_vessels", results)
        self.assertIn("zones", results)

        overlays = res["overlays"]
        self.assertIn("combined", overlays)
        self.assertIn("ridge", overlays)
        self.assertIn("optic_disc", overlays)
        self.assertIn("blood_vessels", overlays)
        self.assertIn("zones", overlays)

        orig = prepare_image(self.test_image_path)
        self.assertEqual(overlays["combined"].shape, orig.shape)

if __name__ == "__main__":
    unittest.main()
