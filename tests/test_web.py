"""
Test suite for Flask Web Endpoints.
"""

import os
import sys
import unittest
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.web import app

class TestWebEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    def test_home_page(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"EPICS ROP Tri-Modal Diagnostic Suite", resp.data)

    def test_model_info_api(self):
        resp = self.client.get("/api/model_info")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("modalities", data["metadata"])

    def test_benchmark_catalog_api(self):
        resp = self.client.get("/api/benchmark_images?target=all")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIsInstance(data["catalog"], list)
        self.assertGreater(len(data["catalog"]), 0)

    def test_predict_endpoint(self):
        dummy_path = os.path.join(os.path.dirname(__file__), "data", "dummy_retina.jpg")
        with open(dummy_path, "rb") as f:
            resp = self.client.post("/api/predict", data={
                "image": (f, "test_retina.jpg")
            }, content_type="multipart/form-data")

        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("combined_overlay_base64", data)
        self.assertIn("ridge_overlay_base64", data)
        self.assertIn("od_overlay_base64", data)
        self.assertIn("bv_overlay_base64", data)
        self.assertIn("zones_overlay_base64", data)
        self.assertIn("metrics", data)

if __name__ == "__main__":
    unittest.main()
