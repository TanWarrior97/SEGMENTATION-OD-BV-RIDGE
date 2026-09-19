"""
Flask Web Application for EPICS ROP Tri-Modal Retinal Segmentation.
"""

import os
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

from app.predictor import ROPPredictor

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

PREDICTOR = None

def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = ROPPredictor(model_dir="model")
    return PREDICTOR

def encode_image_png(img_bgr_or_rgb, is_rgb=True):
    if is_rgb:
        img_bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_bgr_or_rgb
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG image.")
    return base64.b64encode(buf).decode("ascii")

def encode_mask_png(mask_binary):
    ok, buf = cv2.imencode(".png", (mask_binary * 255).astype(np.uint8))
    if not ok:
        raise RuntimeError("Failed to encode PNG mask.")
    return base64.b64encode(buf).decode("ascii")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/model_info", methods=["GET"])
def model_info():
    predictor = get_predictor()
    return jsonify({
        "status": "SUCCESS",
        "metadata": predictor.metadata,
        "device": "CUDA" if predictor.device.type == "cuda" else "CPU",
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"status": "ERROR", "error": "No image file provided."}), 400

    upload = request.files["image"]
    if upload.filename == "":
        return jsonify({"status": "ERROR", "error": "Selected file is empty."}), 400

    try:
        raw_bytes = upload.read()
        npimg = np.frombuffer(raw_bytes, np.uint8)
        image_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"status": "ERROR", "error": "Uploaded file is not a valid image."}), 400

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor = get_predictor()
        res = predictor.predict_all(image_rgb)

        return jsonify({
            "status": "SUCCESS",
            "original_base64": encode_image_png(image_rgb, is_rgb=True),
            "combined_overlay_base64": encode_image_png(res["overlays"]["combined"], is_rgb=True),
            "ridge_overlay_base64": encode_image_png(res["overlays"]["ridge"], is_rgb=True),
            "ridge_mask_base64": encode_mask_png(res["masks"]["ridge"]),
            "od_overlay_base64": encode_image_png(res["overlays"]["optic_disc"], is_rgb=True),
            "od_mask_base64": encode_mask_png(res["masks"]["optic_disc"]),
            "bv_overlay_base64": encode_image_png(res["overlays"]["blood_vessels"], is_rgb=True),
            "bv_mask_base64": encode_mask_png(res["masks"]["blood_vessels"]),
            "zones_overlay_base64": encode_image_png(res["overlays"]["zones"], is_rgb=True),
            "metrics": {
                "dimensions": res["image_dimensions"],
                "device": res["device"],
                "inference_time_sec": res["total_inference_time_sec"],
                "ridge": res["results"]["ridge"]["metrics"],
                "optic_disc": res["results"]["optic_disc"]["metrics"],
                "blood_vessels": res["results"]["blood_vessels"]["metrics"],
                "zones": res["results"]["zones"]["geometry"],
                "ridge_detected": res["results"]["ridge"]["detected"],
                "od_detected": res["results"]["optic_disc"]["detected"],
                "bv_detected": res["results"]["blood_vessels"]["detected"],
            }
        })
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 500
