"""
EPICS ROP Tri-Modal Retinal Segmentation & Diagnostic Package
Unified inference engine for Demarcation Ridge, Optic Disc, and Blood Vessels.
"""

from .predictor import ROPPredictor
from .model_loader import load_ridge_model, load_od_model, load_bv_model

__version__ = "2.1.0"
__all__ = ["ROPPredictor", "load_ridge_model", "load_od_model", "load_bv_model"]
