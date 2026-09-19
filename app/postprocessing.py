"""
Clinical post-processing and geometric analysis for ROP Tri-Modal Segmentation.
Features:
- Ridge: Fundus illumination mask, camera rim twin rejection, area filter.
- Optic Disc: Center localization, diameter extraction, circle fitting.
- Blood Vessels: Skeletonization to 1px centerlines, 3px uniform stroke dilation.
- ROP Zones: Automated concentric Zone I & Zone II geometry (ICROP guidelines).
- Multi-Structure Blending: Individual and combined overlay rendering.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize

# Default Color Palette (RGB) — Bold, high-contrast against dark retinal fundus backgrounds
# No red or green — all chosen to stand out from the warm red/brown retinal background
COLOR_RIDGE = (255, 230, 0)       # Electric Yellow — Ridge demarcation line
COLOR_OD = (255, 0, 200)          # Hot Magenta — Optic Disc
COLOR_BV = (0, 80, 220)           # Dark Blue — Blood Vessels / Optic Nerves
COLOR_ZONE1 = (160, 0, 255)       # Vivid Purple — Zone I boundary ring
COLOR_ZONE2 = (0, 160, 255)       # Sky Blue — Zone II boundary ring
COLOR_OD_CENTER = (255, 255, 255) # White — OD center landmark (maximum contrast)


def fundus_field(image_rgb):
    """
    Extracts the illuminated retinal field to reject out-of-boundary false positives
    and dark circular camera rims.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    raw = (gray > 12).astype(np.uint8)
    total, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    if total <= 1:
        return np.ones(gray.shape, np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    field = (labels == largest).astype(np.uint8)
    # Small erosion removes the black camera boundary ring
    radius = max(1, round(min(gray.shape) * 0.012))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(field, kernel)

def clean_ridge_mask(probability_map, image_rgb, threshold=0.69):
    """
    Isolates the ROP demarcation ridge by applying thresholding, fundus boundary clipping,
    minimum area filtering, and symmetric camera rim twin artifact rejection.
    """
    binary = ((probability_map >= threshold).astype(np.uint8) & fundus_field(image_rgb))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(binary)
    minimum_area = max(80, round(binary.size * 0.00015))
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum_area]

    # Rejection rule for symmetric, similarly sized camera-rim edge twins
    if len(keep) == 2:
        left, right = keep
        h, w = binary.shape
        symmetry = (
            abs(centres[left][0] + centres[right][0] - w) < 0.12 * w
            and abs(centres[left][1] - centres[right][1]) < 0.18 * h
        )
        sizes = min(stats[left, cv2.CC_STAT_AREA], stats[right, cv2.CC_STAT_AREA]) / max(
            stats[left, cv2.CC_STAT_AREA], stats[right, cv2.CC_STAT_AREA]
        )
        if symmetry and sizes > 0.55:
            keep = []

    output = np.zeros_like(binary, dtype=np.uint8)
    for label in keep:
        output[labels == label] = 1
    return output

def clean_od_mask(probability_map, threshold=0.50):
    """
    Cleans Optic Disc binary mask, retaining the dominant contiguous disc structure.
    """
    binary = (probability_map >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    # Keep largest component as the Optic Disc
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    output = (labels == largest_idx).astype(np.uint8)
    return output

def extract_od_geometry(od_mask, image_width=None):
    """
    Calculates geometric properties of Optic Disc:
    - Center (cx, cy)
    - Radius and Diameter
    - ROP Zone I and Zone II boundaries
    """
    contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
    od_center = (int(round(cx)), int(round(cy)))
    od_radius = float(radius)
    od_diameter = 2.0 * od_radius

    # Neo images typically have raw width > 1000px
    is_neo = image_width is not None and image_width > 1000
    macula_distance = (2.75 if is_neo else 3.0) * od_diameter
    zone1_radius = int(round(2.0 * macula_distance))
    zone2_radius = int(round(2.5 * macula_distance))

    return {
        "center": od_center,
        "radius": round(od_radius, 2),
        "diameter": round(od_diameter, 2),
        "macula_distance": round(macula_distance, 2),
        "zone1_radius": zone1_radius,
        "zone2_radius": zone2_radius,
        "area_pixels": int(cv2.contourArea(largest_contour)),
        "is_neo": is_neo,
    }

def clean_and_skeletonize_bv(probability_map, threshold=0.72, dilate_kernel_size=3):
    """
    Segments retinal blood vessels, skeletonizes to isolate centerlines,
    and applies uniform dilation to ensure crisp, standardized vessel visualization.
    """
    binary = (probability_map >= threshold).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=np.uint8), np.zeros_like(binary, dtype=np.uint8)

    # Skeletonize to 1px centerline
    skeleton = skeletonize(binary > 0).astype(np.uint8)

    # Uniform dilation for clinical visibility
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated_vessels = cv2.dilate(skeleton, kernel, iterations=1)

    return binary, dilated_vessels

def create_single_overlay(image_rgb, mask, color, alpha=0.65):
    """
    Blends a single binary mask over the RGB retinal image.
    """
    overlay = image_rgb.copy()
    mask_bool = (mask > 0)
    overlay[mask_bool] = color
    return cv2.addWeighted(image_rgb, 1.0 - alpha, overlay, alpha, 0)

def create_zones_overlay(image_rgb, od_geometry, alpha=0.85):
    """
    Draws ROP Zone I and Zone II concentric circle boundaries and OD landmark.
    """
    annotated = image_rgb.copy()
    if od_geometry is None:
        return annotated

    center = od_geometry["center"]
    z1_r = od_geometry["zone1_radius"]
    z2_r = od_geometry["zone2_radius"]

    # Draw Zone I (Gold / Amber)
    cv2.circle(annotated, center, z1_r, COLOR_ZONE1, 3, lineType=cv2.LINE_AA)
    # Draw Zone II (Light Green)
    cv2.circle(annotated, center, z2_r, COLOR_ZONE2, 3, lineType=cv2.LINE_AA)
    # Mark Optic Disc Center landmark (Yellow)
    cv2.circle(annotated, center, 6, COLOR_OD_CENTER, -1, lineType=cv2.LINE_AA)
    cv2.circle(annotated, center, 8, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    return cv2.addWeighted(image_rgb, 1.0 - alpha, annotated, alpha, 0)

def create_combined_overlay(
    image_rgb,
    ridge_mask=None,
    od_mask=None,
    bv_mask=None,
    od_geometry=None,
    show_ridge=True,
    show_od=True,
    show_bv=True,
    show_zones=True,
    alpha=0.65
):
    """
    Creates the unified multimodal overlay blending all active anatomical structures
    simultaneously without destructive occlusion.
    """
    blended = image_rgb.copy()
    color_layer = image_rgb.copy()
    mask_combined = np.zeros(image_rgb.shape[:2], dtype=bool)

    # 1. Optic Disc Layer (Red)
    if show_od and od_mask is not None and od_mask.any():
        od_bool = (od_mask > 0)
        color_layer[od_bool] = COLOR_OD
        mask_combined |= od_bool

    # 2. Blood Vessel Layer (Emerald Green)
    if show_bv and bv_mask is not None and bv_mask.any():
        bv_bool = (bv_mask > 0)
        color_layer[bv_bool] = COLOR_BV
        mask_combined |= bv_bool

    # 3. Demarcation Ridge Layer (Vibrant Cyan)
    if show_ridge and ridge_mask is not None and ridge_mask.any():
        ridge_bool = (ridge_mask > 0)
        color_layer[ridge_bool] = COLOR_RIDGE
        mask_combined |= ridge_bool

    # Blend colored segmentations
    if mask_combined.any():
        blended[mask_combined] = cv2.addWeighted(
            image_rgb[mask_combined], 1.0 - alpha,
            color_layer[mask_combined], alpha, 0
        )

    # 4. ROP Zones Layer (Vector boundary rings over blended image)
    if show_zones and od_geometry is not None:
        center = od_geometry["center"]
        z1_r = od_geometry["zone1_radius"]
        z2_r = od_geometry["zone2_radius"]

        # Zone I Boundary
        cv2.circle(blended, center, z1_r, COLOR_ZONE1, 3, lineType=cv2.LINE_AA)
        # Zone II Boundary
        cv2.circle(blended, center, z2_r, COLOR_ZONE2, 3, lineType=cv2.LINE_AA)
        # Optic Disc Center
        cv2.circle(blended, center, 5, COLOR_OD_CENTER, -1, lineType=cv2.LINE_AA)

    return blended
