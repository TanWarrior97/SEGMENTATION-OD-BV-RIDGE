# ROP Zone Calculation Methodology & Clinical Geometry

## Overview
Retinopathy of Prematurity (ROP) severity and treatment urgency are determined according to the **International Classification of Retinopathy of Prematurity (ICROP)** guidelines. The anatomical location of retinal lesions (such as the Demarcation Ridge) is categorized into three concentric zones centered on the **Optic Disc (OD)**.

---

## Anatomical Definitions

```text
               +-------------------------------------------+
               |                 ZONE III                  |
               |       +---------------------------+       |
               |       |         ZONE II           |       |
               |       |     +---------------+     |       |
               |       |     |    ZONE I     |     |       |
               |       |     |     (OD)      |     |       |
               |       |     |       *       |     |       |
               |       |     |      / \      |     |       |
               |       |     |   Macula      |     |       |
               |       |     +---------------+     |       |
               |       +---------------------------+       |
               +-------------------------------------------+
```

### 1. Optic Disc (OD) Landmark Detection
- The Optic Disc is segmented using the dedicated UNet++ EfficientNet-B4 network at $384 \times 384$ resolution.
- The minimum enclosing circle is fitted onto the dominant disc contour:
  $$\text{OD Center} = (c_x, c_y)$$
  $$\text{OD Diameter } (D_{OD}) = 2 \times R_{OD}$$

### 2. Distance to the Macula ($d_{\text{macula}}$)
In infant retinal fundus photography, the distance from the center of the optic disc to the foveal center is directly proportional to the optic disc diameter:
- **For RetCam Images**:
  $$d_{\text{macula}} \approx 3.0 \times D_{OD}$$
- **For Neo (Wide-field) Images**:
  $$d_{\text{macula}} \approx 2.75 \times D_{OD}$$

### 3. Concentric Zone Boundaries
- **Zone I**: A circle centered at the optic disc with radius equal to twice the distance from the center of the optic disc to the center of the macula:
  $$R_{\text{Zone I}} = 2 \times d_{\text{macula}}$$
  *Clinical Significance*: ROP in Zone I is high risk and requires urgent clinical management.

- **Zone II**: A circle centered at the optic disc extending from the outer edge of Zone I out to the nasal ora serrata:
  $$R_{\text{Zone II}} = 2.5 \times d_{\text{macula}}$$

- **Zone III**: The residual crescent-shaped area of the peripheral retina outside Zone II:
  $$\text{Zone III} = \text{Peripheral retina outside } R_{\text{Zone II}}$$

---

## Implementation in the EPICS ROP Engine

The geometric calculations are implemented in [`app/postprocessing.py`](file:///d:/S%20RIDGE%20-%20codex/HVD_ROP_EPICS-main/app/postprocessing.py):
```python
def extract_od_geometry(od_mask, image_width=None):
    contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
    od_center = (int(round(cx)), int(round(cy)))
    od_diameter = 2.0 * float(radius)

    is_neo = image_width is not None and image_width > 1000
    macula_distance = (2.75 if is_neo else 3.0) * od_diameter
    zone1_radius = int(round(2.0 * macula_distance))
    zone2_radius = int(round(2.5 * macula_distance))

    return {
        "center": od_center,
        "diameter": round(od_diameter, 2),
        "macula_distance": round(macula_distance, 2),
        "zone1_radius": zone1_radius,
        "zone2_radius": zone2_radius
    }
```
