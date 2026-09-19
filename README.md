# 🔬 EPICS ROP Tri-Modal Diagnostic Platform

> **Unified AI Framework for Retinopathy of Prematurity (ROP) Clinical Diagnosis**  
> Integrated deep learning segmentation for **Demarcation Ridge**, **Optic Disc (OD)**, **Blood Vessels (BV)**, and automated **ROP Zone Mapping** (ICROP Guidelines).

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()

---

## 🌟 Key Capabilities

1. 🔵 **Demarcation Ridge Segmentation**:
   - **Architecture**: Multi-Scale Attention Network (**MAnet**) + **EfficientNet-B4** ($512 \times 512$).
   - **Clinical Logic**: Identifies boundary between vascularized and avascular retina; eliminates false-positive camera edge rim artifacts via connected-component symmetry analysis.
   - **Dice Score**: `74.74%` | **IoU**: `60.15%`.

2. 🔴 **Optic Disc (OD) Localization & Geometry**:
   - **Architecture**: **UNet++** + **EfficientNet-B4** ($384 \times 384$).
   - **Clinical Logic**: Detects geometric disc contour, center $(c_x, c_y)$, and diameter ($D_{OD}$).
   - **Dice Score**: `92.30%` | **IoU**: `85.70%`.

3. 🟢 **Retinal Blood Vessel (BV) Extraction**:
   - **Architecture**: **UNet++** + **EfficientNet-B4** ($384 \times 384$).
   - **Clinical Logic**: High-sensitivity vascular network segmentation followed by morphological skeletonization and uniform 3px dilation for crisp clinical visibility.
   - **Dice Score**: `84.10%` | **IoU**: `72.60%`.

4. 🗺️ **Automated ROP Zone Mapping (ICROP Guidelines)**:
   - Evaluates disc-to-macula distance: $d_{\text{macula}} = 2.75 \times D_{OD}$ (Neo) or $3.0 \times D_{OD}$ (RetCam).
   - Generates concentric clinical circles:
     - **Zone I**: Radius $= 2 \times d_{\text{macula}}$
     - **Zone II**: Radius $= 2.5 \times d_{\text{macula}}$
     - **Zone III**: Crescent outside Zone II extending to peripheral retina.

5. 🎨 **Combined vs. Individual Display System**:
   - **Combined Multimodal View**: Fundus image overlaid simultaneously with Ridge (Cyan), OD (Red), BV (Emerald Green), and ROP Zone rings.
   - **Real-Time Interactive Controls**: Toggle individual layer visibility on/off and dynamically modulate overlay opacity (10%–100%) in real time.
   - **Individual Views**: Dedicated cards and tabs displaying each mask, overlay, pixel counts, coverage %, and confidence scores.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/TanWarrior97/SEGMENTATION-OD-BV-RIDGE.git
cd SEGMENTATION-OD-BV-RIDGE
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **GPU Support (Recommended)**:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 4. Fetch Trained Model Weights
All three verified model weights can be downloaded with a single command:
```bash
python download_models.py
```
This fetches:
- `model/best_model_RIDGE_manet.pth` (~98.5 MB)
- `model/best_model_OD.pth` (~80.2 MB)
- `model/optimized_best_model_BV.pth` (~80.2 MB)

---

## 🖥️ Interactive Web Application

Launch the Flask clinical dashboard:
```bash
python app.py
```
Open your browser at: **`http://localhost:5000`**

### Web Features:
- **Drag & Drop Upload**: Upload any infant retinal fundus image (`.png`, `.jpg`, `.jpeg`).
- **Benchmark Catalog**: One-click evaluation of pre-indexed RetCam and Neo reference cases.
- **Ground Truth Comparison**: Automatic signature matching to display side-by-side ophthalmologist annotations.
- **Export Assets**: Download high-resolution PNG masks and colored overlays for each anatomical structure.

---

## 💻 Command-Line Interface (CLI)

The CLI tool enables automated batch and single-file processing:

### 1. Run Complete Tri-Modal Diagnostic on a Single Image
```bash
python app/cli.py --input tests/data/dummy_retina.jpg --target all --output-dir outputs/
```

### 2. Run Single Target Modality
```bash
# Demarcation Ridge only
python app/cli.py --input path/to/image.png --target ridge

# Optic Disc only
python app/cli.py --input path/to/image.png --target od

# Blood Vessels only
python app/cli.py --input path/to/image.png --target bv

# ROP Zones only
python app/cli.py --input path/to/image.png --target zones
```

### 3. Batch Process an Entire Directory with JSON Export
```bash
python app/cli.py --input path/to/dataset/ --target all --output-dir batch_results/ --json
```

---

## 📦 Windows MSI & Standalone Packaging

To compile a standalone Windows executable (`.exe`) and installer (`.msi`):
```powershell
# In PowerShell:
.\packaging\build_msi.ps1
```
The compiled distribution will be in `dist/ROPTriModalSegmenter/` and the Windows installer will be `ROP_TriModal_Diagnostic_Setup_v2.1.0.msi`.

---

## 🧪 Automated Testing

Execute the test suite to verify model inference and geometric pipelines:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📁 Repository Structure

```text
├── app/
│   ├── __init__.py
│   ├── cli.py                  # Production CLI interface
│   ├── dataset_indexer.py      # Dataset signature indexer & GT matching
│   ├── model_loader.py         # Multi-model loader (MAnet, UNet++)
│   ├── postprocessing.py       # Camera rim filter, skeletonize, Zone geometry
│   ├── predictor.py            # Unified ROPPredictor engine
│   └── preprocessing.py        # Dual-resolution transforms (512px / 384px)
├── model/
│   ├── model_metadata.json     # Model specs, thresholds, and performance metrics
│   ├── best_model_RIDGE_manet.pth # Ridge weights (103 MB)
│   ├── best_model_OD.pth       # Optic Disc weights (84 MB)
│   └── optimized_best_model_BV.pth # Blood Vessel weights (84 MB)
├── packaging/
│   ├── Product.wxs             # WiX Toolset installer definition
│   ├── README.md               # Packaging documentation
│   ├── ROPSegmenter.spec       # PyInstaller standalone build spec
│   └── build_msi.ps1           # Automated PowerShell installer build script
├── static/
│   ├── css/style.css           # Modern clinical dashboard styling
│   └── js/app.js               # Dynamic layer toggles & opacity controls
├── templates/
│   └── index.html              # Interactive web dashboard template
├── tests/
│   ├── data/                   # Test assets
│   └── test_inference.py       # Test suite for all modalities
├── .gitattributes              # Git LFS tracking for weights & binaries
├── .gitignore                  # Git ignore rules
├── app.py                      # Flask web application entrypoint
├── download_models.py          # Zero-friction model weights downloader
├── requirements.txt            # Python dependencies
├── VERSION.txt                 # Version identifier (v2.1.0)
└── Zone_Calculation_Methodology.md # ICROP clinical geometry documentation
```

---

## 📖 Citation & Acknowledgments
- **HVDROPDB Dataset**: RetCam and Neo infant fundus database.
- **ICROP-3**: International Classification of Retinopathy of Prematurity, Third Edition.
- Developed by the **EPICS ROP AI Engineering Team**.
