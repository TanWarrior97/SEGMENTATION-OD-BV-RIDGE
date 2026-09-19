# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

datas = [
    (os.path.join(repo_root, "model", "best_model_RIDGE_manet.pth"), "model"),
    (os.path.join(repo_root, "model", "best_model_OD.pth"), "model"),
    (os.path.join(repo_root, "model", "optimized_best_model_BV.pth"), "model"),
    (os.path.join(repo_root, "model", "model_metadata.json"), "model"),
    (os.path.join(repo_root, "templates"), "templates"),
    (os.path.join(repo_root, "static"), "static"),
    (os.path.join(repo_root, "VERSION.txt"), "."),
]

hiddenimports = [
    "segmentation_models_pytorch",
    "timm",
    "albumentations",
    "albumentations.pytorch",
    "cv2",
    "PIL",
    "torch",
    "torchvision",
    "numpy",
    "skimage",
    "skimage.morphology",
    "flask",
]

a = Analysis(
    [os.path.join(repo_root, "app", "cli.py")],
    pathex=[repo_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ROPTriModalSegmenter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ROPTriModalSegmenter",
)
