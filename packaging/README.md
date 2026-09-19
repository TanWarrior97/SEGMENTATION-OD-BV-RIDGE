# Standalone Packaging & MSI Installer Guide

This directory contains configuration files to compile and package the **EPICS ROP Tri-Modal Diagnostic Suite** into a standalone Windows installer (`.msi` / `.exe`).

## Prerequisites
1. **PyInstaller**:
   ```bash
   pip install pyinstaller
   ```
2. **WiX Toolset** (Optional, for `.msi` generation):
   - WiX v3.11+ or WiX v4+:
     ```powershell
     winget install WiX
     ```

## Build Process

Run the automated build script in PowerShell:
```powershell
.\packaging\build_msi.ps1
```

### What it does:
1. Compiles the entire application (`app/cli.py`, `app/`, `model/`, `templates/`, `static/`) into a self-contained executable distribution in `dist/ROPTriModalSegmenter/`.
2. Packages the files into a standard Windows Installer (`ROP_TriModal_Diagnostic_Setup_v2.1.0.msi`) with Start Menu shortcuts and automatic PATH registration.
