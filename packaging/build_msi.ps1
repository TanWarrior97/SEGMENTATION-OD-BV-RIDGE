<#
.SYNOPSIS
    Automated MSI Package Builder for EPICS ROP Tri-Modal Retinal Diagnostic Suite.
.DESCRIPTION
    1. Compiles the Python application and models into a standalone binary distribution via PyInstaller.
    2. Packages the distribution into a standard Windows Installer (.msi) using WiX Toolset.
#>

param (
    [string]$Configuration = "Release",
    [string]$OutputMsi = "ROP_TriModal_Diagnostic_Setup_v2.1.0.msi"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Resolve-Path "$ScriptDir\.."

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  BUILDING EPICS ROP TRI-MODAL DIAGNOSTIC MSI   " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Step 1: Verify PyInstaller
Write-Host "`n[1/4] Checking PyInstaller installation..." -ForegroundColor Yellow
if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "PyInstaller not found in PATH. Installing via pip..." -ForegroundColor DarkYellow
    python -m pip install pyinstaller
}

# Step 2: Build Executable Distribution
Write-Host "`n[2/4] Running PyInstaller compilation..." -ForegroundColor Yellow
Push-Location $RepoRoot
try {
    pyinstaller --clean -y "$ScriptDir\ROPSegmenter.spec"
} finally {
    Pop-Location
}

if (-not (Test-Path "$RepoRoot\dist\ROPTriModalSegmenter\ROPTriModalSegmenter.exe")) {
    throw "PyInstaller build failed: Executable not found in dist\ROPTriModalSegmenter\"
}
Write-Host "Executable built successfully: dist\ROPTriModalSegmenter\ROPTriModalSegmenter.exe" -ForegroundColor Green

# Step 3: Verify WiX Toolset
Write-Host "`n[3/4] Checking WiX Toolset..." -ForegroundColor Yellow
$Candle = Get-Command candle.exe -ErrorAction SilentlyContinue
$Light = Get-Command light.exe -ErrorAction SilentlyContinue
$WixV4 = Get-Command wix.exe -ErrorAction SilentlyContinue

if ($WixV4) {
    Write-Host "Found WiX v4+: Building MSI using 'wix build'..." -ForegroundColor Green
    Push-Location $RepoRoot
    try {
        wix build "$ScriptDir\Product.wxs" -o "$RepoRoot\$OutputMsi" -ext WixToolset.UI.wixext
    } finally {
        Pop-Location
    }
} elseif ($Candle -and $Light) {
    Write-Host "Found WiX v3: Compiling candle and light..." -ForegroundColor Green
    Push-Location $RepoRoot
    try {
        & $Candle.Source "$ScriptDir\Product.wxs" -o "$ScriptDir\Product.wixobj"
        & $Light.Source -ext WixUIExtension "$ScriptDir\Product.wixobj" -o "$RepoRoot\$OutputMsi"
    } finally {
        Pop-Location
    }
} else {
    Write-Host "`n[NOTE] WiX Toolset (candle/light or wix.exe) is not detected in your system PATH." -ForegroundColor DarkYellow
    Write-Host "The standalone distribution is ready in: $RepoRoot\dist\ROPTriModalSegmenter\" -ForegroundColor Cyan
    Write-Host "To generate the final MSI installer:" -ForegroundColor Cyan
    Write-Host "  1. Install WiX Toolset (https://wixtoolset.org/releases/ or 'winget install WiX')" -ForegroundColor White
    Write-Host "  2. Re-run .\packaging\build_msi.ps1" -ForegroundColor White
    return
}

if (Test-Path "$RepoRoot\$OutputMsi") {
    Write-Host "`n================================================" -ForegroundColor Green
    Write-Host " SUCCESS: MSI Installer successfully created!" -ForegroundColor Green
    Write-Host " File: $RepoRoot\$OutputMsi" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
}
