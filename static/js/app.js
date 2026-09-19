/**
 * Client application logic for EPICS ROP Tri-Modal Diagnostic Platform.
 * Supports dynamic layer blending, real-time opacity manipulation,
 * benchmark catalog browsing, and side-by-side ground truth comparisons.
 */

let currentData = null;
let currentOriginalImg = null;
let currentOverlays = {};
let currentMasks = {};

document.addEventListener("DOMContentLoaded", () => {
    initDropZone();
    initBenchmarkCatalog();
    initLayerControls();
});

// ============================================================================
// File Upload & Drag-and-Drop
// ============================================================================

function initDropZone() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}

function showLoading(text = "Running Tri-Modal Deep Learning Segmentation...") {
    const overlay = document.getElementById("loadingOverlay");
    const textEl = document.getElementById("loadingText");
    textEl.innerText = text;
    overlay.style.display = "flex";
}

function hideLoading() {
    document.getElementById("loadingOverlay").style.display = "none";
}

async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("image", file);

    showLoading("Analyzing Retinal Image with MAnet + UNet++ Models...");

    try {
        const resp = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });
        const data = await resp.json();
        if (data.status === "SUCCESS") {
            renderResults(data);
        } else {
            alert("Analysis error: " + (data.error || "Unknown error"));
        }
    } catch (err) {
        alert("Network or inference error: " + err.message);
    } finally {
        hideLoading();
    }
}

// ============================================================================
// Benchmark Catalog Browsing
// ============================================================================

async function initBenchmarkCatalog() {
    try {
        const resp = await fetch("/api/benchmark_images?target=all");
        const data = await resp.json();
        if (data.status === "SUCCESS" && data.catalog) {
            renderBenchmarkChips(data.catalog);
        }
    } catch (e) {
        console.warn("Could not load benchmark gallery:", e);
    }
}

function renderBenchmarkChips(catalog) {
    const container = document.getElementById("galleryChips");
    container.innerHTML = "";

    // Display first 18 samples across Ridge, OD, and BV
    catalog.slice(0, 18).forEach(item => {
        const btn = document.createElement("button");
        btn.className = "chip-btn";
        btn.innerText = `${item.camera} #${item.filename.split('.')[0]}`;
        btn.title = item.display_name;
        btn.addEventListener("click", () => loadBenchmarkCase(item.id, item.target));
        container.appendChild(btn);
    });
}

async function loadBenchmarkCase(id, target) {
    showLoading("Fetching Benchmark Retinal Image & Ground Truth...");
    try {
        const resp = await fetch(`/api/load_benchmark?id=${id}&target=${target}`);
        const data = await resp.json();
        if (data.status === "SUCCESS") {
            renderResults(data);
        } else {
            alert("Error loading benchmark: " + data.error);
        }
    } catch (e) {
        alert("Failed to load benchmark: " + e.message);
    } finally {
        hideLoading();
    }
}

// ============================================================================
// Render Results into UI
// ============================================================================

function renderResults(data) {
    currentData = data;
    document.getElementById("resultsSection").style.display = "flex";

    // 1. Set Original and Combined Master Images
    const origImg = document.getElementById("originalImage");
    const combinedImg = document.getElementById("combinedImage");

    origImg.src = `data:image/png;base64,${data.original_base64}`;
    combinedImg.src = `data:image/png;base64,${data.combined_overlay_base64}`;

    // 2. Populate Metrics Summary Header
    const m = data.metrics;
    document.getElementById("metricDims").innerText = `${m.dimensions.width} x ${m.dimensions.height} px`;
    document.getElementById("metricTime").innerText = `${m.inference_time_sec}s`;
    document.getElementById("metricDevice").innerText = m.device;

    // Benchmark match notice if any
    const matchNotice = document.getElementById("benchmarkMatchNotice");
    if (data.matched_benchmark || data.benchmark_name) {
        matchNotice.style.display = "block";
        document.getElementById("matchedName").innerText = data.matched_benchmark || data.benchmark_name;
    } else {
        matchNotice.style.display = "none";
    }

    // 3. Populate Demarcation Ridge Card
    const ridgeCard = document.getElementById("ridgeStatus");
    const ridgeImg = document.getElementById("ridgeCardImg");
    ridgeImg.src = `data:image/png;base64,${data.ridge_overlay_base64}`;
    if (m.ridge_detected) {
        ridgeCard.innerText = "RIDGE DETECTED";
        ridgeCard.className = "card-status detected";
    } else {
        ridgeCard.innerText = "NO RIDGE";
        ridgeCard.className = "card-status not-detected";
    }
    document.getElementById("ridgePixels").innerText = `${m.ridge.ridge_pixels.toLocaleString()} px`;
    document.getElementById("ridgeCoverage").innerText = `${m.ridge.coverage_percent}%`;
    document.getElementById("ridgeConf").innerText = `${(m.ridge.mean_confidence * 100).toFixed(1)}%`;

    // Download links
    setupDownloadBtn("downloadRidgeOverlay", data.ridge_overlay_base64, "ridge_overlay.png");
    setupDownloadBtn("downloadRidgeMask", data.ridge_mask_base64, "ridge_mask.png");

    // 4. Populate Optic Disc Card
    const odCard = document.getElementById("odStatus");
    const odImg = document.getElementById("odCardImg");
    odImg.src = `data:image/png;base64,${data.od_overlay_base64}`;
    if (m.od_detected) {
        odCard.innerText = "OD DETECTED";
        odCard.className = "card-status detected";
    } else {
        odCard.innerText = "NO OD DETECTED";
        odCard.className = "card-status not-detected";
    }
    document.getElementById("odPixels").innerText = `${m.optic_disc.od_pixels.toLocaleString()} px`;
    if (m.optic_disc.center) {
        document.getElementById("odCenter").innerText = `(${m.optic_disc.center[0]}, ${m.optic_disc.center[1]})`;
        document.getElementById("odDiameter").innerText = `${m.optic_disc.diameter} px`;
    } else {
        document.getElementById("odCenter").innerText = "N/A";
        document.getElementById("odDiameter").innerText = "N/A";
    }
    setupDownloadBtn("downloadOdOverlay", data.od_overlay_base64, "optic_disc_overlay.png");
    setupDownloadBtn("downloadOdMask", data.od_mask_base64, "optic_disc_mask.png");

    // 5. Populate Blood Vessels Card
    const bvCard = document.getElementById("bvStatus");
    const bvImg = document.getElementById("bvCardImg");
    bvImg.src = `data:image/png;base64,${data.bv_overlay_base64}`;
    if (m.bv_detected) {
        bvCard.innerText = "VESSELS DETECTED";
        bvCard.className = "card-status detected";
    } else {
        bvCard.innerText = "NO VESSELS";
        bvCard.className = "card-status not-detected";
    }
    document.getElementById("bvPixels").innerText = `${m.blood_vessels.vessel_pixels.toLocaleString()} px`;
    document.getElementById("bvDensity").innerText = `${m.blood_vessels.vessel_density_percent}%`;
    setupDownloadBtn("downloadBvOverlay", data.bv_overlay_base64, "blood_vessels_overlay.png");
    setupDownloadBtn("downloadBvMask", data.bv_mask_base64, "blood_vessels_mask.png");

    // 6. Populate Zones Card
    const zonesImg = document.getElementById("zonesCardImg");
    zonesImg.src = `data:image/png;base64,${data.zones_overlay_base64}`;
    if (m.zones) {
        document.getElementById("zoneMaculaDist").innerText = `${m.zones.macula_distance} px`;
        document.getElementById("zone1Radius").innerText = `${m.zones.zone1_radius} px`;
        document.getElementById("zone2Radius").innerText = `${m.zones.zone2_radius} px`;
    }

    // 7. Ground Truth Panel if available
    const gtPanel = document.getElementById("gtPanel");
    if (data.has_gt && data.gt_base64) {
        gtPanel.style.display = "block";
        document.getElementById("gtImage").src = `data:image/png;base64,${data.gt_base64}`;
        const aiVal = document.getElementById("aiValidationImage");
        if (aiVal) {
            aiVal.src = `data:image/png;base64,${data.combined_overlay_base64}`;
        }
    } else {
        gtPanel.style.display = "none";
    }

    // Scroll to results
    document.getElementById("resultsSection").scrollIntoView({ behavior: "smooth" });
}

function setupDownloadBtn(elementId, base64Data, filename) {
    const btn = document.getElementById(elementId);
    if (!btn) return;
    btn.onclick = (e) => {
        e.preventDefault();
        const a = document.createElement("a");
        a.href = `data:image/png;base64,${base64Data}`;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };
}

// ============================================================================
// Interactive Layer Visibility Toggles
// ============================================================================

function initLayerControls() {
    const ridgeToggle = document.getElementById("toggleRidge");
    const odToggle = document.getElementById("toggleOd");
    const bvToggle = document.getElementById("toggleBv");
    const zonesToggle = document.getElementById("toggleZones");
    const opacitySlider = document.getElementById("opacitySlider");
    const opacityVal = document.getElementById("opacityVal");

    [ridgeToggle, odToggle, bvToggle, zonesToggle].forEach(toggle => {
        if (toggle) {
            toggle.addEventListener("change", updateMasterView);
        }
    });

    if (opacitySlider) {
        opacitySlider.addEventListener("input", (e) => {
            const val = e.target.value;
            opacityVal.innerText = `${val}%`;
            const combinedImg = document.getElementById("combinedImage");
            if (combinedImg) {
                combinedImg.style.opacity = val / 100;
            }
        });
    }
}

function updateMasterView() {
    if (!currentData) return;

    const showRidge = document.getElementById("toggleRidge").checked;
    const showOd = document.getElementById("toggleOd").checked;
    const showBv = document.getElementById("toggleBv").checked;
    const showZones = document.getElementById("toggleZones").checked;

    // If all are checked, show full combined overlay
    if (showRidge && showOd && showBv && showZones) {
        document.getElementById("combinedImage").src = `data:image/png;base64,${currentData.combined_overlay_base64}`;
        return;
    }

    // If only one is selected, display that modality's individual overlay
    if (showRidge && !showOd && !showBv && !showZones) {
        document.getElementById("combinedImage").src = `data:image/png;base64,${currentData.ridge_overlay_base64}`;
        return;
    }
    if (!showRidge && showOd && !showBv && !showZones) {
        document.getElementById("combinedImage").src = `data:image/png;base64,${currentData.od_overlay_base64}`;
        return;
    }
    if (!showRidge && !showOd && showBv && !showZones) {
        document.getElementById("combinedImage").src = `data:image/png;base64,${currentData.bv_overlay_base64}`;
        return;
    }
    if (!showRidge && !showOd && !showBv && showZones) {
        document.getElementById("combinedImage").src = `data:image/png;base64,${currentData.zones_overlay_base64}`;
        return;
    }

    if (!showRidge && !showOd && !showBv && !showZones) {
        document.getElementById("combinedImage").src = `data:image/png;base64,${currentData.original_base64}`;
        return;
    }

    // Default fallback
    document.getElementById("combinedImage").src = `data:image/png;base64,${currentData.combined_overlay_base64}`;
}
