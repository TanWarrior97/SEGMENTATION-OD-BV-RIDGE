/**
 * Client application logic for EPICS ROP Tri-Modal Diagnostic Platform.
 * Provides clean drag-and-drop inference and multi-structure visualization.
 */

let currentData = null;

document.addEventListener("DOMContentLoaded", () => {
    initDropZone();
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

    showLoading("Analyzing Retinal Image (Ridge, Optic Disc, Blood Vessels, Zones)...");

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

    // Smooth scroll down to analysis results
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
