// frontend/script.js

const API_URL = "http://127.0.0.1:5000";

let currentDebugImage = null;
let currentOriginalImage = null; // To store the object URL or base64
let isDebugView = false;

document.addEventListener("DOMContentLoaded", () => {
    fetchHistory();
});

// ==========================================
// HISTORY
// ==========================================
function toggleHistory() {
    const sidebar = document.getElementById("historySidebar");
    const overlay = document.getElementById("sidebarOverlay");
    sidebar.classList.toggle("open");
    overlay.classList.toggle("open");
}

async function fetchHistory() {
    try {
        const res = await fetch(`${API_URL}/history`);
        if (!res.ok) throw new Error("Failed to load history");
        const history = await res.json();
        renderHistory(history);
    } catch (err) {
        console.error(err);
    }
}

function renderHistory(history) {
    const list = document.getElementById("historyList");
    if (!history || history.length === 0) {
        list.innerHTML = '<div class="history-empty">No scans yet.</div>';
        return;
    }

    list.innerHTML = history.map(item => `
        <div class="history-card">
            <div class="h-header">
                <strong>${item.animal}</strong>
                <span class="h-conf">${item.confidence}%</span>
            </div>
            <div class="h-meta">
                <span>${item.age_group}</span> • 
                <span>${new Date(item.timestamp).toLocaleString(undefined, {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    })}</span>
            </div>
        </div>
    `).join('');
}

// ==========================================
// UPLOAD & PREVIEW
// ==========================================
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById("imageInput").files = files;
        previewImage({ target: { files: files } });
    }
}

function previewImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Reset state
    resetUI(false);

    // Create object URL for preview
    currentOriginalImage = URL.createObjectURL(file);
    const preview = document.getElementById("preview");
    preview.src = currentOriginalImage;

    const container = document.getElementById("imagePreviewContainer");
    container.style.display = "block";

    // Enable button
    document.getElementById("predictBtn").disabled = false;
}

// ==========================================
// API CALL
// ==========================================
async function uploadImage() {
    const input = document.getElementById("imageInput");
    const file = input.files[0];
    if (!file) return;

    // UI Loading State
    const btn = document.getElementById("predictBtn");
    const loader = document.getElementById("loader");
    const errorDiv = document.getElementById("error");

    btn.disabled = true;
    loader.style.display = "flex";
    errorDiv.style.display = "none";
    document.getElementById("result").style.display = "none";
    document.getElementById("viewToggle").style.display = "none";
    isDebugView = false;
    document.getElementById("debugToggle").checked = false;

    // Prepare form data
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        displayResult(data);

        // Refresh history
        fetchHistory();

    } catch (err) {
        showError(err.message);
    } finally {
        loader.style.display = "none";
        btn.disabled = false;
    }
}

function displayResult(data) {
    const resultSection = document.getElementById("result");
    resultSection.style.display = "block";

    // Text Data
    document.getElementById("animalName").textContent = data.animal;
    document.getElementById("confidenceValue").textContent = data.confidence + "%";
    document.getElementById("ageValue").textContent = data.estimated_age_group;
    document.getElementById("wildnessValue").textContent = data.wildness_level;
    document.getElementById("riskValue").textContent = data.human_risk_level;

    // Bar Animation
    const fill = document.getElementById("confidenceFill");
    fill.style.width = "0%";

    // Color logic
    fill.className = "confidence-bar-fill"; // reset
    if (data.confidence > 80) fill.classList.add("high");
    else if (data.confidence > 50) fill.classList.add("medium");
    else fill.classList.add("low");

    setTimeout(() => {
        fill.style.width = data.confidence + "%";
    }, 100);

    // Debug Image Handling
    if (data.debug_image) {
        currentDebugImage = data.debug_image;
        document.getElementById("debugPreview").src = currentDebugImage;
        document.getElementById("viewToggle").style.display = "flex";
    }

    // Smooth scroll
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// ==========================================
// VIEW TOGGLE
// ==========================================
function toggleDebugView() {
    const toggle = document.getElementById("debugToggle");
    const preview = document.getElementById("preview");
    const debugPreview = document.getElementById("debugPreview");

    if (toggle.checked) {
        preview.style.display = "none";
        debugPreview.style.display = "block";
    } else {
        preview.style.display = "block";
        debugPreview.style.display = "none";
    }
}

// ==========================================
// UTILS
// ==========================================
function resetUI(clearFile = true) {
    document.getElementById("result").style.display = "none";
    document.getElementById("error").style.display = "none";
    document.getElementById("viewToggle").style.display = "none";
    document.getElementById("debugPreview").src = "";
    document.getElementById("debugToggle").checked = false;

    if (clearFile) {
        document.getElementById("imageInput").value = "";
        document.getElementById("imagePreviewContainer").style.display = "none";
        document.getElementById("predictBtn").disabled = true;
        currentOriginalImage = null;
    }

    currentDebugImage = null;
}

function showError(msg) {
    const errDiv = document.getElementById("error");
    errDiv.textContent = msg;
    errDiv.style.display = "block";
}

// Modal functions
function openHowModal(e) {
    e.preventDefault();
    document.getElementById("howModal").style.display = "flex";
    document.getElementById("howModal").setAttribute("aria-hidden", "false");
}

function closeHowModal() {
    document.getElementById("howModal").style.display = "none";
    document.getElementById("howModal").setAttribute("aria-hidden", "true");
}

// Close modal on outside click
window.onclick = function (event) {
    const modal = document.getElementById("howModal");
    if (event.target == modal) {
        closeHowModal();
    }
}
