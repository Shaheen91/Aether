(function () {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const dropzonePreview = document.getElementById("dropzonePreview");
  const previewImg = document.getElementById("previewImg");
  const clearBtn = document.getElementById("clearBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const errorBox = document.getElementById("errorBox");

  const stateIdle = document.getElementById("stateIdle");
  const stateLoading = document.getElementById("stateLoading");
  const stateResult = document.getElementById("stateResult");
  const loadingText = document.getElementById("loadingText");

  const resultBadge = document.getElementById("resultBadge");
  const resultTitle = document.getElementById("resultTitle");
  const ringFill = document.getElementById("ringFill");
  const ringLabel = document.getElementById("ringLabel");
  const probBars = document.getElementById("probBars");
  const explanationContent = document.getElementById("explanationContent");

  const historyGrid = document.getElementById("historyGrid");
  const historyEmpty = document.getElementById("historyEmpty");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  const RING_CIRCUMFERENCE = 214; // 2 * PI * 34
  const HISTORY_KEY = "verdant_ai_history";
  const LOADING_MESSAGES = [
    "Running EfficientNet-B3 inference…",
    "Scoring disease signatures…",
    "Consulting the AI pathologist…",
  ];

  let selectedFile = null;

  // ── Helpers ──────────────────────────────────────────────
  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("d-none");
  }

  function hideError() {
    errorBox.classList.add("d-none");
    errorBox.textContent = "";
  }

  function setState(name) {
    stateIdle.classList.toggle("d-none", name !== "idle");
    stateLoading.classList.toggle("d-none", name !== "loading");
    stateResult.classList.toggle("d-none", name !== "result");
  }

  function formatLabel(label) {
    return label.replace(/_/g, " ");
  }

  // Minimal, safe markdown-ish renderer for the LLM explanation:
  // supports **bold** headers and numbered/bulleted lines.
  function renderExplanation(text) {
    const escaped = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    const withBold = escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    const withBreaks = withBold
      .split(/\n{2,}/)
      .map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`)
      .join("");

    return withBreaks;
  }

  // ── File selection ──────────────────────────────────────
  function handleFile(file) {
    if (!file) return;
    const validTypes = ["image/png", "image/jpeg", "image/webp"];
    if (!validTypes.includes(file.type)) {
      showError("Unsupported file type. Please use PNG, JPG, or WEBP.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError("File is too large. Max size is 10MB.");
      return;
    }
    hideError();
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropzoneEmpty.classList.add("d-none");
      dropzonePreview.classList.remove("d-none");
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  browseBtn.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", (e) => {
    if (dropzonePreview.classList.contains("d-none")) fileInput.click();
  });
  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("vg-dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("vg-dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    handleFile(file);
  });

  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    dropzonePreview.classList.add("d-none");
    dropzoneEmpty.classList.remove("d-none");
    analyzeBtn.disabled = true;
  });

  // ── Analyze ──────────────────────────────────────────────
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    hideError();
    setState("loading");
    analyzeBtn.disabled = true;

    let msgIndex = 0;
    loadingText.textContent = LOADING_MESSAGES[0];
    const loadingInterval = setInterval(() => {
      msgIndex = (msgIndex + 1) % LOADING_MESSAGES.length;
      loadingText.textContent = LOADING_MESSAGES[msgIndex];
    }, 1400);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      clearInterval(loadingInterval);

      if (!response.ok || !data.success) {
        setState("idle");
        showError(data.error || "Something went wrong. Please try again.");
        analyzeBtn.disabled = false;
        return;
      }

      renderResult(data);
      addToHistory(data);
      setState("result");
    } catch (err) {
      clearInterval(loadingInterval);
      setState("idle");
      showError("Could not reach the server. Please check your connection and try again.");
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  // ── Render result ────────────────────────────────────────
  function renderResult(data) {
    const isHealthy = data.is_healthy;
    const topDetected = data.detected_diseases[0];
    const topPrediction = data.all_predictions[0];

    resultBadge.textContent = isHealthy ? "Healthy" : "Disease detected";
    resultBadge.className = "vg-badge " + (isHealthy ? "vg-badge--healthy" : "vg-badge--disease");

    resultTitle.textContent = isHealthy
      ? "No disease detected"
      : data.detected_diseases.map(formatLabel).join(" + ");

    const topConfidence = isHealthy
      ? topPrediction.confidence
      : data.all_predictions.find((p) => p.label === topDetected).confidence;
    const pct = Math.round(topConfidence * 100);

    ringFill.style.stroke = isHealthy ? "var(--green)" : "var(--amber)";
    ringFill.style.strokeDashoffset = RING_CIRCUMFERENCE - (RING_CIRCUMFERENCE * pct) / 100;
    ringLabel.textContent = pct + "%";

    probBars.innerHTML = data.all_predictions
      .map((p) => {
        const barPct = Math.round(p.confidence * 100);
        const isLow = p.confidence <= 0.5;
        return `
          <div class="vg-prob-row">
            <span class="vg-prob-label">${formatLabel(p.label)}</span>
            <div class="vg-prob-track">
              <div class="vg-prob-fill ${isLow ? "vg-prob-fill--low" : ""}" style="width:${barPct}%"></div>
            </div>
            <span class="vg-prob-value">${barPct}%</span>
          </div>`;
      })
      .join("");

    if (data.explanation) {
      explanationContent.innerHTML = renderExplanation(data.explanation);
    } else {
      explanationContent.innerHTML = `<p class="vg-muted">${
        data.llm_error
          ? "AI explanation unavailable: " + data.llm_error
          : "AI explanation unavailable for this request."
      }</p>`;
    }
  }

  // ── History (client-side, persisted in localStorage) ────
  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function saveHistory(items) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 12)));
  }

  function addToHistory(data) {
    const items = loadHistory();
    const top = data.all_predictions[0];
    items.unshift({
      image_url: data.image_url,
      label: data.is_healthy ? "healthy" : data.detected_diseases.join(", "),
      confidence: top.confidence,
      is_healthy: data.is_healthy,
      timestamp: Date.now(),
    });
    saveHistory(items);
    renderHistory();
  }

  function renderHistory() {
    const items = loadHistory();
    historyEmpty.classList.toggle("d-none", items.length > 0);
    historyGrid.innerHTML = items
      .map(
        (item) => `
        <div class="vg-history-item">
          <img src="${item.image_url}" alt="${formatLabel(item.label)}" loading="lazy">
          <div class="vg-history-meta">
            <div class="label" style="color: ${item.is_healthy ? "var(--green)" : "var(--amber)"}">
              ${formatLabel(item.label)}
            </div>
            <div class="conf">${Math.round(item.confidence * 100)}%</div>
          </div>
        </div>`
      )
      .join("");
  }

  clearHistoryBtn.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });

  renderHistory();
})();
