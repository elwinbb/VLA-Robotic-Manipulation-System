const frontendConfig = window.FRONTEND_CONFIG || {};
const promptCatalog = frontendConfig.promptCatalog || {};

const cameraStatusEl = document.getElementById("camera-status");
const robotStatusEl = document.getElementById("robot-status");
const simStatusEl = document.getElementById("sim-status");
const geminiStatusEl = document.getElementById("gemini-status");
const visionStatusEl = document.getElementById("vision-status");
const modelNameEl = document.getElementById("model-name");
const executionHelperEl = document.getElementById("execution-helper");

const cameraRestartBtn = document.getElementById("camera-restart-btn");

const promptForm = document.getElementById("prompt-form");
const promptTypeEl = document.getElementById("prompt-type");
const promptInputEl = document.getElementById("prompt-input");
const promptHelperEl = document.getElementById("prompt-helper");
const executeRobotEl = document.getElementById("execute-robot");
const runPromptBtn = document.getElementById("run-prompt-btn");
const useDefaultBtn = document.getElementById("use-default-btn");
const clearOutputBtn = document.getElementById("clear-output-btn");

const analysisOutputEl = document.getElementById("analysis-output");
const analysisSummaryEl = document.getElementById("analysis-summary");
const detectionListEl = document.getElementById("detection-list");
const streamOutputEl = document.getElementById("stream-output");
const jsonOutputEl = document.getElementById("json-output");

let lastCameraError = "";
let lastOverlayVersion = null;
let lastPromptType = frontendConfig.defaultPromptType || Object.keys(promptCatalog)[0] || "description";

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function defaultPromptFor(type) {
    return promptCatalog[type]?.default_prompt || "";
}

function helperTextFor(type) {
    return promptCatalog[type]?.helper || "No helper text available.";
}

function supportsExecution(type) {
    return Boolean(promptCatalog[type]?.supports_execution);
}

function populatePromptTypes() {
    const entries = Object.entries(promptCatalog);
    promptTypeEl.innerHTML = "";

    for (const [value, meta] of entries) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = meta.label || value;
        promptTypeEl.appendChild(option);
    }

    if (entries.length > 0) {
        promptTypeEl.value = lastPromptType in promptCatalog ? lastPromptType : entries[0][0];
    }

    updatePromptMeta();
}

function updatePromptMeta() {
    const selected = promptTypeEl.value;
    promptHelperEl.textContent = helperTextFor(selected);
    if (executionHelperEl) {
        executionHelperEl.textContent = supportsExecution(selected)
            ? "Preview runs first. If execution is enabled, the robot follows the notebook calibration and motion sequence."
            : "This prompt type is preview-only. The notebook pipeline will not move the robot.";
    }
    if (executeRobotEl) {
        executeRobotEl.disabled = !supportsExecution(selected);
        if (!supportsExecution(selected)) {
            executeRobotEl.checked = false;
        }
    }
}

function applyDefaultPrompt(force = false) {
    const selected = promptTypeEl.value;
    const nextDefault = defaultPromptFor(selected);
    const previousDefault = defaultPromptFor(lastPromptType);
    const currentValue = promptInputEl.value.trim();

    if (force || !currentValue || currentValue === previousDefault) {
        promptInputEl.value = nextDefault;
    }

    lastPromptType = selected;
    updatePromptMeta();
}

function formatPoint(point) {
    if (!point || point.X === undefined || point.Y === undefined) {
        return "n/a";
    }
    return `X ${point.X}, Y ${point.Y}`;
}

function renderDetections(plan) {
    const detections = plan?.detections || [];
    if (!detections.length) {
        detectionListEl.innerHTML = '<div class="empty-card">No detections available for the current run.</div>';
        return;
    }

    detectionListEl.innerHTML = detections
        .map((item, index) => {
            const bbox = Array.isArray(item.bounding_box) ? item.bounding_box.join(", ") : "n/a";
            return `
                <article class="detection-card">
                    <div class="detection-index">${index + 1}</div>
                    <div class="detection-body">
                        <h3>${escapeHtml(item.label || "object")}</h3>
                        <p>Center: ${escapeHtml(formatPoint(item.center))}</p>
                        <p>Box: ${escapeHtml(bbox)}</p>
                    </div>
                </article>
            `;
        })
        .join("");
}

function renderSummary(state) {
    const plan = state.plan;
    if (!plan) {
        analysisSummaryEl.textContent =
            state.status === "error"
                ? "The last run failed. Check the stream output for the error trace."
                : "Run a prompt to populate detections, source and destination points, and the structured plan.";
        return;
    }

    const bits = [
        `<strong>Task:</strong> ${escapeHtml((plan.task || "unknown").replaceAll("_", " "))}`,
        `<strong>Detections:</strong> ${escapeHtml((plan.detections || []).length)}`,
        `<strong>Robot execution:</strong> ${state.execute_robot ? "enabled" : "disabled"}`,
    ];

    if (plan.source) {
        bits.push(`<strong>Source:</strong> ${escapeHtml(formatPoint(plan.source))}`);
    }
    if (plan.destination) {
        bits.push(`<strong>Destination:</strong> ${escapeHtml(formatPoint(plan.destination))}`);
    }
    if (plan.trajectory?.length) {
        bits.push(`<strong>Trajectory:</strong> ${escapeHtml(plan.trajectory.length)} points`);
    }
    if (plan.description) {
        bits.push(`<strong>Description:</strong> ${escapeHtml(plan.description)}`);
    }
    if (plan.notes) {
        bits.push(`<strong>Notes:</strong> ${escapeHtml(plan.notes)}`);
    }

    analysisSummaryEl.innerHTML = bits.join("<br>");
}

function renderLogs(logs) {
    if (!logs || !logs.length) {
        streamOutputEl.textContent = "Waiting for a prompt run...";
        return;
    }

    streamOutputEl.textContent = logs
        .map((item) => `[${item.ts}] ${item.level.toUpperCase()}  ${item.message}`)
        .join("\n");
}

function updateAnalysisImage(state) {
    const overlayAvailable = Boolean(state.overlay_available);
    const overlayVersion = overlayAvailable ? state.overlay_version : `placeholder-${state.status}`;
    if (overlayVersion === lastOverlayVersion) {
        return;
    }

    const cacheBust = `t=${Date.now()}`;
    analysisOutputEl.src = overlayAvailable
        ? `/api/vision/overlay?run=${encodeURIComponent(state.overlay_version)}&${cacheBust}`
        : `/api/vision/overlay?placeholder=1&${cacheBust}`;
    lastOverlayVersion = overlayVersion;
}

function renderVisionState(state) {
    const busy = Boolean(state.busy);
    runPromptBtn.disabled = busy;
    runPromptBtn.textContent = busy ? "Running..." : "Run prompt";

    const phase = state.phase && state.phase !== state.status ? ` / ${state.phase.replaceAll("_", " ")}` : "";
    visionStatusEl.textContent = `Prompt run: ${state.status || "idle"}${phase}`;
    modelNameEl.textContent = state.model || "Unknown model";

    renderSummary(state);
    renderDetections(state.plan);
    renderLogs(state.logs);
    updateAnalysisImage(state);

    if (state.plan) {
        jsonOutputEl.textContent = JSON.stringify(state.plan, null, 2);
    } else if (state.raw_response) {
        jsonOutputEl.textContent = state.raw_response;
    } else {
        jsonOutputEl.textContent = "No plan generated yet.";
    }
}

async function refreshVisionState() {
    try {
        const response = await fetch("/api/vision/state");
        const state = await response.json();
        renderVisionState(state);
    } catch (error) {
        streamOutputEl.textContent = `Failed to fetch prompt state: ${error.message}`;
    }
}

async function refreshHealth() {
    try {
        const response = await fetch("/api/health");
        const data = await response.json();

        const activeIndex = data.camera_active_index;
        const backend = data.camera_backend || "unknown";
        const hasActiveIndex = activeIndex !== null && activeIndex !== undefined;
        const detail = hasActiveIndex ? ` (${activeIndex}, ${backend})` : "";

        cameraStatusEl.textContent = `Camera: ${data.camera_ready ? "online" : "offline"}${detail}`;
        cameraStatusEl.title = data.camera_last_error || "";

        const currentError = data.camera_last_error || "";
        if (!data.camera_ready && currentError && currentError !== lastCameraError) {
            lastCameraError = currentError;
        }
        if (data.camera_ready) {
            lastCameraError = "";
        }

        robotStatusEl.textContent = data.robot_sdk_available
            ? (data.vision_status === "running" ? "Robot: notebook run active" : "Robot: SDK ready")
            : "Robot: SDK missing";
        simStatusEl.textContent = data.realsense_installed ? "Pipeline: RealSense" : "Pipeline: missing";
        geminiStatusEl.textContent = `Gemini: ${data.gemini_installed ? (data.gemini_api_key_present ? "ready" : "missing key") : "package missing"}`;
    } catch (error) {
        cameraStatusEl.textContent = "Camera: unknown";
        robotStatusEl.textContent = "Robot: unknown";
        simStatusEl.textContent = "Pipeline: unknown";
        geminiStatusEl.textContent = "Gemini: unknown";
    }
}

function summarizeCameraResult(data) {
    const requested = data.camera_requested_index;
    const active = data.camera_active_index;
    const backend = data.camera_backend || "unknown";
    const activeText = active === null || active === undefined ? "none" : `${active}`;
    return `requested=${requested}, active=${activeText}, backend=${backend}`;
}

async function setCameraIndex(cameraIndex) {
    const response = await fetch("/api/camera/set_index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_index: cameraIndex }),
    });
    return response.json();
}

async function toggleCamera(direction = "next") {
    const response = await fetch("/api/camera/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ direction }),
    });
    return response.json();
}

async function runPromptAnalysis() {
    const promptType = promptTypeEl.value;
    const prompt = promptInputEl.value.trim();
    const executeRobot = Boolean(executeRobotEl?.checked && supportsExecution(promptType));

    if (!prompt) {
        promptInputEl.value = defaultPromptFor(promptType);
    }

    const response = await fetch("/api/vision/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            prompt_type: promptType,
            prompt: promptInputEl.value.trim(),
            execute_robot: executeRobot,
        }),
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
        throw new Error(data.reply || "Prompt run could not be started.");
    }

    renderVisionState(data);
}

async function clearVisionOutput() {
    const response = await fetch("/api/vision/clear", { method: "POST" });
    const data = await response.json();
    renderVisionState(data);
}

promptTypeEl.addEventListener("change", () => applyDefaultPrompt(false));

promptForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    runPromptBtn.disabled = true;
    runPromptBtn.textContent = "Running...";

    try {
        await runPromptAnalysis();
    } catch (error) {
        streamOutputEl.textContent = `[${new Date().toLocaleTimeString()}] ERROR  ${error.message}`;
        runPromptBtn.disabled = false;
        runPromptBtn.textContent = "Run prompt";
    }
});

useDefaultBtn.addEventListener("click", () => applyDefaultPrompt(true));
clearOutputBtn.addEventListener("click", async () => {
    await clearVisionOutput();
});

if (cameraRestartBtn) {
    cameraRestartBtn.addEventListener("click", async () => {
        try {
            const response = await fetch("/api/camera/restart", { method: "POST" });
            const data = await response.json();
            streamOutputEl.textContent = data.ok
                ? `Camera restarted successfully.\n${summarizeCameraResult(data)}`
                : `Camera restart failed.\n${data.camera_last_error || ""}`;
            await refreshHealth();
        } catch (error) {
            streamOutputEl.textContent = `Camera restart request failed: ${error.message}`;
        }
    });
}

populatePromptTypes();
applyDefaultPrompt(true);
refreshHealth();
refreshVisionState();
setInterval(refreshHealth, 4000);
setInterval(refreshVisionState, 1500);
