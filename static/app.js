const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const modeSelect = document.getElementById("chat-mode");

const cameraStatusEl = document.getElementById("camera-status");
const robotStatusEl = document.getElementById("robot-status");
const simStatusEl = document.getElementById("sim-status");
const cameraIdInput = document.getElementById("camera-id-input");
const cameraApplyIdBtn = document.getElementById("camera-apply-id-btn");
const cameraToggleIdBtn = document.getElementById("camera-toggle-id-btn");
const cameraRestartBtn = document.getElementById("camera-restart-btn");

let isSending = false;
let lastCameraError = "";

function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setSending(flag) {
    isSending = flag;
    sendBtn.disabled = flag;
}

async function sendChat(message) {
    const trimmed = message.trim();
    if (!trimmed || isSending) {
        return;
    }

    addMessage("user", trimmed);
    setSending(true);

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: trimmed, mode: modeSelect.value }),
        });

        const data = await response.json();

        if (data.interpreted_command) {
            addMessage("system", `AI interpreted: ${data.interpreted_command}`);
        }

        if (data.reply) {
            addMessage(data.ok ? "bot" : "system", data.reply);
        } else {
            addMessage("system", "No response message from server.");
        }

        if (data.needs_api_key) {
            addMessage(
                "system",
                "AI mode needs OPENAI_API_KEY. Use Local Parser mode if you do not want to use an API key."
            );
        }
    } catch (err) {
        addMessage("system", `Request failed: ${err.message}`);
    } finally {
        setSending(false);
        chatInput.focus();
    }
}

async function refreshHealth() {
    try {
        const response = await fetch("/api/health");
        const data = await response.json();

        const activeIndex = data.camera_active_index;
        const backend = data.camera_backend || "unknown";
        const hasActiveIndex = activeIndex !== null && activeIndex !== undefined;
        const detail = hasActiveIndex ? ` (idx ${activeIndex}, ${backend})` : "";

        cameraStatusEl.textContent = `Camera: ${data.camera_ready ? "online" : "offline"}${detail}`;
        cameraStatusEl.title = data.camera_last_error || "";

        if (
            cameraIdInput &&
            document.activeElement !== cameraIdInput &&
            data.camera_requested_index !== null &&
            data.camera_requested_index !== undefined
        ) {
            cameraIdInput.value = data.camera_requested_index;
        }

        const currentError = data.camera_last_error || "";
        if (!data.camera_ready && currentError && currentError !== lastCameraError) {
            addMessage("system", `Camera issue: ${currentError}`);
            lastCameraError = currentError;
        }
        if (data.camera_ready) {
            lastCameraError = "";
        }

        robotStatusEl.textContent = `Robot: ${data.robot_connected ? "connected" : "disconnected"}`;
        simStatusEl.textContent = `Mode: ${data.simulated_robot ? "simulation" : "hardware"}`;
    } catch (err) {
        cameraStatusEl.textContent = "Camera: unknown";
        cameraStatusEl.title = "";
        robotStatusEl.textContent = "Robot: unknown";
        simStatusEl.textContent = "Mode: unknown";
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

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = chatInput.value;
    chatInput.value = "";
    await sendChat(text);
});

document.querySelectorAll(".chip").forEach((button) => {
    button.addEventListener("click", async () => {
        const cmd = button.getAttribute("data-command") || "";
        await sendChat(cmd);
    });
});

if (cameraRestartBtn) {
    cameraRestartBtn.addEventListener("click", async () => {
        try {
            const response = await fetch("/api/camera/restart", { method: "POST" });
            const data = await response.json();

            if (data.ok) {
                addMessage("system", "Camera restarted successfully.");
            } else {
                const detail = data.camera_last_error ? ` ${data.camera_last_error}` : "";
                addMessage("system", `Camera restart failed.${detail}`);
            }

            await refreshHealth();
        } catch (err) {
            addMessage("system", `Camera restart request failed: ${err.message}`);
        }
    });
}

if (cameraApplyIdBtn && cameraIdInput) {
    cameraApplyIdBtn.addEventListener("click", async () => {
        const nextValue = Number.parseInt(cameraIdInput.value, 10);
        if (Number.isNaN(nextValue) || nextValue < 0) {
            addMessage("system", "Camera ID must be a non-negative integer.");
            return;
        }

        try {
            const data = await setCameraIndex(nextValue);
            if (data.ok) {
                addMessage("system", `Camera ID updated: ${summarizeCameraResult(data)}`);
            } else {
                const detail = data.camera_last_error ? ` ${data.camera_last_error}` : "";
                const reply = data.reply ? ` ${data.reply}` : "";
                addMessage("system", `Failed to set camera ID.${reply}${detail}`);
            }

            await refreshHealth();
        } catch (err) {
            addMessage("system", `Set camera ID request failed: ${err.message}`);
        }
    });
}

if (cameraToggleIdBtn) {
    cameraToggleIdBtn.addEventListener("click", async () => {
        try {
            const data = await toggleCamera("next");
            if (data.ok) {
                addMessage("system", `Camera toggled: ${summarizeCameraResult(data)}`);
            } else {
                const detail = data.camera_last_error ? ` ${data.camera_last_error}` : "";
                const reply = data.reply ? ` ${data.reply}` : "";
                addMessage("system", `Camera toggle failed.${reply}${detail}`);
            }

            await refreshHealth();
        } catch (err) {
            addMessage("system", `Camera toggle request failed: ${err.message}`);
        }
    });
}

addMessage(
    "system",
    "Ready. Example commands: connect, read positions, move base to 2300, open gripper, home, disconnect."
);

refreshHealth();
setInterval(refreshHealth, 4000);