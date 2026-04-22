import atexit
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
from flask import Flask, Response, jsonify, render_template, request

from robot_adapter import RobotCommandRouter
from vision_service import GeminiVisionService

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


class CameraStream:
    """Continuously captures frames from a camera device for MJPEG streaming."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._capture = None
        self._thread = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame: Optional[bytes] = None
        self._camera_ready = False
        self._active_index: Optional[int] = None
        self._backend_name = "none"
        self._last_error = "Camera not initialized."

    @property
    def active_index(self) -> Optional[int]:
        return self._active_index

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def last_error(self) -> str:
        return self._last_error

    def _candidate_indices(self) -> List[int]:
        # Example: CAMERA_FALLBACK_INDICES="0,1,2"
        fallback_raw = os.getenv("CAMERA_FALLBACK_INDICES", "0,1,2")
        fallback_values: List[int] = []
        for token in fallback_raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                fallback_values.append(int(token))
            except ValueError:
                continue

        unique: List[int] = []
        for idx in [self.camera_index, *fallback_values]:
            if idx not in unique:
                unique.append(idx)
        return unique

    def _backend_candidates(self) -> List[Tuple[int, str]]:
        if os.name == "nt":
            backends: List[Tuple[int, str]] = []
            if hasattr(cv2, "CAP_DSHOW"):
                backends.append((cv2.CAP_DSHOW, "CAP_DSHOW"))
            if hasattr(cv2, "CAP_MSMF"):
                backends.append((cv2.CAP_MSMF, "CAP_MSMF"))
            backends.append((cv2.CAP_ANY, "CAP_ANY"))
            return backends
        return [(cv2.CAP_ANY, "CAP_ANY")]

    def _open_capture(self, index: int, backend_code: int) -> Optional[cv2.VideoCapture]:
        try:
            capture = cv2.VideoCapture(index, backend_code)
        except TypeError:
            capture = cv2.VideoCapture(index)

        if capture and capture.isOpened():
            return capture

        if capture is not None:
            capture.release()
        return None

    def _open_first_available_camera(self) -> bool:
        attempts: List[str] = []
        for index in self._candidate_indices():
            for backend_code, backend_name in self._backend_candidates():
                attempts.append(f"{index}/{backend_name}")
                capture = self._open_capture(index, backend_code)
                if capture is None:
                    continue

                self._capture = capture
                self._active_index = index
                self._backend_name = backend_name
                self._camera_ready = True
                self._last_error = ""
                return True

        self._capture = None
        self._active_index = None
        self._backend_name = "none"
        self._camera_ready = False
        attempted_text = ", ".join(attempts) if attempts else "none"
        self._last_error = f"Failed to open camera. Tried: {attempted_text}"
        return False

    def _attempt_reconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._camera_ready = False
        self._open_first_available_camera()

    def status(self) -> Dict[str, object]:
        return {
            "camera_ready": self._camera_ready,
            "camera_requested_index": self.camera_index,
            "camera_active_index": self._active_index,
            "camera_backend": self._backend_name,
            "camera_last_error": self._last_error,
            "camera_candidates": self._candidate_indices(),
        }

    @property
    def camera_ready(self) -> bool:
        return self._camera_ready

    def start(self) -> None:
        if self._running:
            return

        if not self._open_first_available_camera():
            return

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        failed_reads = 0
        while self._running:
            if self._capture is None:
                self._attempt_reconnect()
                if self._capture is None:
                    time.sleep(0.25)
                    continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= 20:
                    self._last_error = "Camera read failed repeatedly. Reconnecting..."
                    self._attempt_reconnect()
                    failed_reads = 0
                time.sleep(0.05)
                continue

            failed_reads = 0

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            with self._lock:
                self._latest_frame = encoded.tobytes()

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._camera_ready = False

    def restart(self) -> bool:
        self.stop()
        self._latest_frame = None
        self.start()
        return self._camera_ready

    def set_camera_index(self, camera_index: int) -> bool:
        self.camera_index = camera_index
        self._last_error = f"Switching to camera index {camera_index}..."
        return self.restart()

    def toggle_camera_index(self, direction: str = "next") -> bool:
        candidates = self._candidate_indices()
        if not candidates:
            self._last_error = "No camera indices available to toggle."
            return False

        current_index = self._active_index if self._active_index is not None else self.camera_index
        if current_index in candidates:
            current_pos = candidates.index(current_index)
        else:
            current_pos = 0

        step = -1 if direction == "prev" else 1
        target_index = candidates[(current_pos + step) % len(candidates)]
        return self.set_camera_index(target_index)


def make_placeholder_frame(text: str) -> bytes:
    canvas = 255 * (cv2.UMat(480, 640, cv2.CV_8UC3).get())
    cv2.putText(
        canvas,
        text,
        (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (60, 60, 60),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", canvas)
    if not ok:
        return b""
    return encoded.tobytes()


app = Flask(__name__)

router = RobotCommandRouter()
vision_service = GeminiVisionService()
camera_stream = vision_service.camera_stream
camera_stream.start()
atexit.register(camera_stream.stop)

placeholder_frame = make_placeholder_frame("RealSense pipeline unavailable.")
analysis_placeholder_frame = make_placeholder_frame("Run a prompt to see detections.")


def stream_generator():
    boundary = b"--frame\r\n"
    header = b"Content-Type: image/jpeg\r\n\r\n"

    while True:
        frame = camera_stream.get_frame()
        if frame is None:
            if camera_stream.last_error:
                frame = make_placeholder_frame(camera_stream.last_error)
            else:
                frame = placeholder_frame

        yield boundary + header + frame + b"\r\n"
        time.sleep(0.04)


def _call_openai_interpreter(user_message: str) -> Dict[str, object]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "ok": False,
            "needs_api_key": True,
            "reply": "Set OPENAI_API_KEY to use AI mode. Local mode works without a key.",
        }

    if OpenAI is None:
        return {
            "ok": False,
            "reply": "The openai package is not installed. Run: pip install openai",
        }

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    system_prompt = (
        "You translate robot control chat into one concise command for a Dynamixel-based robot. "
        "Return plain text command only. Examples: 'connect', 'disconnect', 'read positions', "
        "'move joint 11 to 2400', 'move base to 2200', 'open gripper', 'close gripper', 'home'."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    interpreted = response.choices[0].message.content or ""
    interpreted = interpreted.strip()
    if not interpreted:
        interpreted = user_message

    return {
        "ok": True,
        "command_text": interpreted,
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        frontend_config={
            "promptCatalog": vision_service.config_summary()["prompt_catalog"],
            "defaultPromptType": "description",
        },
    )


@app.get("/api/video_feed")
def video_feed():
    return Response(stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/health")
def health():
    camera_state = camera_stream.status()
    config = vision_service.config_summary()
    return jsonify(
        {
            **camera_state,
            "robot_connected": router.robot.connected,
            "simulated_robot": router.robot.simulated,
            "vision_status": vision_service.state.snapshot()["status"],
            "gemini_installed": config["gemini_installed"],
            "gemini_api_key_present": config["api_key_present"],
            "gemini_model": config["model"],
            "realsense_installed": config["realsense_installed"],
            "robot_sdk_available": config["robot_sdk_available"],
        }
    )


@app.get("/api/vision/state")
def vision_state():
    return jsonify(
        {
            **vision_service.state.snapshot(),
            **vision_service.config_summary(),
        }
    )


@app.get("/api/vision/overlay")
def vision_overlay():
    overlay = vision_service.state.overlay_image()
    if not overlay:
        overlay = analysis_placeholder_frame
    return Response(overlay, mimetype="image/jpeg")


@app.post("/api/vision/clear")
def clear_vision_state():
    vision_service.state.clear()
    return jsonify(
        {
            "ok": True,
            **vision_service.state.snapshot(),
            **vision_service.config_summary(),
        }
    )


@app.post("/api/vision/run")
def run_vision_prompt():
    body = request.get_json(silent=True) or {}
    prompt_type = str(body.get("prompt_type", "description")).strip()
    prompt_text = str(body.get("prompt", "")).strip()
    raw_execute = body.get("execute_robot", False)
    execute_robot = raw_execute if isinstance(raw_execute, bool) else str(raw_execute).strip().lower() in {"1", "true", "yes", "on"}

    try:
        run_id = vision_service.run_analysis_async(prompt_type, prompt_text, execute_robot=execute_robot)
    except ValueError as exc:
        return jsonify({"ok": False, "reply": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"ok": False, "reply": str(exc)}), 409

    return jsonify(
        {
            "ok": True,
            "run_id": run_id,
            "reply": "Prompt analysis started.",
            **vision_service.state.snapshot(),
            **vision_service.config_summary(),
        }
    )


@app.post("/api/camera/restart")
def restart_camera():
    ok = camera_stream.restart()
    state = camera_stream.status()
    state["ok"] = ok
    return jsonify(state), (200 if ok else 503)


@app.post("/api/camera/set_index")
def set_camera_index():
    body = request.get_json(silent=True) or {}
    raw_index = body.get("camera_index")

    if raw_index is None:
        return jsonify({"ok": False, "reply": "camera_index is required."}), 400

    try:
        camera_index = int(raw_index)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "reply": "camera_index must be an integer."}), 400

    if camera_index < 0:
        return jsonify({"ok": False, "reply": "camera_index must be >= 0."}), 400

    ok = camera_stream.set_camera_index(camera_index)
    state = camera_stream.status()
    state["ok"] = ok
    return jsonify(state), (200 if ok else 503)


@app.post("/api/camera/toggle")
def toggle_camera_index():
    body = request.get_json(silent=True) or {}
    direction = str(body.get("direction", "next")).strip().lower()
    if direction not in {"next", "prev"}:
        return jsonify({"ok": False, "reply": "direction must be 'next' or 'prev'."}), 400

    ok = camera_stream.toggle_camera_index(direction=direction)
    state = camera_stream.status()
    state["ok"] = ok
    state["direction"] = direction
    return jsonify(state), (200 if ok else 503)


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    user_message = str(body.get("message", "")).strip()
    mode = str(body.get("mode", "local")).strip().lower()

    if not user_message:
        return jsonify({"ok": False, "reply": "Message cannot be empty."}), 400

    command_text = user_message
    ai_reply = None

    if mode == "openai":
        ai_result = _call_openai_interpreter(user_message)
        if not ai_result.get("ok"):
            code = 400 if ai_result.get("needs_api_key") else 500
            return jsonify(ai_result), code
        command_text = str(ai_result.get("command_text", user_message))
        ai_reply = command_text

    result = router.handle_message(command_text)
    if ai_reply:
        result["interpreted_command"] = ai_reply

    return jsonify(result)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=False,
        threaded=True,
    )
