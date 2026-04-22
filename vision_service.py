from __future__ import annotations

import io
import json
import math
import os
import re
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - optional dependency
    rs = None

try:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except Exception:  # pragma: no cover - optional dependency
    COMM_SUCCESS = None
    PacketHandler = None
    PortHandler = None

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency
    genai = None
    types = None


PROMPT_CATALOG: Dict[str, Dict[str, Any]] = {
    "description": {
        "label": "Description",
        "task": "describe",
        "default_prompt": "Describe the scene what all objects are inside the workspace defined by the 5 ArUco markers.",
        "helper": "Uses the notebook prompt flow to describe objects inside the ArUco-calibrated workspace.",
        "supports_execution": False,
    },
    "pick_object": {
        "label": "Pick Object",
        "task": "pickup",
        "default_prompt": "Pick up the grey cube.",
        "helper": "Runs the notebook pickup plan: preview -> calibrate workspace -> execute pickup.",
        "supports_execution": True,
    },
    "pick_and_place": {
        "label": "Pick & Place",
        "task": "pick_and_place",
        "default_prompt": "Pick up the grey cube and place it next to the smartphone on the left.",
        "helper": "Runs the notebook pick-and-place plan with the same Gemini model and RealSense pipeline.",
        "supports_execution": True,
    },
}

SYSTEM_PROMPT = """
You are the perception and planning module for a 4-DOF OpenManipulator-X robot.
You receive one RGB image and one user instruction.
Decide whether the instruction is a scene description, pickup task, or pick_and_place task.

Return JSON only. Do not wrap the answer in markdown.

Output schema:
{
  "task": "describe" | "pickup" | "pick_and_place",
  "detections": [
    {
      "label": "string",
      "center": {"Y": int, "X": int},
      "bounding_box": [ymin, xmin, ymax, xmax]
    }
  ],
  "description": "string or null",
  "source": {"Y": int, "X": int} or null,
  "destination": {"Y": int, "X": int} or null,
  "trajectory": [{"Y": int, "X": int}, ...] or null
}

Rules:
- All coordinates must be integers normalized from 0 to 1000.
- Limit detections to at most 15 objects.
- For "pickup", identify the target object and fill "source".
- For "pick_and_place", identify the target object, fill "source", choose a collision-aware destination that satisfies the instruction, and return 15 trajectory points from source to destination.
- For "describe", fill "detections" and "description" only.
- Prefer the graspable center of the object for "source".
- If an object is partly occluded, estimate its center from visible geometry.
- Keep the destination clear of nearby objects when the instruction says "next to", "beside", or "on".
"""

GEMINI_MODEL = "gemini-robotics-er-1.6-preview"

ARM_LINKS = {
    "base_height": -0.160,
    "shoulder": 0.130,
    "elbow": 0.135,
    "wrist": 0.060,
}
JOINT_OFFSET_RAD = math.atan2(0.024, 0.128)

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000
DEVICENAME = "COM9"
DXL_IDS = [11, 12, 13, 14, 15]
GRIPPER_OPEN_POS = 1500
GRIPPER_CLOSED_POS = 2300
DEFAULT_POS = {
    11: 2048,
    12: 1227,
    13: 2524,
    14: 2414,
    15: GRIPPER_OPEN_POS,
}

ROBOT_BASE_FORWARD_OFFSET = 0.000
ROBOT_BASE_LATERAL_OFFSET = 0.000
ROBOT_BASE_VERTICAL_OFFSET = 0.000

TARGET_FORWARD_OFFSET = 0.000
TARGET_LATERAL_OFFSET = 0.000
DEST_FORWARD_OFFSET = 0.000
DEST_LATERAL_OFFSET = 0.000
BOARD_PLANE_PICK_Z = 0.000
BOARD_PLANE_APPROACH_Z = 0.000
BOARD_PLANE_PLACE_Z = 0.000
BOARD_PLANE_PLACE_APPROACH_Z = 0.000
WORKSPACE_X_LIMITS = (0.10, 0.36)
WORKSPACE_Y_LIMIT = 0.20


def prompt_catalog_payload() -> Dict[str, Dict[str, Any]]:
    return {key: dict(value) for key, value in PROMPT_CATALOG.items()}


def make_placeholder_frame(text: str) -> bytes:
    canvas = 255 * (cv2.UMat(480, 640, cv2.CV_8UC3).get())
    cv2.putText(
        canvas,
        text[:80],
        (18, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", canvas)
    return encoded.tobytes() if ok else b""


def extract_json_payload(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Gemini returned an empty response.")

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        brace_positions = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1]
        if brace_positions:
            start = min(brace_positions)
            end = max(cleaned.rfind("}"), cleaned.rfind("]"))
            if end >= start:
                cleaned = cleaned[start : end + 1]

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini returned JSON, but it was not an object.")
    return parsed


def _clip_norm(value: Any) -> Optional[int]:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(1000, number))


def normalize_point(point: Any) -> Optional[Dict[str, int]]:
    if not isinstance(point, dict):
        return None

    y_value = point.get("Y", point.get("y"))
    x_value = point.get("X", point.get("x"))
    y_norm = _clip_norm(y_value)
    x_norm = _clip_norm(x_value)
    if y_norm is None or x_norm is None:
        return None

    return {"Y": y_norm, "X": x_norm}


def normalize_bbox(bbox: Any) -> Optional[List[int]]:
    if isinstance(bbox, dict):
        values = [bbox.get("ymin"), bbox.get("xmin"), bbox.get("ymax"), bbox.get("xmax")]
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        values = list(bbox)
    else:
        return None

    normalized = [_clip_norm(value) for value in values]
    if any(value is None for value in normalized):
        return None

    ymin, xmin, ymax, xmax = normalized
    ymin, ymax = sorted((ymin, ymax))
    xmin, xmax = sorted((xmin, xmax))
    return [ymin, xmin, ymax, xmax]


def normalize_detections(raw_detections: Any) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if not isinstance(raw_detections, list):
        return detections

    for item in raw_detections[:15]:
        if not isinstance(item, dict):
            continue
        detections.append(
            {
                "label": str(item.get("label", "object")).strip() or "object",
                "center": normalize_point(item.get("center") or item.get("point")),
                "bounding_box": normalize_bbox(item.get("bounding_box") or item.get("bbox")),
            }
        )
    return detections


def normalize_plan(raw_plan: Dict[str, Any], prompt_type: str, prompt_text: str) -> Dict[str, Any]:
    expected_task = str(PROMPT_CATALOG.get(prompt_type, PROMPT_CATALOG["description"])["task"])
    raw_task = str(raw_plan.get("task") or expected_task).strip().lower()
    task = raw_task if raw_task in {"describe", "pickup", "pick_and_place"} else expected_task

    detections = normalize_detections(raw_plan.get("detections"))
    source = normalize_point(raw_plan.get("source"))
    destination = normalize_point(raw_plan.get("destination"))

    trajectory: List[Dict[str, int]] = []
    if isinstance(raw_plan.get("trajectory"), list):
        for item in raw_plan["trajectory"][:30]:
            point = normalize_point(item)
            if point is not None:
                trajectory.append(point)

    if source is None and detections:
        source = detections[0].get("center")

    return {
        "task": task,
        "prompt_type": prompt_type,
        "prompt_text": prompt_text,
        "detections": detections,
        "description": str(raw_plan.get("description")).strip() if raw_plan.get("description") else None,
        "source": source,
        "destination": destination,
        "trajectory": trajectory or None,
    }


def normalized_to_pixel(point: Optional[Dict[str, int]], width: int, height: int) -> Optional[tuple[int, int]]:
    if not point:
        return None

    x_norm = point.get("X")
    y_norm = point.get("Y")
    if x_norm is None or y_norm is None or width <= 0 or height <= 0:
        return None

    u = int(round(x_norm * (width - 1) / 1000.0))
    v = int(round(y_norm * (height - 1) / 1000.0))
    return u, v


def get_source_point(plan: Dict[str, Any]) -> Optional[Dict[str, int]]:
    return plan.get("source") or ((plan.get("detections") or [{}])[0].get("center") if plan.get("detections") else None)


def get_destination_point(plan: Dict[str, Any]) -> Optional[Dict[str, int]]:
    return plan.get("destination")


def build_overlay_image(frame_bgr: np.ndarray, plan: Optional[Dict[str, Any]], header_text: Optional[str] = None) -> bytes:
    image = frame_bgr.copy()
    height, width = image.shape[:2]

    header = header_text or "Prompt Output"
    if plan and header_text is None:
        detection_count = len(plan.get("detections") or [])
        header = f"{plan.get('task', 'unknown').replace('_', ' ').title()} | {detection_count} detections"

    cv2.rectangle(image, (0, 0), (width, 58), (20, 27, 45), thickness=-1)
    cv2.putText(image, header[:70], (18, 36), cv2.FONT_HERSHEY_DUPLEX, 0.82, (246, 239, 222), 1, cv2.LINE_AA)

    if plan:
        for item in plan.get("detections") or []:
            label = item.get("label", "object")
            bbox = item.get("bounding_box")
            center = item.get("center")

            if bbox:
                ymin, xmin, ymax, xmax = bbox
                x1 = int(round(xmin * (width - 1) / 1000.0))
                y1 = int(round(ymin * (height - 1) / 1000.0))
                x2 = int(round(xmax * (width - 1) / 1000.0))
                y2 = int(round(ymax * (height - 1) / 1000.0))
                cv2.rectangle(image, (x1, y1), (x2, y2), (70, 210, 180), 2)

            pixel = normalized_to_pixel(center, width, height)
            if pixel is not None:
                u, v = pixel
                cv2.circle(image, (u, v), 7, (44, 93, 255), thickness=-1)
                cv2.putText(image, label[:32], (u + 10, max(84, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (248, 247, 243), 2, cv2.LINE_AA)

        source_pixel = normalized_to_pixel(get_source_point(plan), width, height)
        if source_pixel is not None:
            cv2.circle(image, source_pixel, 11, (255, 120, 50), 3)
            cv2.putText(image, "SOURCE", (source_pixel[0] + 14, source_pixel[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 50), 2, cv2.LINE_AA)

        destination_pixel = normalized_to_pixel(get_destination_point(plan), width, height)
        if destination_pixel is not None:
            cv2.circle(image, destination_pixel, 11, (205, 60, 255), 3)
            cv2.putText(image, "DEST", (destination_pixel[0] + 14, destination_pixel[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 60, 255), 2, cv2.LINE_AA)

        trajectory = plan.get("trajectory") or []
        points = [normalized_to_pixel(item, width, height) for item in trajectory]
        points = [point for point in points if point is not None]
        if len(points) >= 2:
            for idx in range(len(points) - 1):
                cv2.line(image, points[idx], points[idx + 1], (14, 191, 255), 2, cv2.LINE_AA)

    ok, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes() if ok else b""


class RealSenseCameraStream:
    """Notebook-style RealSense pipeline used for both web preview and task execution."""

    def __init__(self) -> None:
        self._pipeline = None
        self._align = None
        self._intrinsics = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._latest_frame: Optional[bytes] = None
        self._camera_ready = False
        self._last_error = "RealSense pipeline not initialized."

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def camera_ready(self) -> bool:
        return self._camera_ready

    def _open_pipeline_locked(self) -> bool:
        if rs is None:
            self._last_error = "pyrealsense2 is not installed. Run 'pip install -r requirements.txt'."
            self._camera_ready = False
            return False

        self._close_pipeline_locked()

        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        except Exception as exc:
            self._last_error = f"Failed to open RealSense pipeline: {exc}"
            self._camera_ready = False
            self._pipeline = None
            self._align = None
            self._intrinsics = None
            return False

        self._pipeline = pipeline
        self._align = align
        self._intrinsics = intrinsics
        self._camera_ready = True
        self._last_error = ""
        return True

    def _close_pipeline_locked(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._align = None
        self._intrinsics = None
        self._camera_ready = False

    def _get_frames_locked(self):
        if self._pipeline is None or self._align is None:
            return None, None

        frames = self._pipeline.wait_for_frames(10000)
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame

    def _reader_loop(self) -> None:
        while self._running:
            try:
                with self._lock:
                    if not self._camera_ready and not self._open_pipeline_locked():
                        time.sleep(0.35)
                        continue

                    color_image, _ = self._get_frames_locked()
                    if color_image is None:
                        self._last_error = "RealSense frame read failed. Reconnecting..."
                        self._close_pipeline_locked()
                        time.sleep(0.2)
                        continue

                    ok, encoded = cv2.imencode(".jpg", color_image)
                    if ok:
                        self._latest_frame = encoded.tobytes()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"RealSense reader error: {exc}"
                    self._close_pipeline_locked()
                time.sleep(0.35)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            self._close_pipeline_locked()

    def restart(self) -> bool:
        with self._lock:
            self._close_pipeline_locked()
            self._latest_frame = None
            return self._open_pipeline_locked()

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    def capture_live_scene(self, warmup_frames: int = 15):
        with self._lock:
            last_error = None
            for attempt in range(2):
                if not self._camera_ready and not self._open_pipeline_locked():
                    last_error = self._last_error or "RealSense pipeline is not available."
                    continue

                try:
                    color_image, depth_frame = None, None
                    for _ in range(max(1, warmup_frames)):
                        color_image, depth_frame = self._get_frames_locked()
                    if color_image is None or depth_frame is None or self._intrinsics is None:
                        raise RuntimeError("Failed to capture aligned RealSense frames.")

                    ok, encoded = cv2.imencode(".jpg", color_image)
                    if ok:
                        self._latest_frame = encoded.tobytes()

                    return color_image, depth_frame, self._intrinsics
                except Exception as exc:
                    last_error = str(exc)
                    self._last_error = f"RealSense capture error: {exc}"
                    self._close_pipeline_locked()
                    self._open_pipeline_locked()

            raise RuntimeError(last_error or "RealSense pipeline is not available.")

    @contextmanager
    def pipeline_session(self) -> Iterator[tuple[Any, Any, Any]]:
        with self._lock:
            if not self._camera_ready and not self._open_pipeline_locked():
                raise RuntimeError(self._last_error or "RealSense pipeline is not available.")
            yield self._pipeline, self._align, self._intrinsics

    def status(self) -> Dict[str, object]:
        return {
            "camera_ready": self._camera_ready,
            "camera_requested_index": None,
            "camera_active_index": "realsense" if self._camera_ready else None,
            "camera_backend": "pyrealsense2" if rs is not None else "missing",
            "camera_last_error": self._last_error,
            "camera_candidates": ["realsense"],
        }

    def set_camera_index(self, camera_index: int) -> bool:
        self._last_error = "Camera index switching is not supported in RealSense notebook mode."
        return False

    def toggle_camera_index(self, direction: str = "next") -> bool:
        self._last_error = "Camera index switching is not supported in RealSense notebook mode."
        return False


class VisionRunStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self._status = "idle"
        self._phase = "idle"
        self._run_id = 0
        self._updated_at = time.time()
        self._prompt_type = "description"
        self._prompt_text = PROMPT_CATALOG["description"]["default_prompt"]
        self._execute_robot = False
        self._plan: Optional[Dict[str, Any]] = None
        self._raw_response = ""
        self._overlay_image: bytes = b""
        self._overlay_version = 0
        self._logs: List[Dict[str, Any]] = []

    def start_run(self, prompt_type: str, prompt_text: str, execute_robot: bool) -> int:
        with self._lock:
            if self._busy:
                raise RuntimeError("A prompt run is already in progress.")

            self._busy = True
            self._status = "running"
            self._phase = "starting"
            self._run_id += 1
            self._updated_at = time.time()
            self._prompt_type = prompt_type
            self._prompt_text = prompt_text
            self._execute_robot = execute_robot
            self._plan = None
            self._raw_response = ""
            self._overlay_image = b""
            self._logs = []
            return self._run_id

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            self._updated_at = time.time()

    def append_log(self, message: str, level: str = "info") -> None:
        entry = {"ts": time.strftime("%H:%M:%S"), "level": level, "message": str(message)}
        with self._lock:
            self._logs.append(entry)
            self._logs = self._logs[-300:]
            self._updated_at = time.time()

    def set_overlay(self, overlay_image: bytes) -> None:
        with self._lock:
            self._overlay_image = overlay_image
            self._overlay_version += 1
            self._updated_at = time.time()

    def set_plan(self, plan: Dict[str, Any], raw_response: str = "") -> None:
        with self._lock:
            self._plan = json.loads(json.dumps(plan))
            if raw_response:
                self._raw_response = raw_response
            self._updated_at = time.time()

    def complete(self, plan: Dict[str, Any], raw_response: str, overlay_image: bytes) -> None:
        with self._lock:
            self._busy = False
            self._status = "complete"
            self._phase = "complete"
            self._plan = json.loads(json.dumps(plan))
            self._raw_response = raw_response
            self._overlay_image = overlay_image
            self._overlay_version += 1
            self._updated_at = time.time()

    def fail(self, message: str, raw_response: str = "") -> None:
        with self._lock:
            self._busy = False
            self._status = "error"
            self._phase = "error"
            self._raw_response = raw_response
            self._updated_at = time.time()
        self.append_log(message, level="error")

    def clear(self) -> None:
        with self._lock:
            self._busy = False
            self._status = "idle"
            self._phase = "idle"
            self._updated_at = time.time()
            self._plan = None
            self._raw_response = ""
            self._overlay_image = b""
            self._logs = []

    def overlay_image(self) -> bytes:
        with self._lock:
            return self._overlay_image

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            plan = json.loads(json.dumps(self._plan)) if self._plan is not None else None
            logs = list(self._logs)
            return {
                "busy": self._busy,
                "status": self._status,
                "phase": self._phase,
                "run_id": self._run_id,
                "updated_at": self._updated_at,
                "prompt_type": self._prompt_type,
                "prompt_text": self._prompt_text,
                "execute_robot": self._execute_robot,
                "plan": plan,
                "raw_response": self._raw_response,
                "logs": logs,
                "overlay_available": bool(self._overlay_image),
                "overlay_version": self._overlay_version,
            }


class GeminiVisionService:
    def __init__(self) -> None:
        self.camera_stream = RealSenseCameraStream()
        self.state = VisionRunStore()

    @property
    def model_name(self) -> str:
        return os.getenv("GEMINI_MODEL", GEMINI_MODEL)

    def config_summary(self) -> Dict[str, Any]:
        return {
            "gemini_installed": genai is not None and types is not None,
            "realsense_installed": rs is not None,
            "robot_sdk_available": PacketHandler is not None and PortHandler is not None and COMM_SUCCESS is not None,
            "api_key_present": bool(os.getenv("GEMINI_API_KEY", "").strip()),
            "model": self.model_name,
            "prompt_catalog": prompt_catalog_payload(),
        }

    def run_analysis_async(self, prompt_type: str, prompt_text: str, execute_robot: bool = False) -> int:
        if prompt_type not in PROMPT_CATALOG:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

        prompt_text = prompt_text.strip() or str(PROMPT_CATALOG[prompt_type]["default_prompt"])
        execute_robot = bool(execute_robot) and bool(PROMPT_CATALOG[prompt_type].get("supports_execution"))
        run_id = self.state.start_run(prompt_type, prompt_text, execute_robot)
        worker = threading.Thread(
            target=self._run_workflow,
            args=(run_id, prompt_type, prompt_text, execute_robot),
            daemon=True,
        )
        worker.start()
        return run_id

    def _prepare_png(self, color_image: np.ndarray, target_height: int = 720) -> bytes:
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        if pil_image.height != target_height:
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            new_width = max(1, int(round(pil_image.width * target_height / pil_image.height)))
            pil_image = pil_image.resize((new_width, target_height), resample)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _run_gemini_task(self, color_image: np.ndarray, task_prompt: str) -> tuple[str, Dict[str, Any]]:
        if genai is None or types is None:
            raise RuntimeError("google-genai is not installed. Run 'pip install -r requirements.txt'.")

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before running the notebook pipeline in Flask.")

        self.state.append_log(f"Calling Gemini model '{self.model_name}'.")
        client = genai.Client(api_key=api_key)
        image_bytes = self._prepare_png(color_image)

        config_kwargs: Dict[str, Any] = {"temperature": 0.2}
        if hasattr(types, "ThinkingConfig"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        response = client.models.generate_content(
            model=self.model_name,
            contents=[
                SYSTEM_PROMPT,
                task_prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        raw_response = (response.text or "").strip()
        raw_plan = extract_json_payload(raw_response)
        plan = normalize_plan(raw_plan, self.state.snapshot()["prompt_type"], task_prompt)
        return raw_response, plan

    def _run_workflow(self, run_id: int, prompt_type: str, prompt_text: str, execute_robot: bool) -> None:
        raw_response = ""
        port_handler = None

        try:
            self.state.append_log(f"Run {run_id} started with notebook pipeline.")
            self.state.append_log(f"Prompt: {prompt_text}")

            if rs is None:
                raise RuntimeError("pyrealsense2 is not installed. The notebook pipeline requires a RealSense camera.")

            self.state.set_phase("capturing_preview")
            preview_color, _, _ = self.camera_stream.capture_live_scene()
            self.state.append_log("Captured RealSense preview frame.")

            raw_response, preview_plan = self._run_gemini_task(preview_color, prompt_text)
            preview_overlay = build_overlay_image(preview_color, preview_plan, header_text="Notebook Preview")
            self.state.set_overlay(preview_overlay)
            self.state.set_plan(preview_plan, raw_response)
            self._log_plan_details(preview_plan, label_prefix="Preview")

            preview_task = str(preview_plan.get("task", "")).lower()
            if preview_task == "describe":
                self.state.complete(preview_plan, raw_response=raw_response, overlay_image=preview_overlay)
                self.state.append_log("Describe task completed. No robot motion required.")
                return

            if not execute_robot:
                self.state.complete(preview_plan, raw_response=raw_response, overlay_image=preview_overlay)
                self.state.append_log("Preview completed. Robot execution is disabled in the UI.")
                return

            if PacketHandler is None or PortHandler is None or COMM_SUCCESS is None:
                raise RuntimeError("dynamixel_sdk is not installed. Hardware execution requires the SDK.")

            self.state.set_phase("connecting_robot")
            self.state.append_log("Connecting to Dynamixel chain using notebook defaults.")
            port_handler, packet_handler = self._init_dynamixel()
            if not port_handler:
                raise RuntimeError("Failed to connect to Dynamixel. Please verify COM port.")

            self.state.append_log("Moving robot to notebook default position.")
            self._set_default_positions(port_handler, packet_handler)

            self.state.set_phase("calibrating_workspace")
            self.state.append_log("Calibrating workspace using ArUco markers 0,1,2,3,4.")
            with self.camera_stream.pipeline_session() as (pipeline, align, intrinsics):
                detector_params, aruco_dict, detector, opencv_old = self._init_aruco()
                robot_frame = self._calibrate_workspace(
                    pipeline,
                    align,
                    intrinsics,
                    detector_params,
                    aruco_dict,
                    detector,
                    opencv_old,
                )

                self.state.set_phase("capturing_execution_scene")
                self.state.append_log("Capturing a fresh scene for execution.")
                color_image, depth_frame = self._capture_live_scene_locked(pipeline, align)

                self.state.set_phase("planning_execution")
                raw_response, execution_plan = self._run_gemini_task(color_image, prompt_text)
                execution_overlay = build_overlay_image(color_image, execution_plan, header_text="Notebook Execution Plan")
                self.state.set_overlay(execution_overlay)
                self.state.set_plan(execution_plan, raw_response)
                self._log_plan_details(execution_plan, label_prefix="Execution")

                task = str(execution_plan.get("task", "")).lower()
                self.state.set_phase("executing_robot")
                if task == "pickup":
                    self._execute_pickup_plan(execution_plan, color_image, depth_frame, intrinsics, robot_frame, port_handler, packet_handler)
                elif task == "pick_and_place":
                    self._execute_pick_and_place_plan(execution_plan, color_image, depth_frame, intrinsics, robot_frame, port_handler, packet_handler)
                elif task == "describe":
                    self.state.append_log("Execution prompt resolved to describe, so robot motion was skipped.")
                else:
                    raise ValueError(f"Unsupported task returned by Gemini: {task}")

            final_overlay = self.state.overlay_image() or preview_overlay
            self.state.complete(self.state.snapshot()["plan"] or preview_plan, raw_response=raw_response, overlay_image=final_overlay)
            self.state.append_log("Notebook pipeline run completed.")
        except Exception as exc:  # pragma: no cover - hardware / external API dependent
            trace = traceback.format_exc(limit=4)
            self.state.fail(str(exc), raw_response=raw_response)
            self.state.append_log(trace, level="error")
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            if port_handler is not None:
                try:
                    self.state.append_log("Closing Dynamixel port while keeping torque enabled, matching the notebook.")
                    port_handler.closePort()
                except Exception:
                    pass

    def _log_plan_details(self, plan: Dict[str, Any], label_prefix: str = "Plan") -> None:
        task = str(plan.get("task", "unknown")).replace("_", " ")
        detections = plan.get("detections") or []
        self.state.append_log(f"{label_prefix} task: {task}. Detections: {len(detections)}.")
        source = plan.get("source")
        if source:
            self.state.append_log(f"{label_prefix} source: X={source['X']}, Y={source['Y']}.")
        destination = plan.get("destination")
        if destination:
            self.state.append_log(f"{label_prefix} destination: X={destination['X']}, Y={destination['Y']}.")
        trajectory = plan.get("trajectory") or []
        if trajectory:
            self.state.append_log(f"{label_prefix} trajectory points: {len(trajectory)}.")
        description = plan.get("description")
        if description:
            self.state.append_log(f"{label_prefix} description: {description}")

    def _capture_live_scene_locked(self, pipeline, align, warmup_frames: int = 15):
        color_image, depth_frame = None, None
        for _ in range(max(1, warmup_frames)):
            color_image, depth_frame = self._get_frames_locked(pipeline, align)
        if color_image is None or depth_frame is None:
            raise RuntimeError("Failed to capture aligned RealSense frames.")

        ok, encoded = cv2.imencode(".jpg", color_image)
        if ok:
            self.camera_stream._latest_frame = encoded.tobytes()
        return color_image, depth_frame

    def _get_frames_locked(self, pipeline, align):
        frames = pipeline.wait_for_frames(10000)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame

    def _init_aruco(self):
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            detector_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
            return detector_params, aruco_dict, detector, False
        except AttributeError:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            detector_params = cv2.aruco.DetectorParameters_create()
            return detector_params, aruco_dict, None, True

    def _detect_markers(self, image, aruco_dict, detector, opencv_old, detector_params):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if opencv_old:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        else:
            corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids

    def _get_smoothed_depth(self, depth_frame, u, v, radius: int = 2) -> float:
        values = []
        height = depth_frame.get_height()
        width = depth_frame.get_width()
        for dv in range(-radius, radius + 1):
            for du in range(-radius, radius + 1):
                uu = max(0, min(width - 1, u + du))
                vv = max(0, min(height - 1, v + dv))
                depth_value = depth_frame.get_distance(uu, vv)
                if depth_value > 0:
                    values.append(depth_value)
        return float(np.median(values)) if values else 0.0

    def _normalize_vector(self, vec):
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return None
        return vec / norm

    def _project_point_to_plane(self, point, plane_point, plane_normal):
        return point - np.dot(point - plane_point, plane_normal) * plane_normal

    def _get_marker_center_camera_point(self, corners, ids, marker_id, depth_frame, intrinsics):
        if ids is None:
            return None

        detected_ids = ids.flatten().tolist()
        if marker_id not in detected_ids:
            return None

        idx = detected_ids.index(marker_id)
        marker_corners = corners[idx][0]
        u = int(np.mean(marker_corners[:, 0]))
        v = int(np.mean(marker_corners[:, 1]))
        depth_value = self._get_smoothed_depth(depth_frame, u, v)
        if depth_value <= 0:
            return None

        point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_value)
        return np.array(point, dtype=np.float32)

    def _build_robot_frame(self, corners, ids, depth_frame, intrinsics):
        required_ids = [0, 1, 2, 3, 4]
        marker_points = {}
        for marker_id in required_ids:
            point = self._get_marker_center_camera_point(corners, ids, marker_id, depth_frame, intrinsics)
            if point is None:
                return None, f"missing 3D point for marker {marker_id}"
            marker_points[marker_id] = point

        p0 = marker_points[0]
        p1 = marker_points[1]
        p2 = marker_points[2]
        p3 = marker_points[3]
        p4 = marker_points[4]

        board_right = self._normalize_vector(p1 - p0)
        board_forward_raw = p3 - p0
        board_forward_raw = board_forward_raw - np.dot(board_forward_raw, board_right) * board_right
        board_forward = self._normalize_vector(board_forward_raw)
        if board_right is None or board_forward is None:
            return None, "failed to build board axes"

        board_up = self._normalize_vector(np.cross(board_forward, board_right))
        if board_up is None:
            return None, "failed to build board normal"

        workspace_center = 0.25 * (p0 + p1 + p2 + p3)
        if np.dot(p4 - workspace_center, board_up) < 0:
            board_up = -board_up

        base_projection = self._project_point_to_plane(p4, p0, board_up)
        origin = (
            base_projection
            + ROBOT_BASE_FORWARD_OFFSET * board_forward
            + ROBOT_BASE_LATERAL_OFFSET * board_right
            + ROBOT_BASE_VERTICAL_OFFSET * board_up
        )

        return {
            "origin": origin,
            "x_axis": board_forward,
            "y_axis": board_right,
            "z_axis": board_up,
            "workspace_center": workspace_center,
            "base_marker_center": p4,
        }, None

    def _average_robot_frames(self, frames):
        if not frames:
            return None

        def avg_vec(key):
            return np.mean([frame[key] for frame in frames], axis=0)

        x_axis = self._normalize_vector(avg_vec("x_axis"))
        if x_axis is None:
            return None

        y_axis_seed = avg_vec("y_axis")
        y_axis_seed = y_axis_seed - np.dot(y_axis_seed, x_axis) * x_axis
        y_axis = self._normalize_vector(y_axis_seed)
        if y_axis is None:
            return None

        z_axis = self._normalize_vector(np.cross(x_axis, y_axis))
        if z_axis is None:
            return None

        y_axis = self._normalize_vector(np.cross(z_axis, x_axis))
        if y_axis is None:
            return None

        return {
            "origin": avg_vec("origin"),
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
            "workspace_center": avg_vec("workspace_center"),
            "base_marker_center": avg_vec("base_marker_center"),
        }

    def _calibrate_workspace(self, pipeline, align, intrinsics, detector_params, aruco_dict, detector, opencv_old, stable_frames: int = 8):
        collected_frames = []
        last_count = -1
        self.state.append_log("Keep markers 0,1,2,3,4 clearly visible during calibration.")

        while True:
            color_image, depth_frame = self._get_frames_locked(pipeline, align)
            if color_image is None:
                continue

            corners, ids = self._detect_markers(color_image, aruco_dict, detector, opencv_old, detector_params)
            display_image = color_image.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display_image, corners, ids)
            self.state.set_overlay(build_overlay_image(display_image, None, header_text="ArUco Workspace Calibration"))

            robot_frame, frame_error = self._build_robot_frame(corners, ids, depth_frame, intrinsics)
            if robot_frame is None:
                if last_count != 0:
                    self.state.append_log(f"Calibration waiting: {frame_error}.")
                    last_count = 0
                collected_frames.clear()
                continue

            collected_frames.append(robot_frame)
            if len(collected_frames) > stable_frames:
                collected_frames.pop(0)

            if len(collected_frames) != last_count:
                self.state.append_log(f"Calibration progress: {len(collected_frames)}/{stable_frames} stable frames.")
                last_count = len(collected_frames)

            if len(collected_frames) >= stable_frames:
                calibrated_frame = self._average_robot_frames(collected_frames)
                if calibrated_frame is None:
                    collected_frames.clear()
                    last_count = 0
                    self.state.append_log("Calibration rejected because the averaged frame was invalid. Retrying...")
                    continue

                self.state.append_log("Workspace calibration locked.")
                self.state.append_log(f"Origin: {[round(v, 3) for v in calibrated_frame['origin']]}")
                self.state.append_log(f"X axis: {[round(v, 3) for v in calibrated_frame['x_axis']]}")
                self.state.append_log(f"Y axis: {[round(v, 3) for v in calibrated_frame['y_axis']]}")
                self.state.append_log(f"Z axis: {[round(v, 3) for v in calibrated_frame['z_axis']]}")
                return calibrated_frame

    def _pixel_to_camera(self, u, v, depth_meters, intrinsics):
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_meters)
        return np.array(point, dtype=np.float32)

    def _transform_camera_to_robot(self, point_camera, robot_frame):
        delta = point_camera[:3] - robot_frame["origin"]
        return np.array(
            [
                float(np.dot(delta, robot_frame["x_axis"])),
                float(np.dot(delta, robot_frame["y_axis"])),
                float(np.dot(delta, robot_frame["z_axis"])),
            ],
            dtype=np.float32,
        )

    def _inverse_kinematics(self, x, y, z):
        l1 = ARM_LINKS["base_height"]
        l2 = ARM_LINKS["shoulder"]
        l3 = ARM_LINKS["elbow"]
        l4 = ARM_LINKS["wrist"]

        joint1 = math.atan2(y, x)
        r = math.sqrt(x**2 + y**2)
        z_prime = z - l1

        wrist_r = r - l4
        if wrist_r <= 0:
            return None

        reach = math.sqrt(wrist_r**2 + z_prime**2)
        min_reach = abs(l2 - l3) + 1e-6
        max_reach = (l2 + l3) - 1e-6
        if reach < min_reach or reach > max_reach:
            return None

        d_value = (wrist_r**2 + z_prime**2 - l2**2 - l3**2) / (2 * l2 * l3)
        d_value = max(-1.0, min(1.0, d_value))

        elbow_eff = math.atan2(math.sqrt(max(0.0, 1 - d_value**2)), d_value)
        shoulder_eff = math.atan2(z_prime, wrist_r) - math.atan2(l3 * math.sin(elbow_eff), l2 + l3 * math.cos(elbow_eff))

        joint2 = shoulder_eff + JOINT_OFFSET_RAD
        joint3 = elbow_eff - JOINT_OFFSET_RAD
        joint4 = -(joint2 + joint3)
        return [joint1, joint2, joint3, joint4]

    def _init_dynamixel(self):
        port_handler = PortHandler(DEVICENAME)
        packet_handler = PacketHandler(PROTOCOL_VERSION)
        if not port_handler.openPort() or not port_handler.setBaudRate(BAUDRATE):
            return None, None

        for dxl_id in DXL_IDS:
            packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        return port_handler, packet_handler

    def _rad_to_dynamixel(self, angle):
        pos = int((angle + math.pi) * (4095.0 / (2 * math.pi)))
        return max(0, min(4095, pos))

    def _dynamixel_to_rad(self, pos):
        return (pos * (2 * math.pi) / 4095.0) - math.pi

    def _smooth_profile(self, t):
        return 0.5 * (1 - math.cos(math.pi * t))

    def _send_joint_positions(self, port_handler, packet_handler, q_positions):
        for i, dxl_id in enumerate([11, 12, 13, 14]):
            pos = self._rad_to_dynamixel(q_positions[i])
            packet_handler.write4ByteTxRx(port_handler, dxl_id, ADDR_GOAL_POSITION, pos)

    def _move_smooth(self, port_handler, packet_handler, q_start, q_goal, duration):
        steps = 80
        dt = duration / steps
        for step in range(steps + 1):
            t = step / steps
            alpha = self._smooth_profile(t)
            q = [qs + alpha * (qg - qs) for qs, qg in zip(q_start, q_goal)]
            self._send_joint_positions(port_handler, packet_handler, q)
            time.sleep(dt)

    def _set_gripper(self, port_handler, packet_handler, open_gripper: bool = True):
        pos = GRIPPER_OPEN_POS if open_gripper else GRIPPER_CLOSED_POS
        packet_handler.write4ByteTxRx(port_handler, 15, ADDR_GOAL_POSITION, pos)

    def _read_positions(self, port_handler, packet_handler):
        positions = {}
        for dxl_id in DXL_IDS:
            pos, dxl_comm, dxl_err = packet_handler.read4ByteTxRx(port_handler, dxl_id, ADDR_PRESENT_POSITION)
            if dxl_comm == COMM_SUCCESS and dxl_err == 0:
                positions[dxl_id] = pos
        return positions

    def _set_default_positions(self, port_handler, packet_handler):
        current_positions = self._read_positions(port_handler, packet_handler)
        if not current_positions:
            self.state.append_log("Failed to read current positions. Moving directly to default...")
            for dxl_id, pos in DEFAULT_POS.items():
                packet_handler.write4ByteTxRx(port_handler, dxl_id, ADDR_GOAL_POSITION, pos)
            time.sleep(1.5)
            return

        steps = 50
        step_delay = 0.03
        increments = {}
        for dxl_id in DEFAULT_POS:
            if dxl_id in current_positions:
                increments[dxl_id] = (DEFAULT_POS[dxl_id] - current_positions[dxl_id]) / steps

        for step in range(1, steps + 1):
            for dxl_id in DEFAULT_POS:
                if dxl_id in increments:
                    int_pos = int(current_positions[dxl_id] + (increments[dxl_id] * step))
                    int_pos = max(0, min(4095, int_pos))
                    packet_handler.write4ByteTxRx(port_handler, dxl_id, ADDR_GOAL_POSITION, int_pos)
            time.sleep(step_delay)

    def _is_target_in_workspace(self, x, y):
        if x < WORKSPACE_X_LIMITS[0]:
            return False, f"x={x:.3f} is too close to the robot base"
        if x > WORKSPACE_X_LIMITS[1]:
            return False, f"x={x:.3f} is beyond the current workspace limit"
        if abs(y) > WORKSPACE_Y_LIMIT:
            return False, f"|y|={abs(y):.3f} is beyond the side-to-side limit"
        return True, "inside workspace"

    def _current_joint_radians(self, port_handler, packet_handler):
        curr_pos_dict = self._read_positions(port_handler, packet_handler)
        if not curr_pos_dict or not all(i in curr_pos_dict for i in [11, 12, 13, 14]):
            raise RuntimeError("Hardware missed a position read. Check the Dynamixels and retry.")
        return [
            self._dynamixel_to_rad(curr_pos_dict[11]),
            self._dynamixel_to_rad(curr_pos_dict[12]),
            self._dynamixel_to_rad(curr_pos_dict[13]),
            self._dynamixel_to_rad(curr_pos_dict[14]),
        ]

    def _default_joint_radians(self):
        return [
            self._dynamixel_to_rad(DEFAULT_POS[11]),
            self._dynamixel_to_rad(DEFAULT_POS[12]),
            self._dynamixel_to_rad(DEFAULT_POS[13]),
            self._dynamixel_to_rad(DEFAULT_POS[14]),
        ]

    def _gemini_point_to_robot_target(self, point, color_image, depth_frame, intrinsics, robot_frame, x_offset=0.0, y_offset=0.0):
        h, w = color_image.shape[:2]
        pixel = normalized_to_pixel(point, w, h)
        if pixel is None:
            raise ValueError("Gemini response did not include a usable point.")

        u, v = pixel
        depth_value = self._get_smoothed_depth(depth_frame, u, v)
        if depth_value <= 0:
            raise RuntimeError(f"Depth lookup failed at pixel ({u}, {v}).")

        point_camera = self._pixel_to_camera(u, v, depth_value, intrinsics)
        point_robot = self._transform_camera_to_robot(point_camera, robot_frame)
        tx, ty, raw_tz = point_robot.tolist()
        tx += x_offset
        ty += y_offset

        return {
            "pixel": (u, v),
            "depth": float(depth_value),
            "raw_robot": tuple(float(value) for value in point_robot),
            "target_xy": (float(tx), float(ty)),
            "raw_tz": float(raw_tz),
        }

    def _log_target_info(self, name: str, info: Dict[str, Any]) -> None:
        self.state.append_log(
            f"{name} -> pixel={info['pixel']}, depth={info['depth']:.3f}m, "
            f"robot_xyz=({info['raw_robot'][0]:.3f}, {info['raw_robot'][1]:.3f}, {info['raw_robot'][2]:.3f})"
        )

    def _solve_plan_ik(self, x, y, goal_z, approach_z):
        q_goal = self._inverse_kinematics(x, y, goal_z)
        q_above = self._inverse_kinematics(x, y, approach_z)
        if q_goal is None or q_above is None:
            raise RuntimeError(f"IK failed for x={x:.3f}, y={y:.3f}, goal_z={goal_z:.3f}, approach_z={approach_z:.3f}.")
        return q_goal, q_above

    def _execute_pickup_plan(self, plan, color_image, depth_frame, intrinsics, robot_frame, port_handler, packet_handler):
        source_point = get_source_point(plan)
        if source_point is None:
            raise ValueError("Gemini did not return a source point for the pickup task.")

        source_info = self._gemini_point_to_robot_target(
            source_point,
            color_image,
            depth_frame,
            intrinsics,
            robot_frame,
            x_offset=TARGET_FORWARD_OFFSET,
            y_offset=TARGET_LATERAL_OFFSET,
        )
        self._log_target_info("Pick source", source_info)

        tx, ty = source_info["target_xy"]
        workspace_ok, workspace_reason = self._is_target_in_workspace(tx, ty)
        if not workspace_ok:
            raise RuntimeError(f"Pickup target is outside the safe workspace: {workspace_reason}")

        q_goal, q_above = self._solve_plan_ik(tx, ty, BOARD_PLANE_PICK_Z, BOARD_PLANE_APPROACH_Z)
        q_start = self._current_joint_radians(port_handler, packet_handler)
        q_default = self._default_joint_radians()

        self.state.append_log("Executing pickup sequence...")
        self._set_gripper(port_handler, packet_handler, open_gripper=True)
        time.sleep(0.25)

        q_rotate = [q_above[0], q_start[1], q_start[2], q_start[3]]
        self._move_smooth(port_handler, packet_handler, q_start, q_rotate, 1.2)
        self._move_smooth(port_handler, packet_handler, q_rotate, q_above, 2.0)
        self._move_smooth(port_handler, packet_handler, q_above, q_goal, 1.5)
        self._set_gripper(port_handler, packet_handler, open_gripper=False)
        time.sleep(0.4)
        self._move_smooth(port_handler, packet_handler, q_goal, q_above, 2.0)
        self._move_smooth(port_handler, packet_handler, q_above, q_default, 3.0)
        self.state.append_log("Pickup completed.")

    def _execute_pick_and_place_plan(self, plan, color_image, depth_frame, intrinsics, robot_frame, port_handler, packet_handler):
        source_point = get_source_point(plan)
        destination_point = get_destination_point(plan)
        if source_point is None:
            raise ValueError("Gemini did not return a source point for the pick-and-place task.")
        if destination_point is None:
            raise ValueError("Gemini did not return a destination point for the pick-and-place task.")

        source_info = self._gemini_point_to_robot_target(
            source_point,
            color_image,
            depth_frame,
            intrinsics,
            robot_frame,
            x_offset=TARGET_FORWARD_OFFSET,
            y_offset=TARGET_LATERAL_OFFSET,
        )
        destination_info = self._gemini_point_to_robot_target(
            destination_point,
            color_image,
            depth_frame,
            intrinsics,
            robot_frame,
            x_offset=DEST_FORWARD_OFFSET,
            y_offset=DEST_LATERAL_OFFSET,
        )
        self._log_target_info("Pick source", source_info)
        self._log_target_info("Place destination", destination_info)

        source_ok, source_reason = self._is_target_in_workspace(*source_info["target_xy"])
        if not source_ok:
            raise RuntimeError(f"Pickup target is outside the safe workspace: {source_reason}")

        destination_ok, destination_reason = self._is_target_in_workspace(*destination_info["target_xy"])
        if not destination_ok:
            raise RuntimeError(f"Placement target is outside the safe workspace: {destination_reason}")

        q_pick, q_pick_above = self._solve_plan_ik(source_info["target_xy"][0], source_info["target_xy"][1], BOARD_PLANE_PICK_Z, BOARD_PLANE_APPROACH_Z)
        q_place, q_place_above = self._solve_plan_ik(destination_info["target_xy"][0], destination_info["target_xy"][1], BOARD_PLANE_PLACE_Z, BOARD_PLANE_PLACE_APPROACH_Z)

        if plan.get("trajectory"):
            self.state.append_log("Gemini trajectory returned. Execution still uses source/destination waypoints with vertical approach and retreat, matching the notebook.")

        q_start = self._current_joint_radians(port_handler, packet_handler)
        q_default = self._default_joint_radians()

        self.state.append_log("Executing pick-and-place sequence...")
        self._set_gripper(port_handler, packet_handler, open_gripper=True)
        time.sleep(0.25)

        q_rotate = [q_pick_above[0], q_start[1], q_start[2], q_start[3]]
        self._move_smooth(port_handler, packet_handler, q_start, q_rotate, 1.2)
        self._move_smooth(port_handler, packet_handler, q_rotate, q_pick_above, 2.0)
        self._move_smooth(port_handler, packet_handler, q_pick_above, q_pick, 1.5)
        self._set_gripper(port_handler, packet_handler, open_gripper=False)
        time.sleep(0.4)
        self._move_smooth(port_handler, packet_handler, q_pick, q_pick_above, 2.0)
        self._move_smooth(port_handler, packet_handler, q_pick_above, q_default, 3.0)
        self._move_smooth(port_handler, packet_handler, q_default, q_place_above, 2.5)
        self._move_smooth(port_handler, packet_handler, q_place_above, q_place, 1.5)
        self._set_gripper(port_handler, packet_handler, open_gripper=True)
        time.sleep(0.4)
        self._move_smooth(port_handler, packet_handler, q_place, q_place_above, 2.0)
        self._move_smooth(port_handler, packet_handler, q_place_above, q_default, 3.0)
        self.state.append_log("Pick-and-place completed.")
