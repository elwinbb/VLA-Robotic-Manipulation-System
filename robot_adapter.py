import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple

try:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except Exception:  # pragma: no cover - optional dependency
    COMM_SUCCESS = None
    PacketHandler = None
    PortHandler = None


@dataclass
class RobotConfig:
    ids: Tuple[int, ...] = (11, 12, 13, 14, 15)
    baudrate: int = 1000000
    device_name: str = "COM9"
    protocol_version: float = 2.0
    addr_torque_enable: int = 64
    addr_goal_position: int = 116
    addr_present_position: int = 132
    torque_enable: int = 1
    torque_disable: int = 0


class RobotBridge:
    """Thin wrapper around Dynamixel SDK with a simulation fallback."""

    def __init__(self, config: RobotConfig | None = None):
        self.config = config or RobotConfig()
        self.simulated = bool(PortHandler is None or PacketHandler is None)
        self.connected = False
        self._lock = threading.Lock()
        self._positions: Dict[int, int] = {dxl_id: 2048 for dxl_id in self.config.ids}

        self._port_handler = None
        self._packet_handler = None

        if not self.simulated:
            self._port_handler = PortHandler(self.config.device_name)
            self._packet_handler = PacketHandler(self.config.protocol_version)

    def _clamp(self, value: int) -> int:
        return max(0, min(4095, value))

    def connect(self) -> tuple[bool, str]:
        with self._lock:
            if self.connected:
                return True, "Robot is already connected."

            if self.simulated:
                self.connected = True
                return True, "Simulation mode connected (no hardware SDK detected)."

            assert self._port_handler is not None
            assert self._packet_handler is not None

            if not self._port_handler.openPort():
                return False, "Failed to open serial port. Check device name and cable."

            if not self._port_handler.setBaudRate(self.config.baudrate):
                self._port_handler.closePort()
                return False, "Failed to set baudrate."

            for dxl_id in self.config.ids:
                comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
                    self._port_handler,
                    dxl_id,
                    self.config.addr_torque_enable,
                    self.config.torque_enable,
                )
                if comm_result != COMM_SUCCESS:
                    self._port_handler.closePort()
                    msg = self._packet_handler.getTxRxResult(comm_result)
                    return False, f"ID {dxl_id} communication error: {msg}"
                if dxl_error != 0:
                    self._port_handler.closePort()
                    msg = self._packet_handler.getRxPacketError(dxl_error)
                    return False, f"ID {dxl_id} packet error: {msg}"

            self.connected = True
            return True, "Robot connected and torque enabled."

    def disconnect(self) -> tuple[bool, str]:
        with self._lock:
            if not self.connected:
                return True, "Robot is already disconnected."

            if self.simulated:
                self.connected = False
                return True, "Simulation mode disconnected."

            assert self._port_handler is not None
            assert self._packet_handler is not None

            for dxl_id in self.config.ids:
                self._packet_handler.write1ByteTxRx(
                    self._port_handler,
                    dxl_id,
                    self.config.addr_torque_enable,
                    self.config.torque_disable,
                )

            self._port_handler.closePort()
            self.connected = False
            return True, "Robot disconnected and torque disabled."

    def read_positions(self) -> tuple[bool, Dict[int, int], str]:
        with self._lock:
            if not self.connected:
                return False, {}, "Robot is not connected."

            if self.simulated:
                return True, dict(self._positions), "Read simulated positions."

            assert self._port_handler is not None
            assert self._packet_handler is not None

            data: Dict[int, int] = {}
            for dxl_id in self.config.ids:
                pos, comm_result, dxl_error = self._packet_handler.read4ByteTxRx(
                    self._port_handler,
                    dxl_id,
                    self.config.addr_present_position,
                )
                if comm_result != COMM_SUCCESS:
                    msg = self._packet_handler.getTxRxResult(comm_result)
                    return False, {}, f"ID {dxl_id} communication error: {msg}"
                if dxl_error != 0:
                    msg = self._packet_handler.getRxPacketError(dxl_error)
                    return False, {}, f"ID {dxl_id} packet error: {msg}"
                data[dxl_id] = int(pos)

            self._positions.update(data)
            return True, data, "Read live joint positions."

    def set_position(self, dxl_id: int, position: int) -> tuple[bool, str]:
        with self._lock:
            if not self.connected:
                return False, "Robot is not connected."
            if dxl_id not in self.config.ids:
                return False, f"Unknown joint ID {dxl_id}."

            target = self._clamp(position)

            if self.simulated:
                self._positions[dxl_id] = target
                return True, f"Simulated move: joint {dxl_id} -> {target}."

            assert self._port_handler is not None
            assert self._packet_handler is not None

            comm_result, dxl_error = self._packet_handler.write4ByteTxRx(
                self._port_handler,
                dxl_id,
                self.config.addr_goal_position,
                target,
            )
            if comm_result != COMM_SUCCESS:
                msg = self._packet_handler.getTxRxResult(comm_result)
                return False, f"Communication error: {msg}"
            if dxl_error != 0:
                msg = self._packet_handler.getRxPacketError(dxl_error)
                return False, f"Packet error: {msg}"

            self._positions[dxl_id] = target
            return True, f"Moved joint {dxl_id} to {target}."

    def move_home(self) -> tuple[bool, str]:
        with self._lock:
            if not self.connected:
                return False, "Robot is not connected."

            # Slightly open gripper at home for safer default behavior.
            home_map = {11: 2048, 12: 2048, 13: 2048, 14: 2048, 15: 2350}

        for dxl_id, target in home_map.items():
            ok, msg = self.set_position(dxl_id, target)
            if not ok:
                return False, msg
            time.sleep(0.02)

        return True, "Robot moved to home/default position."


class RobotCommandRouter:
    """Parses chat text into robot actions."""

    JOINT_BY_NAME = {
        "base": 11,
        "shoulder": 12,
        "elbow": 13,
        "wrist": 14,
        "gripper": 15,
    }

    def __init__(self):
        self.robot = RobotBridge()

    def _format_positions(self, positions: Dict[int, int]) -> str:
        pairs = [f"ID {k}: {v}" for k, v in sorted(positions.items())]
        return ", ".join(pairs)

    def _reply(self, ok: bool, reply: str, positions: Dict[int, int] | None = None) -> Dict[str, object]:
        return {
            "ok": ok,
            "reply": reply,
            "positions": positions or {},
            "robot_connected": self.robot.connected,
            "simulated_robot": self.robot.simulated,
        }

    def handle_message(self, message: str) -> Dict[str, object]:
        text = (message or "").strip().lower()
        if not text:
            return self._reply(False, "Please type a command.")

        if text in {"connect", "start", "enable torque", "power on"}:
            ok, msg = self.robot.connect()
            return self._reply(ok, msg)

        if text in {"disconnect", "stop", "disable torque", "power off"}:
            ok, msg = self.robot.disconnect()
            return self._reply(ok, msg)

        if text in {"read", "read positions", "status", "where are you"}:
            ok, positions, msg = self.robot.read_positions()
            if ok:
                return self._reply(True, f"{msg} {self._format_positions(positions)}", positions)
            return self._reply(False, msg)

        if text in {"home", "default", "default position", "go home"}:
            ok, msg = self.robot.move_home()
            return self._reply(ok, msg)

        if "open gripper" in text:
            ok, msg = self.robot.set_position(15, 2600)
            return self._reply(ok, msg)

        if "close gripper" in text:
            ok, msg = self.robot.set_position(15, 1750)
            return self._reply(ok, msg)

        # Examples: "move joint 11 to 2200", "move id 12 to 1800"
        match_id = re.search(r"move\s+(?:joint|id)\s*(\d+)\s*(?:to)?\s*(\d+)", text)
        if match_id:
            dxl_id = int(match_id.group(1))
            target = int(match_id.group(2))
            ok, msg = self.robot.set_position(dxl_id, target)
            return self._reply(ok, msg)

        # Example: "move base to 2300"
        match_name = re.search(r"move\s+(base|shoulder|elbow|wrist|gripper)\s*(?:to)?\s*(\d+)", text)
        if match_name:
            joint_name = match_name.group(1)
            dxl_id = self.JOINT_BY_NAME[joint_name]
            target = int(match_name.group(2))
            ok, msg = self.robot.set_position(dxl_id, target)
            return self._reply(ok, msg)

        return self._reply(
            False,
            "I did not understand that. Try: connect, read positions, home, move joint 11 to 2300, open gripper.",
        )