from __future__ import annotations

import json
import socket
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import Env, spaces


@dataclass
class NeonWhiteConfig:
    """Runtime configuration for the Neon White RL bridge."""

    host: str = "127.0.0.1"
    port: int = 5555
    connect_timeout: float = 5.0
    read_timeout: float = 2.0
    ready_timeout: float = 15.0
    default_stage: int = 0
    default_timescale: float = 1.0
    wait_for_ready: bool = True
    auto_reconnect: bool = True
    enemy_buffer: int = 32
    action_rate_hz: float = 60.0


class NeonWhiteEnv(Env):
    """Gymnasium-compatible environment that wraps the NeonRLBridge TCP stream."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[NeonWhiteConfig] = None) -> None:
        self.config = config or NeonWhiteConfig()

        self.action_space = spaces.Dict(
            {
                "move": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "look": spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
                "jump": spaces.MultiBinary(1),
                "shoot": spaces.MultiBinary(1),
                "use": spaces.MultiBinary(1),
                "reset": spaces.MultiBinary(1),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "yaw": spaces.Box(-360.0, 360.0, shape=(1,), dtype=np.float32),
                "goal_dist": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                "goal_dir": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "grounded": spaces.MultiBinary(1),
                "surface": spaces.Discrete(3),
                "height_gap": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "time": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                "stage": spaces.Discrete(4096),
                "nearest_enemy_dist": spaces.Box(-1.0, np.inf, shape=(1,), dtype=np.float32),
                "nearest_enemy_dir": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "enemies_n": spaces.Box(
                    low=0,
                    high=self.config.enemy_buffer,
                    shape=(1,),
                    dtype=np.int32,
                ),
            }
        )

        self._sock: Optional[socket.socket] = None
        self._reader: Optional[Any] = None
        self._writer: Optional[Any] = None
        self._events: Deque[Dict[str, Any]] = deque()
        self._latest_level: Optional[str] = None
        self._step_sleep = 1.0 / max(1e-6, self.config.action_rate_hz)
        self._episode_steps = 0
        self._connected = False
        self._last_obs: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _ensure_connection(self) -> None:
        if self._connected and self._sock is not None:
            return

        if self._sock is not None:
            self._close_socket()

        addr = (self.config.host, self.config.port)
        self._sock = socket.create_connection(addr, timeout=self.config.connect_timeout)
        self._sock.settimeout(self.config.read_timeout)
        self._reader = self._sock.makefile("r", encoding="utf-8")
        self._writer = self._sock.makefile("w", encoding="utf-8")
        self._connected = True
        self._events.clear()
        self._latest_level = None
        self._last_obs = None

    def _close_socket(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._reader = None
        self._writer = None
        self._connected = False

    # ------------------------------------------------------------------
    # Communication helpers
    # ------------------------------------------------------------------

    def _send_command(self, name: str, **payload: Any) -> None:
        if not self._connected or self._writer is None:
            raise RuntimeError("RL bridge is not connected")
        packet = {"type": "command", "name": name}
        packet.update(payload)
        line = json.dumps(packet, separators=(",", ":"))
        self._writer.write(line + "\n")
        self._writer.flush()

    def _read_packet(self) -> Dict[str, Any]:
        if not self._connected or self._reader is None:
            raise RuntimeError("RL bridge is not connected")
        line = self._reader.readline()
        if line == "":
            raise ConnectionError("Bridge closed the connection")
        return json.loads(line)

    def _wait_for_ready(self, target_level: Optional[str]) -> None:
        deadline = time.time() + self.config.ready_timeout
        while True:
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for bridge ready packet")
            pkt = self._read_packet()
            ptype = pkt.get("type")
            if ptype == "ready":
                self._latest_level = pkt.get("level")
                if target_level is None or pkt.get("level") == target_level:
                    return
            elif ptype == "event":
                self._events.append(pkt)
            elif ptype == "obs":
                # ignore stale obs while loading levels
                continue
            else:
                continue

    def _await_obs(self) -> Dict[str, Any]:
        while True:
            pkt = self._read_packet()
            ptype = pkt.get("type")
            if ptype == "obs":
                self._last_obs = pkt
                return pkt
            if ptype == "event":
                self._events.append(pkt)
                continue
            if ptype == "ready":
                self._latest_level = pkt.get("level")
                continue
            # drop unknown packets silently

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = options or {}

        try:
            self._ensure_connection()
        except Exception:
            if not self.config.auto_reconnect:
                raise
            self._close_socket()
            self._ensure_connection()

        self._events.clear()
        self._episode_steps = 0

        stage = options.get("stage", self.config.default_stage)
        level = options.get("level")
        timescale = options.get("timescale", self.config.default_timescale)

        if level is not None:
            self._send_command("load_level", level=level)
            if self.config.wait_for_ready:
                self._wait_for_ready(level)
        elif self.config.wait_for_ready and self._latest_level is None:
            # consume initial ready after connecting
            try:
                self._wait_for_ready(None)
            except TimeoutError:
                pass

        if stage is not None:
            self._send_command("set_stage", stage=int(stage))
        if timescale is not None:
            self._send_command("timescale", value=float(timescale))

        obs = self._await_obs()
        observation = self._format_obs(obs)
        info = self._build_info(obs, reset=True)
        return observation, info

    def step(self, action: Union[np.ndarray, Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("Environment must be reset before stepping")

        payload = self._format_action(action)
        self._send_command("action", **payload)

        # sleep lightly to avoid over-driving the bridge when action rate is high
        if self._step_sleep > 0:
            time.sleep(self._step_sleep)

        obs = self._await_obs()
        observation = self._format_obs(obs)
        reward = float(obs.get("reward", 0.0))
        done = bool(obs.get("done", False))
        terminated = done
        truncated = False
        info = self._build_info(obs, reset=False)
        self._episode_steps += 1
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._close_socket()
        super().close()

    def _format_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        def arr(name: str, length: int, fallback: float = 0.0) -> np.ndarray:
            value = obs.get(name)
            if value is None:
                return np.full((length,), fallback, dtype=np.float32)
            arr_val = np.asarray(value, dtype=np.float32)
            if arr_val.shape == (length,):
                return arr_val
            return np.reshape(arr_val, (length,)).astype(np.float32)

        formatted = {
            "pos": arr("pos", 3),
            "vel": arr("vel", 3),
            "yaw": np.array([obs.get("yaw_deg", 0.0)], dtype=np.float32),
            "goal_dist": np.array([obs.get("goal_dist", 0.0)], dtype=np.float32),
            "goal_dir": arr("goal_dir", 2),
            "grounded": np.array([1 if obs.get("grounded") else 0], dtype=np.int32),
            "surface": np.array([int(obs.get("surface", 0))], dtype=np.int32),
            "height_gap": np.array([obs.get("height_gap", 0.0)], dtype=np.float32),
            "time": np.array([obs.get("time_unscaled", 0.0)], dtype=np.float32),
            "stage": np.array([int(obs.get("stage", 0))], dtype=np.int32),
            "nearest_enemy_dist": np.array([obs.get("nearest_enemy_dist", -1.0)], dtype=np.float32),
            "nearest_enemy_dir": arr("nearest_enemy_dir", 2),
            "enemies_n": np.array([min(self.config.enemy_buffer, int(obs.get("enemies_n", 0)))], dtype=np.int32),
        }
        return formatted

    def _format_action(self, action: Union[np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(action, dict):
            move = np.asarray(action.get("move", [0.0, 0.0]), dtype=np.float32)
            look = np.asarray(action.get("look", [0.0, 0.0]), dtype=np.float32)
            jump = bool(np.any(action.get("jump", False)))
            shoot = bool(np.any(action.get("shoot", False)))
            use = bool(np.any(action.get("use", False)))
            reset = bool(np.any(action.get("reset", False)))
        else:
            vec = np.asarray(action, dtype=np.float32).flatten()
            if vec.shape[0] < 6:
                raise ValueError("Expected action vector of length >= 6")
            move = vec[0:2]
            look = vec[2:4]
            jump = vec[4] > 0.5
            shoot = vec[5] > 0.5
            use = bool(vec[6] > 0.5) if vec.shape[0] > 6 else False
            reset = bool(vec[7] > 0.5) if vec.shape[0] > 7 else False

        payload = {
            "move": np.clip(move, -1.0, 1.0).tolist(),
            "look": np.clip(look, -10.0, 10.0).tolist(),
            "jump": jump,
            "shoot": shoot,
            "use": use,
            "reset": reset,
        }
        return payload

    def _build_info(self, obs: Dict[str, Any], *, reset: bool) -> Dict[str, Any]:
        events: List[Dict[str, Any]] = []
        while self._events:
            events.append(self._events.popleft())
        info: Dict[str, Any] = {
            "events": events,
            "stage": int(obs.get("stage", 0)),
            "raw_obs": obs,
            "reset": reset,
        }
        death_reason = obs.get("death_reason")
        if death_reason:
            info["death_reason"] = death_reason
        if self._latest_level is not None:
            info["level"] = self._latest_level
        return info

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def send_stage(self, stage: int) -> None:
        self._send_command("set_stage", stage=int(stage))

    def send_timescale(self, value: float) -> None:
        self._send_command("timescale", value=float(value))

    def send_load_level(self, level: str) -> None:
        self._send_command("load_level", level=level)

    def render(self) -> None:
        return None



