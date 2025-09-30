#!/usr/bin/env python3
from __future__ import annotations

"""
Reinforcement learning training script for Neon White movement drills.

Additions in this version:
- External per-level JSON config (goals, enemies, names).
- CLI to add/update level entries right from the console at launch.
- Coordinate-frame transforms for JSON coords (axis reorder, sign flips, offset, scale).
- Optional live pos/goal debug printout to calibrate transforms.
- Observation overrides from JSON (goal_dist/dir, nearest_enemy_* , enemies_n).
- Termination + big reward when within goal_radius of hardcoded (transformed) goal.
- OCR "LEVEL COMPLETE" remains optional and throttled (off by default).
- JSON-goal–based reward shaping with tunable scaling, step penalty, and death/timeout/stuck penalties.
- Fixed progress spikes by re-baselining progress on reset and warmup suppression.
- Jump-reset guard: if goal_dist jumps up by a lot (respawn), reset baseline & warmup automatically.
- Fall penalty bugfix: apply after initializing reward_val, so it isn’t overwritten.
- **Robust fall detection**: combine info/event markers *and/or* position heuristics
  (axis threshold / one-step drop). Penalty can apply immediately or only on terminal.
"""

import argparse
import json
import random
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Optional dependencies for OCR-based completion detection
try:
    import pytesseract  # type: ignore
except ImportError:
    pytesseract = None  # type: ignore

try:
    import pyautogui  # type: ignore
except ImportError:
    pyautogui = None  # type: ignore

if __package__ in (None, ""):
    import sys
    pkg_root = Path(__file__).resolve().parent
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from neon_white_rl.env import NeonWhiteConfig, NeonWhiteEnv
    from neon_white_rl.ppo import PPOConfig, ActorCritic, RolloutBuffer
else:
    from .neon_white_rl.env import NeonWhiteConfig, NeonWhiteEnv
    from .neon_white_rl.ppo import PPOConfig, ActorCritic, RolloutBuffer

# Observation keys used to flatten the observation dictionary.
OBS_KEYS: List[str] = [
    "pos",
    "vel",
    "yaw",
    "goal_dist",
    "goal_dir",
    "grounded",
    "surface",
    "height_gap",
    "time",
    "stage",
    "nearest_enemy_dist",
    "nearest_enemy_dir",
    "enemies_n",
]

# Action space dims
CONTINUOUS_COMPONENT = 4  # move_x, move_y, look_x, look_y
DISCRETE_COMPONENT = 4    # jump, shoot, use, reset
LOOK_SCALE = 100.0


# ===== Deprecated pyautogui helpers (guardrails) =====
def ensure_pyautogui():
    raise RuntimeError("pyautogui is not used by this script")

def press_sequence(pyauto, keys, press_duration=0.08, gap=0.12):
    raise RuntimeError("press_sequence is not used; resets are handled via env.step")

def send_start_pulse(pyauto):
    raise RuntimeError("send_start_pulse is not used; use send_start_action instead")

def send_reset_sequence(pyauto):
    raise RuntimeError("send_reset_sequence is not used; use send_reset_action instead")


# ================== Env interaction via RL bridge =================
def send_start_action(env: NeonWhiteEnv) -> None:
    action = {"move": [0.0, 0.0], "look": [0.0, 0.0], "jump": True, "shoot": False, "use": False, "reset": False}
    env.step(action)
    time.sleep(0.1)

def send_reset_action(env: NeonWhiteEnv) -> None:
    action = {"move": [0.0, 0.0], "look": [0.0, 0.0], "jump": False, "shoot": False, "use": False, "reset": True}
    env.step(action)
    time.sleep(0.1)


# ============================== Action toggles ================================
@dataclass
class ActionToggles:
    move_x: bool = True
    move_y: bool = True
    look_x: bool = True
    look_y: bool = True
    jump: bool = True
    shoot: bool = True
    use: bool = True
    reset: bool = False

@dataclass
class ActionForces:
    move_x: Optional[float] = None
    move_y: Optional[float] = None
    look_x: Optional[float] = None
    look_y: Optional[float] = None
    jump: Optional[bool] = None
    shoot: Optional[bool] = None
    use: Optional[bool] = None

    def is_active(self) -> bool:
        return any(v is not None for v in (
            self.move_x, self.move_y, self.look_x, self.look_y,
            self.jump, self.shoot, self.use
        ))


def apply_action_mask(continuous: torch.Tensor, discrete: torch.Tensor, mask: ActionToggles) -> Tuple[torch.Tensor, torch.Tensor]:
    cont = continuous.clone()
    disc = discrete.clone()
    if not mask.move_x: cont[..., 0] = 0.0
    if not mask.move_y: cont[..., 1] = 0.0
    if not mask.look_x: cont[..., 2] = 0.0
    if not mask.look_y: cont[..., 3] = 0.0
    if not mask.jump:   disc[..., 0] = 0.0
    if not mask.shoot:  disc[..., 1] = 0.0
    if not mask.use:    disc[..., 2] = 0.0
    if not mask.reset:  disc[..., 3] = 0.0
    return cont, disc

def apply_action_forces(continuous: torch.Tensor, discrete: torch.Tensor, forces: ActionForces) -> Tuple[torch.Tensor, torch.Tensor]:
    if not forces.is_active():
        return continuous, discrete
    cont = continuous
    disc = discrete
    cont_mut, disc_mut = False, False

    def ensure_cont():
        nonlocal cont, cont_mut
        if not cont_mut:
            cont = cont.clone(); cont_mut = True
        return cont

    def ensure_disc():
        nonlocal disc, disc_mut
        if not disc_mut:
            disc = disc.clone(); disc_mut = True
        return disc

    if forces.move_x is not None: ensure_cont()[..., 0] = float(np.clip(forces.move_x, -1.0, 1.0))
    if forces.move_y is not None: ensure_cont()[..., 1] = float(np.clip(forces.move_y, -1.0, 1.0))
    if forces.look_x is not None: ensure_cont()[..., 2] = float(np.clip(forces.look_x, -1.0, 1.0))
    if forces.look_y is not None: ensure_cont()[..., 3] = float(np.clip(forces.look_y, -1.0, 1.0))
    if forces.jump  is not None:  ensure_disc()[..., 0] = 1.0 if bool(forces.jump) else 0.0
    if forces.shoot is not None:  ensure_disc()[..., 1] = 1.0 if bool(forces.shoot) else 0.0
    if forces.use   is not None:  ensure_disc()[..., 2] = 1.0 if bool(forces.use) else 0.0
    return cont, disc


# ============================ JSON level config ===============================
def parse_vec3(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected vec3 'x,y,z'")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        raise argparse.ArgumentTypeError("Vector must be numeric 'x,y,z'")

def load_level_db(path: Path) -> Dict:
    if not path.exists():
        return {"levels": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "levels" not in data or not isinstance(data["levels"], dict):
            raise ValueError("Invalid JSON schema: missing 'levels' dict")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load level config {path}: {e}")

def save_level_db(path: Path, db: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f"[level-config] Saved to {path}")

def upsert_level_entry(
    db: Dict,
    level_id: str,
    name: Optional[str],
    goal: Optional[Tuple[float, float, float]],
    goal_radius: Optional[float],
    enemies: Optional[List[Tuple[float, float, float]]],
    overwrite: bool,
) -> None:
    levels = db.setdefault("levels", {})
    entry = levels.get(level_id, {})
    if entry and not overwrite:
        if name is not None: entry["name"] = name
        if goal is not None: entry["goal"] = list(goal)
        if goal_radius is not None: entry["goal_radius"] = float(goal_radius)
        if enemies is not None: entry["enemies"] = [list(e) for e in enemies]
        levels[level_id] = entry
        return
    levels[level_id] = {
        "name": name or level_id,
        "goal": list(goal) if goal is not None else [0.0, 0.0, 0.0],
        "goal_radius": float(goal_radius) if goal_radius is not None else 1.0,
        "enemies": [list(e) for e in (enemies or [])],
    }

def compute_dir_and_dist(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    v = (dst - src).astype(np.float32)
    d = float(np.linalg.norm(v))
    if d > 1e-6:
        return (v / d), d
    return np.zeros_like(v), 0.0

def apply_level_overrides_inplace(obs: Dict[str, np.ndarray], level_spec: Dict) -> None:
    """
    Overrides goal/enemy observation fields based on a level_spec:
      - obs['goal_dist'] (scalar)
      - obs['goal_dir']  (vec3)
      - obs['nearest_enemy_dist'] (scalar)
      - obs['nearest_enemy_dir']  (vec3)
      - obs['enemies_n'] (scalar)
    Expects level_spec['goal'] and level_spec['enemies'] to already be in WORLD coordinates.
    """
    if "pos" not in obs:
        return
    pos = np.asarray(obs["pos"], dtype=np.float32).reshape(-1)
    goal = np.asarray(level_spec.get("goal", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    goal_dir, goal_dist = compute_dir_and_dist(pos, goal)
    obs["goal_dir"] = goal_dir.astype(np.float32)
    obs["goal_dist"] = np.array([goal_dist], dtype=np.float32)

    enemies = np.asarray(level_spec.get("enemies", []), dtype=np.float32).reshape(-1, 3) if level_spec.get("enemies") else np.zeros((0,3), dtype=np.float32)
    if enemies.shape[0] == 0:
        obs["nearest_enemy_dir"] = np.zeros(3, dtype=np.float32)
        obs["nearest_enemy_dist"] = np.array([0.0], dtype=np.float32)
        obs["enemies_n"] = np.array([0.0], dtype=np.float32)
        return

    deltas = enemies - pos[None, :]
    dists = np.linalg.norm(deltas, axis=1)
    idx = int(np.argmin(dists))
    nearest_delta = deltas[idx]
    nearest_dist = float(dists[idx])
    if nearest_dist > 1e-6:
        nearest_dir = (nearest_delta / nearest_dist).astype(np.float32)
    else:
        nearest_dir = np.zeros(3, dtype=np.float32)

    obs["nearest_enemy_dir"] = nearest_dir
    obs["nearest_enemy_dist"] = np.array([nearest_dist], dtype=np.float32)
    obs["enemies_n"] = np.array([float(enemies.shape[0])], dtype=np.float32)


# ========================= Coordinate transform utils =========================
def reorder(v: Sequence[float], order: str) -> np.ndarray:
    idx = {"x": 0, "y": 1, "z": 2}
    return np.array([v[idx[order[0]]], v[idx[order[1]]], v[idx[order[2]]]], dtype=np.float32)

@dataclass
class CoordXform:
    order: str = "xyz"                 # e.g., "xzy" to swap Y/Z from JSON to world
    negate_x: bool = False
    negate_y: bool = False
    negate_z: bool = False
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0

    def apply(self, v_xyz: Sequence[float]) -> np.ndarray:
        v = np.array(v_xyz, dtype=np.float32)
        v = reorder(v, self.order)
        if self.negate_x: v[0] = -v[0]
        if self.negate_y: v[1] = -v[1]
        if self.negate_z: v[2] = -v[2]
        v = v * float(self.scale)
        v = v + np.array(self.offset, dtype=np.float32)
        return v

def transform_level_spec(level_spec: Dict, xf: CoordXform) -> Dict:
    out = dict(level_spec)
    if "goal" in level_spec:
        out["goal"] = xf.apply(level_spec["goal"]).tolist()
    if "enemies" in level_spec and level_spec["enemies"]:
        out["enemies"] = [xf.apply(e).tolist() for e in level_spec["enemies"]]
    return out


# ================================ Utilities ===================================
def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "t", "yes", "y", "1"}: return True
    if lowered in {"false", "f", "no", "n", "0"}: return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got '{value}'")

def flatten_observation(obs: Dict[str, np.ndarray]) -> np.ndarray:
    pieces: List[np.ndarray] = []
    for key in OBS_KEYS:
        value = obs.get(key)
        if value is None:
            raise KeyError(f"Missing observation key '{key}' from bridge payload")
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        pieces.append(arr)
    return np.concatenate(pieces, axis=0)


# ===================== OCR (optional, throttled) =========================
def parse_bbox(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    parts = value.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Bounding box must be 'x,y,w,h'")
    try:
        x, y, w, h = map(int, parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Bounding box coordinates must be integers")
    return (x, y, w, h)

def detect_level_complete_text(bbox: Optional[Tuple[int, int, int, int]]) -> bool:
    if pytesseract is None:
        raise RuntimeError("pytesseract not available")
    if pyautogui is None:
        raise RuntimeError("pyautogui not available")
    try:
        screenshot = pyautogui.screenshot(region=bbox) if bbox is not None else pyautogui.screenshot()
    except Exception as e:
        raise RuntimeError(f"Failed to capture screenshot: {e}")
    try:
        text = pytesseract.image_to_string(screenshot).lower()
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}")
    return "level complete" in text


# ===================== Robust fall detection helpers =====================
def _pos_axis(o: Dict[str, np.ndarray], axis_idx: int) -> Optional[float]:
    p = o.get("pos")
    if p is None:
        return None
    arr = np.asarray(p, dtype=np.float32).reshape(-1)
    if arr.size < axis_idx + 1:
        return None
    return float(arr[axis_idx])

def detect_fall_event(
    args: argparse.Namespace,
    info: object,
    prev_axis_val: Optional[float],
    curr_axis_val: Optional[float],
    terminated: bool,
    truncated: bool,
) -> Tuple[bool, str]:
    """
    Returns (fell, why). 'fell' is True if we consider this step a fall.
    'why' is a short label for logging ('info', 'pos_threshold', 'pos_drop_delta', ...).
    """
    why = ""

    # (A) INFO/EVENT-based detection
    fell_info = False
    if args.fall_detect in ("auto", "info"):
        fall_markers = {"fall", "void", "out_of_bounds", "kill_plane", "pit", "fell", "oob", "killvolume", "kill volume"}
        reason_lc = ""
        death_event_flag = False
        if isinstance(info, dict):
            raw_info = info.get("raw_obs")
            if isinstance(raw_info, dict):
                reason_lc = str(raw_info.get("death_reason") or "").lower()
                if raw_info.get("done") and not raw_info.get("reached", False):
                    death_event_flag = True
            # other channels
            reason_lc = (reason_lc or str(info.get("death_reason") or "")).lower()
            for evt in (info.get("events") or []):
                if isinstance(evt, dict) and (evt.get("name") == "death" or evt.get("type") == "death"):
                    death_event_flag = True
                    payload = str(evt.get("payload") or "")
                    if not reason_lc:
                        reason_lc = payload.lower()

        if ("death" in reason_lc) or any(m in reason_lc for m in fall_markers) or death_event_flag:
            fell_info = True
            why = why or "info"

    # (B) POS-based detection (threshold or one-step big drop)
    fell_pos = False
    if args.fall_detect in ("auto", "pos"):
        if curr_axis_val is not None and args.fall_y_threshold is not None:
            if curr_axis_val <= float(args.fall_y_threshold):
                fell_pos = True
                why = why or "pos_threshold"
        if (not fell_pos and args.fall_drop_delta is not None
            and args.fall_drop_delta > 0
            and prev_axis_val is not None and curr_axis_val is not None):
            if (prev_axis_val - curr_axis_val) >= float(args.fall_drop_delta):
                fell_pos = True
                why = why or "pos_drop_delta"

    fell = fell_info or fell_pos

    # Conservative inference on terminal if needed (disabled by default)
    # if (not fell) and (terminated or truncated) and isinstance(info, dict):
    #     events = {e.get("type") for e in info.get("events", []) if isinstance(e, dict)}
    #     if not (("timeout" in events) or ("stuck" in events) or info.get("level_complete")):
    #         fell = True
    #         why = why or "terminal_inferred"

    return fell, why


# ============================ Action conversion ===============================
def continuous_action_to_env(action: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    action = action.detach().cpu().numpy()
    move = action[..., :2]
    look = action[..., 2:] * LOOK_SCALE
    return move, look

def build_action_dictionary(continuous: torch.Tensor, discrete: torch.Tensor) -> Dict[str, object]:
    move, look = continuous_action_to_env(continuous)
    disc_np = discrete.detach().cpu().numpy()
    return {
        "move": move.tolist(),
        "look": look.tolist(),
        "jump": bool(disc_np[0] > 0.5),
        "shoot": bool(disc_np[1] > 0.5),
        "use": bool(disc_np[2] > 0.5),
        "reset": bool(disc_np[3] > 0.5),
    }


# ============================ Logging & metrics ===============================
def make_writer(log_dir: Path, run_name: str) -> SummaryWriter:
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(run_dir))

def explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.numel() == 0:
        return 0.0
    var_y = torch.var(y_true)
    if var_y.item() == 0:
        return 0.0
    return 1.0 - torch.var(y_true - y_pred) / var_y

def save_checkpoint(
    directory: Path,
    run_name: str,
    step: int,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{run_name}_step_{step}.pt"
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ================================= Training ===================================
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    # Load / mutate level JSON if requested
    level_db = None
    level_spec_world = None
    selected_level_id = None
    level_goal_radius = None

    if args.use_level_config:
        cfg_path = Path(args.level_config).expanduser().resolve()
        level_db = load_level_db(cfg_path)

        if args.add_level:
            if not args.level_id:
                raise RuntimeError("--add-level requires --level-id")
            goal = parse_vec3(args.goal) if args.goal else None
            enemies = [parse_vec3(e) for e in (args.enemy or [])]
            upsert_level_entry(
                level_db,
                level_id=args.level_id,
                name=args.level_name,
                goal=goal,
                goal_radius=args.goal_radius if args.goal_radius is not None else None,
                enemies=enemies if enemies else None,
                overwrite=args.overwrite_level,
            )
            save_level_db(cfg_path, level_db)

        if not args.level_id:
            raise RuntimeError("--use-level-config requires --level-id to choose a level entry")
        selected_level_id = args.level_id
        level_spec_raw = level_db.get("levels", {}).get(selected_level_id)
        if level_spec_raw is None:
            raise RuntimeError(f"Level id '{selected_level_id}' not found in {cfg_path}")
        level_goal_radius = float(level_spec_raw.get("goal_radius", args.goal_radius or 1.0))

        # Build coordinate transform and produce WORLD-space spec
        coord_xf = CoordXform(
            order=args.coord_order,
            negate_x=args.negate_x,
            negate_y=args.negate_y,
            negate_z=args.negate_z,
            offset=args.coord_offset or (0.0, 0.0, 0.0),
            scale=args.coord_scale,
        )
        level_spec_world = transform_level_spec(level_spec_raw, coord_xf)

        print(f"[level-config] Using '{selected_level_id}' (name='{level_spec_raw.get('name', selected_level_id)}') "
              f"goal(raw)={level_spec_raw.get('goal')} -> goal(world)={level_spec_world.get('goal')} "
              f"radius={level_goal_radius} enemies={len(level_spec_world.get('enemies', []))} "
              f"order={coord_xf.order} neg=({coord_xf.negate_x},{coord_xf.negate_y},{coord_xf.negate_z}) "
              f"offset={coord_xf.offset} scale={coord_xf.scale}")

    env_config = NeonWhiteConfig(
        host=args.host,
        port=args.port,
        action_rate_hz=args.action_rate_hz,
        default_stage=args.stage if args.stage is not None else 0,
        default_timescale=args.timescale,
        wait_for_ready=not args.no_wait,
    )
    env = NeonWhiteEnv(config=env_config)
    control_enabled = args.control_enabled

    action_mask = ActionToggles(
        move_x=args.allow_move_x,
        move_y=args.allow_move_y,
        look_x=args.allow_look_x,
        look_y=args.allow_look_y,
        jump=args.allow_jump,
        shoot=args.allow_shoot,
        use=args.allow_use,
    )
    action_forces = ActionForces(
        move_x=args.force_move_x,
        move_y=args.force_move_y,
        look_x=args.force_look_x,
        look_y=args.force_look_y,
        jump=args.force_jump,
        shoot=args.force_shoot,
        use=args.force_use,
    )
    forces_active = action_forces.is_active()

    reset_options: Dict[str, object] = {}
    if args.level:
        reset_options["level"] = args.level
    if args.stage is not None:
        reset_options["stage"] = args.stage
    reset_options["timescale"] = args.timescale

    # Initial reset
    obs_sample, info = env.reset(options=reset_options if reset_options else None)

    # Avoid reloading the same scene on subsequent resets
    if "level" in reset_options:
        reset_options.pop("level")

    # Apply level-config observation overrides on first obs
    if level_spec_world is not None:
        apply_level_overrides_inplace(obs_sample, level_spec_world)

    if control_enabled:
        send_start_action(env)

    stuck_seconds = max(0.0, args.stuck_seconds)
    stuck_distance = max(0.0, args.stuck_distance)
    pos_init = obs_sample.get('pos')
    if isinstance(pos_init, np.ndarray):
        last_motion_pos = pos_init.astype(np.float32, copy=False).copy()
    elif pos_init is not None:
        last_motion_pos = np.asarray(pos_init, dtype=np.float32)
    else:
        last_motion_pos = np.zeros(3, dtype=np.float32)
    last_motion_time = time.time()

    # Helpers to read goal_dist safely
    def get_goal_dist_scalar(o: Dict[str, np.ndarray]) -> Optional[float]:
        gd = o.get("goal_dist", None)
        if isinstance(gd, (float, int, np.floating, np.integer)):
            return float(gd)
        if isinstance(gd, np.ndarray) and gd.size > 0:
            return float(gd.reshape(-1)[0])
        return None

    # ===== Progress shaping state (baseline & warmup) =====
    last_progress_dist: Optional[float] = get_goal_dist_scalar(obs_sample) if args.enable_progress_reward else None
    progress_warmup_left: int = int(args.progress_warmup_steps) if args.enable_progress_reward else 0

    # ===== Fall detection state =====
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.fall_axis]
    prev_axis_val = _pos_axis(obs_sample, axis_idx)
    fall_pending = False
    fall_applied_this_ep = False

    obs_vector = flatten_observation(obs_sample)
    obs_dim = int(obs_vector.shape[0])

    policy = ActorCritic(
        obs_dim=obs_dim,
        continuous_dim=CONTINUOUS_COMPONENT,
        discrete_dim=DISCRETE_COMPONENT,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    # Optional resume
    loaded_step: int = 0
    if args.load_checkpoint is not None:
        ckpt_path = Path(args.load_checkpoint).expanduser().resolve()
        if ckpt_path.is_file():
            try:
                payload = torch.load(str(ckpt_path), map_location=device)
                model_state = payload.get("model_state")
                opt_state = payload.get("optimizer_state")
                step_val = payload.get("step")
                if model_state is not None:
                    policy.load_state_dict(model_state)
                if opt_state is not None:
                    optimizer.load_state_dict(opt_state)
                if isinstance(step_val, int):
                    loaded_step = step_val
                print(f"[train] Loaded checkpoint from {ckpt_path} (step={loaded_step}).")
            except Exception as e:
                print(f"[train] Failed to load checkpoint from {ckpt_path}: {e}")
        else:
            print(f"[train] Warning: checkpoint {ckpt_path} not found; starting fresh.")

    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )

    # Logging & buffers
    log_root = Path(args.log_dir).expanduser().resolve()
    run_dir = log_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir or (run_dir / "checkpoints")).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    rollout_buffer = RolloutBuffer(
        num_steps=cfg.rollout_steps,
        obs_dim=obs_dim,
        cont_dim=CONTINUOUS_COMPONENT,
        disc_dim=DISCRETE_COMPONENT,
        device=device,
    )

    episode_rewards: deque[float] = deque(maxlen=50)
    episode_lengths: deque[int] = deque(maxlen=50)
    current_reward = 0.0
    current_length = 0

    global_step = loaded_step
    start_time = time.time()
    obs_tensor = torch.tensor(obs_vector, device=device, dtype=torch.float32)
    episode_start_wall = time.time()

    # OCR throttling state
    last_ocr_check_step = -10**9
    last_ocr_check_time = 0.0

    # Geo debug print throttle
    last_geo_print = 0.0

    action_repeat = max(1, args.action_repeat)
    repeat_steps_remaining = 0
    cached_cont_action: Optional[torch.Tensor] = None
    cached_disc_action: Optional[torch.Tensor] = None

    stop_requested = False
    last_done = 0.0

    def signal_handler(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        while global_step < cfg.total_steps and not stop_requested:
            rollout_buffer.reset()

            for _ in range(cfg.rollout_steps):
                elapsed = time.time() - episode_start_wall
                timed_out = args.episode_seconds > 0 and elapsed >= args.episode_seconds

                use_cached = (
                    action_repeat > 1
                    and repeat_steps_remaining > 0
                    and cached_cont_action is not None
                    and cached_disc_action is not None
                )

                if use_cached:
                    cont_action = cached_cont_action.clone()
                    disc_action = cached_disc_action.clone()
                    repeat_steps_remaining -= 1
                else:
                    with torch.no_grad():
                        cont_action_sample, disc_action_sample, _, _, _ = policy.act(
                            obs_tensor.unsqueeze(0)
                        )
                    cont_action = cont_action_sample.squeeze(0)
                    disc_action = disc_action_sample.squeeze(0)
                    repeat_steps_remaining = max(action_repeat - 1, 0)

                cont_action, disc_action = apply_action_mask(cont_action, disc_action, action_mask)
                if forces_active:
                    cont_action, disc_action = apply_action_forces(cont_action, disc_action, action_forces)

                with torch.no_grad():
                    log_prob_eval, _, value_eval, _ = policy.evaluate_actions(
                        obs_tensor.unsqueeze(0),
                        cont_action.unsqueeze(0),
                        disc_action.unsqueeze(0),
                    )
                log_prob = log_prob_eval.squeeze(0)
                value_scalar = value_eval.squeeze(0)

                cached_cont_action = cont_action.clone()
                cached_disc_action = disc_action.clone()

                action_dict = build_action_dictionary(cont_action, disc_action)
                next_obs, env_reward_raw, terminated, truncated, info = env.step(action_dict)

                # Apply JSON overrides to new observation (goal/enemy fields)
                if level_spec_world is not None:
                    apply_level_overrides_inplace(next_obs, level_spec_world)

                # Optional geo debug print for calibration
                if args.print_pos and level_spec_world is not None and (time.time() - last_geo_print) > 0.5:
                    p = next_obs.get("pos")
                    if p is not None:
                        p = np.asarray(p, dtype=np.float32).reshape(-1)
                        g = np.asarray(level_spec_world["goal"], dtype=np.float32).reshape(-1)
                        dist = float(np.linalg.norm(p - g))
                        print(f"[geo] pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) "
                              f"goal=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f}) dist={dist:.2f} r={level_goal_radius:.2f}")
                        last_geo_print = time.time()

                # Stuck/time/death handling (events)
                current_pos = next_obs.get('pos')
                if isinstance(current_pos, np.ndarray):
                    current_pos = current_pos.astype(np.float32, copy=False)
                elif current_pos is not None:
                    current_pos = np.asarray(current_pos, dtype=np.float32)
                if current_pos is not None:
                    if np.linalg.norm(current_pos - last_motion_pos) > stuck_distance:
                        last_motion_pos = current_pos.copy()
                        last_motion_time = time.time()

                death_reason = None
                death_flag = False
                raw_info = None
                if isinstance(info, dict):
                    raw_info = info.get('raw_obs')
                    if isinstance(raw_info, dict):
                        death_reason = raw_info.get('death_reason')
                        if raw_info.get('done') and not raw_info.get('reached', False):
                            death_flag = True
                    for evt in (info.get('events') or []):
                        if isinstance(evt, dict) and evt.get('name') == 'death':
                            death_flag = True
                            death_reason = death_reason or evt.get('payload')

                if control_enabled:
                    if death_flag and not (terminated or truncated):
                        truncated = True
                        terminated = False
                        if not isinstance(info, dict):
                            info = {}
                        else:
                            info = dict(info)
                        if isinstance(raw_info, dict):
                            info['raw_obs'] = dict(raw_info)
                        info['death_reason'] = death_reason or info.get('death_reason') or 'death'

                    if stuck_seconds > 0 and not (terminated or truncated):
                        if time.time() - last_motion_time >= stuck_seconds:
                            truncated = True
                            terminated = False
                            if not isinstance(info, dict):
                                info = {}
                            else:
                                info = dict(info)
                            events = list(info.get('events', ()))
                            events.append({'type': 'stuck'})
                            info['events'] = events

                    if timed_out and not (terminated or truncated):
                        truncated = True
                        if isinstance(info, dict):
                            info = dict(info)
                            events = list(info.get('events', ()))
                            events.append({'type': 'timeout'})
                            info['events'] = events

                # --------- Initialize reward from env scale ----------
                reward_val = float(args.reward_env_scale) * float(env_reward_raw)

                # ======= Robust fall detection (info + position) =======
                curr_axis_val = _pos_axis(next_obs, axis_idx)
                fell_now, fell_why = detect_fall_event(
                    args=args,
                    info=info,
                    prev_axis_val=prev_axis_val,
                    curr_axis_val=curr_axis_val,
                    terminated=terminated,
                    truncated=truncated,
                )
                prev_axis_val = curr_axis_val
                # =======================================================

                # ----- Progress shaping (baseline/warmup + jump reset) -----
                progress_delta = 0.0
                if args.enable_progress_reward:
                    curr_dist = get_goal_dist_scalar(next_obs)

                    if curr_dist is not None and last_progress_dist is not None:
                        jump_amt = curr_dist - last_progress_dist

                        # 1) Big jump away from goal -> treat as respawn (apply penalty once)
                        if args.progress_jump_reset > 0.0 and jump_amt > float(args.progress_jump_reset):
                            if args.fall_penalty > 0 and not fall_applied_this_ep:
                                reward_val -= float(args.fall_penalty)
                                fall_applied_this_ep = True
                                writer.add_scalar("events/fall", 1, global_step)
                                writer.add_scalar("rewards/fall_penalty", -float(args.fall_penalty), global_step)
                                if args.debug_reward:
                                    print(f"[debug_reward] jump-reset fall penalty applied: -{args.fall_penalty} (Δdist=+{jump_amt:.2f})")
                            # re-baseline & warmup after respawn
                            last_progress_dist = curr_dist
                            progress_warmup_left = max(progress_warmup_left, int(args.progress_warmup_steps))

                        # 2) Normal per-step progress
                        elif progress_warmup_left <= 0 and not (terminated or truncated):
                            progress_delta = float(last_progress_dist - curr_dist)

                            cap = float(args.progress_delta_cap) if args.progress_delta_cap and args.progress_delta_cap > 0 \
                                else float(args.reward_progress_clip or 0.0)
                            if args.reward_progress_nonneg:
                                progress_delta = max(progress_delta, 0.0)
                            if cap > 0.0:
                                progress_delta = float(np.clip(progress_delta, -cap, cap))

                            scale = float(args.reward_progress_scale if args.progress_scale is None else args.progress_scale)
                            reward_val += scale * progress_delta

                        # Always update the baseline when we have a reading
                        last_progress_dist = curr_dist

                    if progress_warmup_left > 0:
                        progress_warmup_left -= 1
                # -----------------------------------------------------------


                # Constant per-step term (can be negative = time penalty)
                reward_val += float(args.reward_step)

                # Penalties on events (applied once at this step)
                if death_flag:
                    reward_val += float(args.reward_death_penalty)
                if isinstance(info, dict):
                    if any(evt.get("type") == "stuck" for evt in info.get("events", [])):
                        reward_val += float(args.reward_stuck_penalty)
                    if any(evt.get("type") == "timeout" for evt in info.get("events", [])):
                        reward_val += float(args.reward_timeout_penalty)

                # ======= Apply fall penalty (single-source of truth) =======
                if args.fall_apply_on_terminal:
                    if fell_now:
                        fall_pending = True
                    if (terminated or truncated) and fall_pending and (not fall_applied_this_ep):
                        reward_val -= float(args.fall_penalty)
                        fall_applied_this_ep = True
                        fall_pending = False
                        writer.add_scalar("events/fall", 1, global_step)
                        writer.add_scalar("rewards/fall_penalty", -float(args.fall_penalty), global_step)
                        if args.debug_reward:
                            print(f"[debug_reward] FALL(-{args.fall_penalty:.1f}) applied on terminal ({fell_why})")
                else:
                    if fell_now and (not fall_applied_this_ep):
                        reward_val -= float(args.fall_penalty)
                        fall_applied_this_ep = True
                        writer.add_scalar("events/fall", 1, global_step)
                        writer.add_scalar("rewards/fall_penalty", -float(args.fall_penalty), global_step)
                        if args.debug_reward:
                            print(f"[debug_reward] FALL(-{args.fall_penalty:.1f}) applied immediately ({fell_why})")
                # ===========================================================

                # ----------------- JSON goal-based completion -----------------
                if level_spec_world is not None and not (terminated or truncated):
                    dist_val = get_goal_dist_scalar(next_obs)
                    if dist_val is not None and dist_val <= level_goal_radius:
                        reward_val += args.level_complete_reward
                        terminated = True
                        if not isinstance(info, dict):
                            info = {}
                        info['level_complete'] = True
                        info['completion_source'] = "json_goal_radius"
                        info['finish_dist'] = dist_val

                # ----------------- Optional OCR completion (throttled) --------
                if args.detect_level_text and not (terminated or truncated):
                    gd = get_goal_dist_scalar(next_obs) if level_spec_world is not None else None
                    near_goal_ok = (
                        args.ocr_goal_dist <= 0.0
                        or (gd is not None and gd <= float(args.ocr_goal_dist))
                    )
                    step_ok = (args.ocr_interval_steps <= 0) or ((global_step - last_ocr_check_step) >= int(args.ocr_interval_steps))
                    time_ok = (args.ocr_interval_seconds <= 0.0) or ((time.time() - last_ocr_check_time) >= float(args.ocr_interval_seconds))
                    if near_goal_ok and step_ok and time_ok:
                        try:
                            if detect_level_complete_text(args.level_complete_bbox):
                                reward_val += args.level_complete_reward
                                terminated = True
                                if not isinstance(info, dict):
                                    info = {}
                                info['level_complete'] = True
                                info['completion_source'] = "ocr"
                        except Exception as e:
                            if global_step == 0:
                                print(f"[train] Warning: OCR detection failed: {e}")
                        finally:
                            last_ocr_check_step = global_step
                            last_ocr_check_time = time.time()

                # reward debug + telemetry
                curr_goal_dist = get_goal_dist_scalar(next_obs)
                if curr_goal_dist is not None:
                    writer.add_scalar("charts/json_goal_dist", curr_goal_dist, global_step)
                writer.add_scalar("charts/json_progress", progress_delta, global_step)

                if args.debug_reward:
                    done_flag = bool(terminated or truncated)
                    events = info.get('events') if isinstance(info, dict) else None
                    print(f"[debug_reward] step={global_step} env={float(env_reward_raw):+.4f} "
                          f"prog={progress_delta:+.4f} total={reward_val:+.4f} done={done_flag} events={events}")

                done_val = float(terminated or truncated)

                # Store step
                rollout_buffer.add(
                    obs=obs_tensor,
                    cont_action=cont_action,
                    disc_action=disc_action,
                    log_prob=log_prob,
                    value=value_scalar,
                    reward=reward_val,
                    done=done_val,
                )

                # update trackers
                current_reward += reward_val
                current_length += 1
                global_step += 1

                if terminated or truncated:
                    episode_rewards.append(current_reward)
                    episode_lengths.append(current_length)
                    if control_enabled:
                        send_reset_action(env)
                        next_obs, info = env.reset(options=reset_options if reset_options else None)
                        time.sleep(0.2)
                    else:
                        stop_requested = True
                        break

                    # Reapply JSON overrides + reset progress baseline & warmup
                    if level_spec_world is not None:
                        apply_level_overrides_inplace(next_obs, level_spec_world)
                    if args.enable_progress_reward:
                        last_progress_dist = get_goal_dist_scalar(next_obs)
                        progress_warmup_left = int(args.progress_warmup_steps)

                    # Reset fall detector for new episode
                    fall_pending = False
                    fall_applied_this_ep = False
                    prev_axis_val = _pos_axis(next_obs, axis_idx)

                    pos_after = next_obs.get('pos')
                    if isinstance(pos_after, np.ndarray):
                        last_motion_pos = pos_after.astype(np.float32, copy=False).copy()
                    elif pos_after is not None:
                        last_motion_pos = np.asarray(pos_after, dtype=np.float32)
                    last_motion_time = time.time()
                    episode_start_wall = time.time()
                    obs_tensor = torch.tensor(flatten_observation(next_obs), device=device, dtype=torch.float32)
                    repeat_steps_remaining = 0
                    cached_cont_action = None
                    cached_disc_action = None
                    current_reward = 0.0
                    current_length = 0
                    last_done = done_val
                else:
                    obs_tensor = torch.tensor(flatten_observation(next_obs), device=device, dtype=torch.float32)
                    last_done = done_val

                if isinstance(info, dict) and info.get("escape_requested"):
                    stop_requested = True
                    break

                if global_step >= cfg.total_steps or stop_requested:
                    break

            if stop_requested:
                break
            if rollout_buffer.ptr == 0:
                continue

            with torch.no_grad():
                _, _, value = policy.get_dists(obs_tensor.unsqueeze(0))
                last_value = value.squeeze(0)

            rollout_buffer.compute_returns_and_advantages(last_value, last_done, cfg.gamma, cfg.gae_lambda)

            advantages = rollout_buffer.advantages[: rollout_buffer.ptr]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            rollout_buffer.advantages[: rollout_buffer.ptr] = advantages

            losses: Dict[str, float] = {}
            clip_fracs: List[float] = []
            approx_kls: List[float] = []

            for epoch in range(cfg.update_epochs):
                for batch in rollout_buffer.iter_minibatches(cfg.minibatch_size):
                    obs_batch = batch["obs"]
                    cont_batch = batch["cont_actions"]
                    disc_batch = batch["disc_actions"]
                    logprob_batch = batch["log_probs"]
                    adv_batch = batch["advantages"]
                    return_batch = batch["returns"]
                    value_batch = batch["values"]

                    new_log_prob, entropy, new_values, _ = policy.evaluate_actions(
                        obs_batch, cont_batch, disc_batch
                    )
                    log_ratio = new_log_prob - logprob_batch
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).abs().mean().item()
                        approx_kls.append(approx_kl)

                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_batch
                    policy_loss = -torch.min(surr1, surr2).mean()

                    clip_frac = (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean().item()
                    clip_fracs.append(clip_frac)

                    value_loss = 0.5 * F.mse_loss(new_values, return_batch)
                    entropy_loss = entropy.mean()

                    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                if cfg.target_kl is not None and len(approx_kls) > 0:
                    if np.mean(approx_kls) > cfg.target_kl:
                        break

            losses["policy_loss"] = policy_loss.detach().cpu().item()
            losses["value_loss"] = value_loss.detach().cpu().item()
            losses["entropy"] = entropy_loss.detach().cpu().item()
            losses["clip_fraction"] = float(np.mean(clip_fracs)) if clip_fracs else 0.0
            losses["approx_kl"] = float(np.mean(approx_kls)) if approx_kls else 0.0

            writer.add_scalar("charts/global_step", global_step, global_step)
            writer.add_scalar("loss/policy", losses["policy_loss"], global_step)
            writer.add_scalar("loss/value", losses["value_loss"], global_step)
            writer.add_scalar("loss/entropy", losses["entropy"], global_step)
            writer.add_scalar("diagnostics/clip_fraction", losses["clip_fraction"], global_step)
            writer.add_scalar("diagnostics/approx_kl", losses["approx_kl"], global_step)

            with torch.no_grad():
                ev = explained_variance(
                    rollout_buffer.returns[: rollout_buffer.ptr],
                    rollout_buffer.values[: rollout_buffer.ptr],
                )
            writer.add_scalar("diagnostics/explained_variance", ev, global_step)

            if episode_rewards:
                writer.add_scalar("charts/episode_reward", np.mean(episode_rewards), global_step)
                writer.add_scalar("charts/episode_length", np.mean(episode_lengths), global_step)

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

            elapsed_total = time.time() - start_time
            fps = int(global_step / max(elapsed_total, 1e-6))
            writer.add_scalar("charts/fps", fps, global_step)

            if global_step >= cfg.total_steps or stop_requested:
                break

            if args.checkpoint_interval and (global_step // cfg.rollout_steps) % args.checkpoint_interval == 0:
                save_checkpoint(checkpoint_dir, args.run_name, global_step, policy, optimizer, cfg)

        path = save_checkpoint(checkpoint_dir, args.run_name, global_step, policy, optimizer, cfg)
        if stop_requested:
            print(f"[train] Stop requested. Saved checkpoint to {path}")
        else:
            print(f"[train] Training complete. Final checkpoint saved to {path}")

    finally:
        signal.signal(signal.SIGINT, original_handler)
        writer.flush(); writer.close()
        env.close()


# ================================ CLI parsing =================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO training for Neon White movement drills")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--level", default=None)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--timescale", type=float, default=1.0)
    parser.add_argument("--action-rate-hz", type=float, default=60.0)
    parser.add_argument("--no-wait", action="store_true")

    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--action-repeat", type=int, default=1, help="Repeat each chosen action N steps (>=1)")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)

    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--run-name", default=f"movement_{int(time.time())}")
    parser.add_argument("--checkpoint-interval", type=int, default=30, help="How many rollouts between checkpoints")

    # Action toggles/forces
    parser.add_argument("--allow-move-x", type=str2bool, default=True)
    parser.add_argument("--allow-move-y", type=str2bool, default=True)
    parser.add_argument("--allow-look-x", type=str2bool, default=True)
    parser.add_argument("--allow-look-y", type=str2bool, default=True)
    parser.add_argument("--allow-jump", type=str2bool, default=True)
    parser.add_argument("--allow-shoot", type=str2bool, default=True)
    parser.add_argument("--allow-use", type=str2bool, default=True)
    parser.add_argument("--force-move-x", type=float, default=None)
    parser.add_argument("--force-move-y", type=float, default=None)
    parser.add_argument("--force-look-x", type=float, default=None)
    parser.add_argument("--force-look-y", type=float, default=None)
    parser.add_argument("--force-jump", type=str2bool, default=None)
    parser.add_argument("--force-shoot", type=str2bool, default=None)
    parser.add_argument("--force-use", type=str2bool, default=None)

    parser.add_argument("--control-enabled", action="store_true")
    parser.add_argument("--stuck-seconds", type=float, default=2.0)
    parser.add_argument("--stuck-distance", type=float, default=0.15)
    parser.add_argument("--episode-seconds", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # Resume
    parser.add_argument("--load-checkpoint", type=str, default=None)

    # Debug
    parser.add_argument("--debug-reward", action="store_true")

    # Convenience toggles
    parser.add_argument("--constant-forward", action="store_true")
    parser.add_argument("--disable-strafe", action="store_true")
    parser.add_argument("--disable-shoot", action="store_true")
    parser.add_argument("--disable-use", action="store_true")
    parser.add_argument("--disable-jump", action="store_true")
    parser.add_argument("--disable-look", action="store_true")
    parser.add_argument("--disable-look-x", action="store_true")
    parser.add_argument("--disable-look-y", action="store_true")
    parser.add_argument("--constant-look-x", type=float, default=None)
    parser.add_argument("--constant-look-y", type=float, default=None)

    # OCR detection (optional)
    parser.add_argument("--detect-level-text", action="store_true")
    parser.add_argument("--level-complete-bbox", type=str, default=None)
    parser.add_argument("--level-complete-reward", type=float, default=2000.0)
    parser.add_argument("--ocr-interval-steps", type=int, default=60)
    parser.add_argument("--ocr-interval-seconds", type=float, default=0.0)
    parser.add_argument("--ocr-goal-dist", type=float, default=0.0)

    # ========= JSON level-config controls =========
    parser.add_argument("--use-level-config", action="store_true",
                        help="Use JSON-configured goal/enemy data (overrides obs fields and enables radius-based completion).")
    parser.add_argument("--level-config", type=str, default="levels.json",
                        help="Path to the JSON file with level specs.")
    parser.add_argument("--level-id", type=str, default=None,
                        help="Key of the level inside the JSON file to use.")
    parser.add_argument("--goal-radius", type=float, default=None,
                        help="If provided with --add-level, sets/overrides the level's goal radius. Otherwise JSON value is used.")
    parser.add_argument("--add-level", action="store_true",
                        help="Add or update a level entry in the JSON using CLI values, then continue training.")
    parser.add_argument("--overwrite-level", action="store_true",
                        help="Overwrite the entire level entry instead of merging.")
    parser.add_argument("--goal", type=str, default=None,
                        help="Vec3 'x,y,z' for level goal (used with --add-level).")
    parser.add_argument("--enemy", action="append", default=None,
                        help="Vec3 'x,y,z' for an enemy; repeat flag for multiple (used with --add-level).")
    parser.add_argument("--level-name", type=str, default=None,
                        help="Human-friendly name for the level (used with --add-level).")

    # ======= Coordinate transform controls for JSON -> world =======
    parser.add_argument("--coord-order",
                        choices=["xyz","xzy","yxz","yzx","zxy","zyx"],
                        default="xyz",
                        help="Axis order of numbers in levels.json relative to world (default xyz).")
    parser.add_argument("--negate-x", action="store_true", help="Flip X sign after reordering")
    parser.add_argument("--negate-y", action="store_true", help="Flip Y sign after reordering")
    parser.add_argument("--negate-z", action="store_true", help="Flip Z sign after reordering")
    parser.add_argument("--coord-offset", type=str, default=None,
                        help="Offset 'dx,dy,dz' to add to transformed coords")
    parser.add_argument("--coord-scale", type=float, default=1.0,
                        help="Scale factor applied to transformed coords")
    parser.add_argument("--print-pos", action="store_true",
                        help="Print player pos and transformed goal every ~0.5s for calibration")
    # ===============================================================

    # ======= Reward shaping knobs (JSON-goal based) =======
    parser.add_argument("--reward-env-scale", type=float, default=0.0,
                        help="Multiply environment's own reward (default 0 = ignore).")
    parser.add_argument("--reward-progress-scale", type=float, default=1.0,
                        help="Scale for per-step progress (prev_dist - curr_dist) towards JSON goal.")
    parser.add_argument("--reward-progress-nonneg", type=str2bool, default=True,
                        help="If true, only reward positive progress (no penalty for moving away).")
    parser.add_argument("--reward-progress-clip", type=float, default=0.0,
                        help="Clip absolute per-step progress before scaling (0 disables).")
    parser.add_argument("--reward-step", type=float, default=0.0,
                        help="Constant added each step (use negative for time penalty).")
    parser.add_argument("--reward-death-penalty", type=float, default=0.0)
    parser.add_argument("--reward-stuck-penalty", type=float, default=0.0)
    parser.add_argument("--reward-timeout-penalty", type=float, default=0.0)

    parser.add_argument(
        "--fall-penalty",
        type=float,
        default=20.0,
        help="Penalty applied once when a fall/void/out-of-bounds death occurs."
    )

    # ======= NEW robust fall detection controls =======
    parser.add_argument("--fall-detect", choices=["auto", "info", "pos"], default="auto",
                        help="Use env info/events, position heuristics, or both (auto).")
    parser.add_argument("--fall-axis", choices=["x", "y", "z"], default="y",
                        help="Axis that represents 'up' for position-based fall detection.")
    parser.add_argument("--fall-y-threshold", type=float, default=None,
                        help="Treat pos[axis] <= this value as a fall (disabled if None).")
    parser.add_argument("--fall-drop-delta", type=float, default=6.0,
                        help="Treat a one-step drop >= this distance as a fall (<=0 disables).")
    parser.add_argument("--fall-apply-on-terminal", type=str2bool, default=True,
                        help="If true, apply penalty when the episode terminates/truncates; otherwise immediately.")
    # ===================================================

    # ======= NEW progress controls (baseline & stability) =======
    parser.add_argument("--enable-progress-reward", type=str2bool, default=True,
                        help="Use progress shaping based on change in goal_dist.")
    parser.add_argument("--progress-scale", type=float, default=None,
                        help="Alias for --reward-progress-scale; if set, overrides it.")
    parser.add_argument("--progress-delta-cap", type=float, default=-1.0,
                        help="Clamp |progress delta| per step; <=0 uses --reward-progress-clip; 0 disables.")
    parser.add_argument("--progress-warmup-steps", type=int, default=3,
                        help="Steps after reset/respawn where progress shaping is suppressed.")
    parser.add_argument("--progress-jump-reset", type=float, default=5.0,
                        help="If goal_dist increases by more than this in one step, treat as respawn and reset baseline. Set <=0 to disable.")
    # ===========================================================

    args = parser.parse_args()

    # Apply convenience toggles
    if getattr(args, "constant_forward", False):
        args.force_move_y = 1.0
    if getattr(args, "disable_strafe", False):
        args.allow_move_x = False
    if getattr(args, "disable_shoot", False):
        args.allow_shoot = False
    if getattr(args, "disable_use", False):
        args.allow_use = False
    if getattr(args, "disable_jump", False):
        args.allow_jump = False
    if getattr(args, "disable_look", False):
        args.allow_look_x = False
        args.allow_look_y = False

    if getattr(args, "disable_look_x", False):
        args.allow_look_x = False
    if getattr(args, "disable_look_y", False):
        args.allow_look_y = False
    if getattr(args, "constant_look_x", None) is not None:
        args.force_look_x = max(-1.0, min(1.0, args.constant_look_x))
    if getattr(args, "constant_look_y", None) is not None:
        args.force_look_y = max(-1.0, min(1.0, args.constant_look_y))

    # Parse OCR bbox
    args.level_complete_bbox = parse_bbox(args.level_complete_bbox)

    # Parse coord offset vec3
    if args.coord_offset is not None:
        args.coord_offset = parse_vec3(args.coord_offset)
    else:
        args.coord_offset = None

    # Alias: --progress-scale overrides --reward-progress-scale if set
    if args.progress_scale is not None:
        args.reward_progress_scale = float(args.progress_scale)

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)