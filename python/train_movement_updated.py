#!/usr/bin/env python3
from __future__ import annotations

"""
Reinforcement learning training script for Neon White movement drills.

This script is largely identical to the original `train_movement.py` from the
`Neon-White-DRL-AI` project, with one key difference: the in‑game reset
sequence has been modified to better mirror how a human player restarts a level
in Neon White.  The original implementation used the RL bridge’s reset
mechanism, which only pressed the `F` key and then relied on a separate
`send_start_pulse` to press the space bar afterwards.  In certain cases this
could leave the game in a hanging state, because the restart confirmation
dialog would appear but no confirmation key was sent.

To avoid this issue, `send_reset_sequence` now presses both `F` and `Space`
in sequence whenever a reset is requested, and the training loop no longer
calls `send_start_pulse` after each reset.  This way the reset happens
entirely via the game’s UI (rather than through the bridge), and the
environment reset call is used only to synchronise the observation state.

See the original project for further details on PPO training and the Neon
White environment.
"""

import argparse
import random
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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

# Dimensions for continuous and discrete action components.  The continuous
# component controls movement and camera look, and the discrete component
# controls jump, shoot, use and reset flags.
CONTINUOUS_COMPONENT = 4  # move_x, move_y, look_x, look_y (normalized)
DISCRETE_COMPONENT = 4    # jump, shoot, use, reset
LOOK_SCALE = 10.0


def ensure_pyautogui():
    """
    Deprecated: Previously used to import and configure pyautogui when
    --control-enabled was passed.  The current implementation of the
    training script no longer relies on pyautogui for resets; instead, it
    sends reset and start commands through the RL bridge directly using
    environment actions.  This function remains for backwards
    compatibility but is no longer used.
    """
    raise RuntimeError("pyautogui is no longer used by this script")


def press_sequence(pyauto, keys, press_duration=0.08, gap=0.12):
    """
    Deprecated: Previously used to press a sequence of keys via pyautogui.
    This function is no longer called, because resets are now performed
    entirely via environment actions rather than through OS‑level input.
    """
    raise RuntimeError("press_sequence is no longer used; resets are handled via env.step")


def send_start_pulse(pyauto):
    """
    Deprecated: In earlier versions of this script, a start pulse was sent via
    pyautogui to press Space.  The new implementation uses the RL bridge
    directly and issues a jump action through env.step instead.  This
    function should not be called.
    """
    raise RuntimeError("send_start_pulse is no longer used; use send_start_action instead")


def send_reset_sequence(pyauto):
    """
    Deprecated: Previously sent the F and Space keys via pyautogui to
    reset the level.  Resets are now accomplished by sending a reset
    action through the RL bridge (see send_reset_action).  This function
    should not be called.
    """
    raise RuntimeError("send_reset_sequence is no longer used; use send_reset_action instead")


# -----------------------------------------------------------------------------
# New helper functions: interacting with the game via the RL bridge
# -----------------------------------------------------------------------------
def send_start_action(env: NeonWhiteEnv) -> None:
    """
    Send a jump action through the environment to start the level.  This
    function issues a single step with 'jump' set to True and then waits
    briefly.  The RL bridge will translate this into a Space key press
    inside the game.  It should be used immediately after the initial
    env.reset() call to start or resume the level without relying on
    pyautogui.
    """
    action = {
        "move": [0.0, 0.0],
        "look": [0.0, 0.0],
        "jump": True,
        "shoot": False,
        "use": False,
        "reset": False,
    }
    # One step to press space
    env.step(action)
    # Wait a short time to allow the game to register the jump
    time.sleep(0.1)


def send_reset_action(env: NeonWhiteEnv) -> None:
    """
    Send a reset command through the RL bridge.  This issues a single step
    with the 'reset' flag set to True.  The patched NeonRLBridge
    automatically translates a reset into pressing F and Space within the
    game, so this function replaces the pyautogui-based reset sequence.

    After calling this function you should wait briefly and then call
    env.reset() to synchronise the observation state with the new level
    state.  A short pause allows the game to process the restart.
    """
    action = {
        "move": [0.0, 0.0],
        "look": [0.0, 0.0],
        "jump": False,
        "shoot": False,
        "use": False,
        "reset": True,
    }
    # Send the reset action via env.step.  This should trigger F and Space
    # inside the game due to the NeonRLBridge modifications.
    env.step(action)
    # Short pause to allow the reset to complete.  Tune as necessary.
    time.sleep(0.1)


@dataclass
class ActionToggles:
    """
    Flags indicating which action dimensions are allowed to vary.  If a flag
    is False the corresponding action component will be zeroed out.
    """
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
    """
    Constant values that override sampled actions.  If any field is not None
    the corresponding action component will be forced to the specified value.
    """
    move_x: Optional[float] = None
    move_y: Optional[float] = None
    look_x: Optional[float] = None
    look_y: Optional[float] = None
    jump: Optional[bool] = None
    shoot: Optional[bool] = None
    use: Optional[bool] = None

    def is_active(self) -> bool:
        return any(value is not None for value in (
            self.move_x,
            self.move_y,
            self.look_x,
            self.look_y,
            self.jump,
            self.shoot,
            self.use,
        ))


def apply_action_mask(continuous: torch.Tensor, discrete: torch.Tensor, mask: ActionToggles) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero out action components according to the provided mask.  This is used to
    restrict the agent from using certain inputs (e.g., disabling jump).
    """
    cont = continuous.clone()
    disc = discrete.clone()
    if not mask.move_x:
        cont[..., 0] = 0.0
    if not mask.move_y:
        cont[..., 1] = 0.0
    if not mask.look_x:
        cont[..., 2] = 0.0
    if not mask.look_y:
        cont[..., 3] = 0.0
    if not mask.jump:
        disc[..., 0] = 0.0
    if not mask.shoot:
        disc[..., 1] = 0.0
    if not mask.use:
        disc[..., 2] = 0.0
    if not mask.reset:
        disc[..., 3] = 0.0
    return cont, disc


def apply_action_forces(continuous: torch.Tensor, discrete: torch.Tensor, forces: ActionForces) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Override action components with fixed values specified in `forces`.  Returns
    clones of the input tensors only if mutations are necessary to avoid
    inadvertent in‑place modification.
    """
    if not forces.is_active():
        return continuous, discrete

    cont = continuous
    disc = discrete
    cont_mutated = False
    disc_mutated = False

    def ensure_cont() -> torch.Tensor:
        nonlocal cont, cont_mutated
        if not cont_mutated:
            cont = cont.clone()
            cont_mutated = True
        return cont

    def ensure_disc() -> torch.Tensor:
        nonlocal disc, disc_mutated
        if not disc_mutated:
            disc = disc.clone()
            disc_mutated = True
        return disc

    if forces.move_x is not None:
        ensure_cont()[..., 0] = float(np.clip(forces.move_x, -1.0, 1.0))
    if forces.move_y is not None:
        ensure_cont()[..., 1] = float(np.clip(forces.move_y, -1.0, 1.0))
    if forces.look_x is not None:
        ensure_cont()[..., 2] = float(np.clip(forces.look_x, -1.0, 1.0))
    if forces.look_y is not None:
        ensure_cont()[..., 3] = float(np.clip(forces.look_y, -1.0, 1.0))

    if forces.jump is not None:
        ensure_disc()[..., 0] = 1.0 if bool(forces.jump) else 0.0
    if forces.shoot is not None:
        ensure_disc()[..., 1] = 1.0 if bool(forces.shoot) else 0.0
    if forces.use is not None:
        ensure_disc()[..., 2] = 1.0 if bool(forces.use) else 0.0

    return cont, disc


def str2bool(value: str | bool) -> bool:
    """
    Parse a string or boolean into a boolean value.  Accepts various
    representations such as 'true', 'false', 'yes', 'no', '1', '0'.
    """
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got '{value}'")


def flatten_observation(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten the structured observation dictionary into a 1D numpy array.  The
    order of elements is defined by `OBS_KEYS`.
    """
    pieces: List[np.ndarray] = []
    for key in OBS_KEYS:
        value = obs.get(key)
        if value is None:
            raise KeyError(f"Missing observation key '{key}' from bridge payload")
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        pieces.append(arr)
    return np.concatenate(pieces, axis=0)


def continuous_action_to_env(action: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the network’s continuous action output into the environment’s expected
    move and look vectors.  The look components are scaled by LOOK_SCALE.
    """
    action = action.detach().cpu().numpy()
    move = action[..., :2]
    look = action[..., 2:] * LOOK_SCALE
    return move, look


def build_action_dictionary(
    continuous: torch.Tensor,
    discrete: torch.Tensor,
) -> Dict[str, object]:
    """
    Build the action dictionary expected by the Neon White environment.  The
    discrete actions are thresholded at 0.5 to produce boolean commands.
    """
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


def make_writer(log_dir: Path, run_name: str) -> SummaryWriter:
    """
    Create a TensorBoard SummaryWriter given a log directory and run name.  The
    directory will be created if it does not exist.
    """
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(run_dir))


def explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute the explained variance between true and predicted values.  Returns 0
    when the target variance is zero to avoid division by zero.
    """
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
    """
    Save a checkpoint containing the current training step, model state,
    optimizer state, and PPO configuration.  Returns the path to the saved
    checkpoint.
    """
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
    """
    Set Python, NumPy, and PyTorch random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    """
    Main training loop for PPO.  Handles environment interaction, data
    collection, advantage computation, and gradient updates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

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
    # We no longer use pyautogui to send resets; instead, we interact
    # with the game via the RL bridge by sending actions.  pyauto is unused.
    pyauto = None

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
    # Perform the initial environment reset.  If a level is specified in
    # reset_options this will cause the bridge to load the level.  On
    # subsequent resets we avoid specifying the level again to prevent
    # redundant scene loads (which can cause a black screen when the level
    # is already active).  See `env.reset` in neon_white_rl/env.py for details【939372537959003†L204-L216】.
    obs_sample, info = env.reset(options=reset_options if reset_options else None)

    # After the first reset, remove the "level" entry from reset_options so
    # that future resets do not reload the same scene.  This avoids
    # potential hangs when the game attempts to load a level that is
    # already active.  We leave stage and timescale entries intact.
    if "level" in reset_options:
        reset_options.pop("level")

    if control_enabled:
        # Start the level by issuing a jump action via the RL bridge.  This
        # replaces the old pyautogui-based start pulse.
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
    obs_vector = flatten_observation(obs_sample)
    obs_dim = int(obs_vector.shape[0])

    policy = ActorCritic(
        obs_dim=obs_dim,
        continuous_dim=CONTINUOUS_COMPONENT,
        discrete_dim=DISCRETE_COMPONENT,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    # Optional resume from checkpoint.  If --load-checkpoint is provided and
    # points to a valid .pt file, load the model and optimizer state from
    # that file and store the saved global step.  We defer assigning
    # global_step until later so that it overrides the default of 0.
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
            print(f"[train] Warning: specified checkpoint file {ckpt_path} does not exist; starting from scratch.")

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

    # Prepare the logging and checkpoint directories.  All runs are
    # organised under `log_dir/run_name`.  The tensorboard writer will log
    # into this directory, and checkpoints will be saved into a
    # "checkpoints" subdirectory.  When resuming from a checkpoint, you
    # should specify the same run_name and supply --load-checkpoint
    # pointing to a .pt file inside the corresponding checkpoints
    # directory.
    log_root = Path(args.log_dir).expanduser().resolve()
    run_dir = log_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # If a specific checkpoint directory was supplied, use it; otherwise
    # default to run_dir/checkpoints.  This allows users to override
    # checkpoint storage if desired.
    checkpoint_dir = Path(args.checkpoint_dir or (run_dir / "checkpoints")).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create a tensorboard writer.  We construct it directly rather than
    # using make_writer to avoid nesting the run name twice.
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

    # Initialise the global step.  If resuming from a checkpoint, use the
    # step value loaded earlier; otherwise start at 0.
    global_step = loaded_step
    start_time = time.time()
    obs_tensor = torch.tensor(obs_vector, device=device, dtype=torch.float32)
    episode_start_wall = time.time()

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
                next_obs, reward, terminated, truncated, info = env.step(action_dict)

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
                    # If the agent dies but env.step() does not report termination,
                    # treat it as a truncated episode and record the death reason.
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

                    # Stuck detection: if minimal movement for stuck_seconds.
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

                    # Timeout detection: if episode runs longer than episode_seconds.
                    if timed_out and not (terminated or truncated):
                        truncated = True
                        if isinstance(info, dict):
                            info = dict(info)
                            events = list(info.get('events', ()))
                            events.append({'type': 'timeout'})
                            info['events'] = events

                reward_val = float(reward)

                # Debugging: print reward information if enabled.  This can help
                # diagnose the reward shaping by showing the immediate reward at
                # each step along with termination flags and any events.
                if args.debug_reward:
                    # Collect some details to make debugging easier
                    done_flag = bool(terminated or truncated)
                    events = info.get('events') if isinstance(info, dict) else None
                    print(f"[debug_reward] step={global_step} reward={reward_val:.5f} done={done_flag} events={events}")
                done_val = float(terminated or truncated)

                rollout_buffer.add(
                    obs=obs_tensor,
                    cont_action=cont_action,
                    disc_action=disc_action,
                    log_prob=log_prob,
                    value=value_scalar,
                    reward=reward_val,
                    done=done_val,
                )

                current_reward += reward_val
                current_length += 1
                global_step += 1

                if terminated or truncated:
                    # Record statistics
                    episode_rewards.append(current_reward)
                    episode_lengths.append(current_length)
                    if control_enabled:
                        # RESET SEQUENCE
                        # Send a reset command through the RL bridge.  This will
                        # press F and Space inside the game via the patched
                        # NeonRLBridge.  Afterwards, perform env.reset() to
                        # synchronise the observation state.  A short pause
                        # allows the game to process the reset.
                        send_reset_action(env)
                        # Call env.reset() to receive the first observation of
                        # the new episode.  Since reset_options no longer
                        # contains "level", this will not reload the scene.
                        next_obs, info = env.reset(options=reset_options if reset_options else None)
                        # Brief pause to allow the environment to settle
                        time.sleep(0.2)
                    else:
                        stop_requested = True
                        break
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
                    last_done = done_val
                    last_done = done_val

                if info.get("escape_requested"):
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

            if stop_requested:
                break

        if stop_requested:
            path = save_checkpoint(checkpoint_dir, args.run_name, global_step, policy, optimizer, cfg)
            print(f"[train] Stop requested. Saved checkpoint to {path}")
        else:
            path = save_checkpoint(checkpoint_dir, args.run_name, global_step, policy, optimizer, cfg)
            print(f"[train] Training complete. Final checkpoint saved to {path}")
    finally:
        signal.signal(signal.SIGINT, original_handler)
        writer.flush()
        writer.close()
        env.close()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.  Provides many options for environment
    configuration, PPO hyperparameters, and training behaviour.
    """
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
    parser.add_argument("--action-repeat", type=int, default=1, help="Number of consecutive env steps to repeat each chosen action (>=1)")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--run-name", default=f"movement_{int(time.time())}")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="How many rollouts between checkpoints")
    parser.add_argument("--allow-move-x", type=str2bool, default=True, help="Allow horizontal movement inputs")
    parser.add_argument("--allow-move-y", type=str2bool, default=True, help="Allow forward/back movement inputs")
    parser.add_argument("--allow-look-x", type=str2bool, default=True, help="Allow horizontal camera look inputs")
    parser.add_argument("--allow-look-y", type=str2bool, default=True, help="Allow vertical camera look inputs")
    parser.add_argument("--allow-jump", type=str2bool, default=True, help="Allow jump (space) presses")
    parser.add_argument("--allow-shoot", type=str2bool, default=True, help="Allow shoot (left click)")
    parser.add_argument("--allow-use", type=str2bool, default=True, help="Allow use (right click)")
    parser.add_argument("--force-move-x", type=float, default=None, help="Force strafe axis to constant value in [-1, 1]")
    parser.add_argument("--force-move-y", type=float, default=None, help="Force forward/back axis to constant value in [-1, 1]")
    parser.add_argument("--force-look-x", type=float, default=None, help="Force horizontal look axis to constant value in [-1, 1]")
    parser.add_argument("--force-look-y", type=float, default=None, help="Force vertical look axis to constant value in [-1, 1]")
    parser.add_argument("--force-jump", type=str2bool, default=None, help="Force jump pressed (true) or released (false) each step")
    parser.add_argument("--force-shoot", type=str2bool, default=None, help="Force shoot pressed (true) or released (false) each step")
    parser.add_argument("--force-use", type=str2bool, default=None, help="Force use pressed (true) or released (false) each step")
    parser.add_argument("--control-enabled", action="store_true", help="Allow trainer to send start/reset commands")
    parser.add_argument("--stuck-seconds", type=float, default=2.0, help="Seconds of minimal movement before triggering a reset; set <=0 to disable")
    parser.add_argument("--stuck-distance", type=float, default=0.15, help="Movement threshold in meters to consider progress for stuck detection")
    parser.add_argument("--episode-seconds", type=float, default=45.0, help="Seconds before forcing an environment reset; set <=0 to disable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    # When resuming training you can specify the path to an existing checkpoint
    # file via --load-checkpoint.  When this is provided, the model and
    # optimizer states will be restored from the file and training will
    # continue from the saved global step.  The run name should point to
    # the directory you wish to continue logging into.
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint file (.pt) to resume training from.  "
            "If supplied, the model and optimizer state will be loaded "
            "before training begins, and global step will be restored."
        ),
    )

    # Debugging options
    parser.add_argument("--debug-reward", action="store_true", help="Print the reward received at each environment step for debugging")

    # ----------------------------------------------------------------------
    # Convenience toggles for isolating action components
    #
    # --constant-forward    : Force the agent to always move forward at maximum speed.
    # --disable-strafe      : Disable horizontal movement (strafe left/right).
    # --disable-shoot       : Disable shooting actions.
    # --disable-use         : Disable use (right click) actions.
    # --disable-jump        : Disable jumping actions.
    # --disable-look        : Disable camera look on both axes.
    #
    # These flags provide a shorthand for setting the corresponding
    # --force-move-y, --allow-move-x, --allow-shoot, --allow-use,
    # --allow-jump, --allow-look-x, and --allow-look-y arguments.  They
    # can be combined as needed to test specific abilities in isolation.
    parser.add_argument("--constant-forward", action="store_true", help="Force constant forward movement (move_y=1.0)")
    parser.add_argument("--disable-strafe", action="store_true", help="Disable horizontal movement (strafe)")
    parser.add_argument("--disable-shoot", action="store_true", help="Disable shooting actions")
    parser.add_argument("--disable-use", action="store_true", help="Disable use actions")
    parser.add_argument("--disable-jump", action="store_true", help="Disable jump actions")
    parser.add_argument("--disable-look", action="store_true", help="Disable camera look in both axes")

    # Additional look control toggles
    parser.add_argument("--disable-look-x", action="store_true", help="Disable horizontal (X-axis) camera look")
    parser.add_argument("--disable-look-y", action="store_true", help="Disable vertical (Y-axis) camera look")
    parser.add_argument("--constant-look-x", type=float, default=None, help="Force horizontal look to a constant value in [-1.0, 1.0]")
    parser.add_argument("--constant-look-y", type=float, default=None, help="Force vertical look to a constant value in [-1.0, 1.0]")
    args = parser.parse_args()

    # Apply convenience toggles.  These flags override or disable specific
    # action components by setting the appropriate force or allow variables.
    if getattr(args, "constant_forward", False):
        # Force move_y to 1.0 (always move forward)
        args.force_move_y = 1.0
    if getattr(args, "disable_strafe", False):
        # Disable horizontal movement
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

    # Individual look-axis toggles and forced values
    if getattr(args, "disable_look_x", False):
        args.allow_look_x = False
    if getattr(args, "disable_look_y", False):
        args.allow_look_y = False
    if getattr(args, "constant_look_x", None) is not None:
        # Clamp the constant within [-1, 1]
        args.force_look_x = max(-1.0, min(1.0, args.constant_look_x))
    if getattr(args, "constant_look_y", None) is not None:
        args.force_look_y = max(-1.0, min(1.0, args.constant_look_y))

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)