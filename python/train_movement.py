#!/usr/bin/env python3
from __future__ import annotations

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

CONTINUOUS_COMPONENT = 4  # move_x, move_y, look_x, look_y (normalized)
DISCRETE_COMPONENT = 4    # jump, shoot, use, reset
LOOK_SCALE = 10.0


def ensure_pyautogui():
    try:
        import pyautogui
    except ImportError as exc:
        raise RuntimeError("--control-enabled requires the pyautogui package (pip install pyautogui)") from exc
    pyautogui.FAILSAFE = False
    return pyautogui


def press_sequence(pyauto, keys, press_duration=0.08, gap=0.12):
    for key in keys:
        pyauto.keyDown(key)
        time.sleep(press_duration)
        pyauto.keyUp(key)
        if gap > 0:
            time.sleep(gap)


def send_start_pulse(pyauto):
    press_sequence(pyauto, ['space'])


def send_reset_sequence(pyauto):
    press_sequence(pyauto, ['f'])




START_PULSE = {
    "move": [0.0, 0.0],
    "look": [0.0, 0.0],
    "jump": True,
    "shoot": False,
    "use": False,
}


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
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
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




def build_start_action(forces: ActionForces) -> Dict[str, object]:
    action = START_PULSE.copy()
    if forces.move_x is not None:
        action["move"][0] = float(np.clip(forces.move_x, -1.0, 1.0))
    if forces.move_y is not None:
        action["move"][1] = float(np.clip(forces.move_y, -1.0, 1.0))
    if forces.look_x is not None:
        action["look"][0] = float(np.clip(forces.look_x, -1.0, 1.0))
    if forces.look_y is not None:
        action["look"][1] = float(np.clip(forces.look_y, -1.0, 1.0))
    if forces.jump is not None:
        action["jump"] = bool(forces.jump)
    if forces.shoot is not None:
        action["shoot"] = bool(forces.shoot)
    if forces.use is not None:
        action["use"] = bool(forces.use)
    return action.copy()


def continuous_action_to_env(action: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    action = action.detach().cpu().numpy()
    move = action[..., :2]
    look = action[..., 2:] * LOOK_SCALE
    return move, look


def build_action_dictionary(
    continuous: torch.Tensor,
    discrete: torch.Tensor,
) -> Dict[str, object]:
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
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
    pyauto = ensure_pyautogui() if control_enabled else None

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
    obs_sample, info = env.reset(options=reset_options if reset_options else None)
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
    if control_enabled:
        time.sleep(0.2)
        send_start_pulse(pyauto)
        time.sleep(0.15)
    obs_vector = flatten_observation(obs_sample)
    obs_dim = int(obs_vector.shape[0])

    policy = ActorCritic(
        obs_dim=obs_dim,
        continuous_dim=CONTINUOUS_COMPONENT,
        discrete_dim=DISCRETE_COMPONENT,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

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

    log_dir = Path(args.log_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir or (log_dir / "checkpoints")).expanduser().resolve()
    writer = make_writer(log_dir, args.run_name)

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

    global_step = 0
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

                reward_val = float(reward)
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
                    episode_rewards.append(current_reward)
                    episode_lengths.append(current_length)
                    if control_enabled:
                        send_reset_sequence(pyauto)
                        time.sleep(0.3)
                        next_obs, info = env.reset(options=reset_options if reset_options else None)
                        time.sleep(0.2)
                        send_start_pulse(pyauto)
                        time.sleep(0.15)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

