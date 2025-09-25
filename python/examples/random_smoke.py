#!/usr/bin/env python3
"""Quick smoke test that drives NeonRLBridge with random actions."""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict
from pathlib import Path



if __package__ in (None, ):
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from neon_white_rl.env import NeonWhiteEnv, NeonWhiteConfig
else:
    from ..neon_white_rl.env import NeonWhiteEnv, NeonWhiteConfig


def _make_env(host: str, port: int, wait_for_ready: bool) -> NeonWhiteEnv:
    config = NeonWhiteConfig(host=host, port=port, wait_for_ready=wait_for_ready)
    return NeonWhiteEnv(config=config)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Random-policy bridge smoke test")
    parser.add_argument("--episodes", type=int, default=1, help="number of rollouts to run")
    parser.add_argument("--steps", type=int, default=600, help="max steps per episode")
    parser.add_argument("--port", type=int, default=5555, help="bridge TCP port")
    parser.add_argument("--host", default="127.0.0.1", help="bridge host address")
    parser.add_argument("--stage", type=int, default=None, help="stage value to send on reset")
    parser.add_argument("--timescale", type=float, default=None, help="timescale to request on reset")
    parser.add_argument("--level", default=None, help="level name to load on reset")
    parser.add_argument("--no-wait", action="store_true", help="skip waiting for ready packets on reset")
    parser.add_argument("--sleep", type=float, default=0.0, help="extra sleep (seconds) after each action")

    args = parser.parse_args(argv)

    try:
        env = _make_env(args.host, args.port, not args.no_wait)
    except (ConnectionRefusedError, TimeoutError) as exc:
        print(f"[bridge] failed to connect: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"[bridge] socket error: {exc}", file=sys.stderr)
        return 1

    print(f"[bridge] connected to {args.host}:{args.port}")

    options: Dict[str, Any] = {}
    if args.stage is not None:
        options["stage"] = args.stage
    if args.timescale is not None:
        options["timescale"] = args.timescale
    if args.level is not None:
        options["level"] = args.level
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(options=options)
            print(f"[episode {episode}] reset: stage={info.get('stage')} level={info.get('level')} esc={info.get('escape_requested')}")
            if info["events"]:
                print(f"[episode {episode}] events: {info['events']}")

            for step_idx in range(args.steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if info["events"]:
                    print(f"[episode {episode} step {step_idx}] events: {info['events']}")
                if reward:
                    print(f"[episode {episode} step {step_idx}] reward={reward:.3f} dist={obs['goal_dist'][0]:.2f}")
                if info.get("escape_requested"):
                    print("[bridge] escape requested, stopping rollouts")
                    return 0
                if terminated or truncated:
                    print(f"[episode {episode}] finished step {step_idx} terminated={terminated} truncated={truncated}")
                    break
                if args.sleep > 0:
                    time.sleep(args.sleep)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))





