from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import MujocoSimConfig
from .keyboard import KeyboardCommander
from .mujoco_env import MujocoGo1Env
from .policy_runner import PolicyRunner


def _print_status(
    t: int,
    cfg: MujocoSimConfig,
    commander: KeyboardCommander,
    action: np.ndarray,
    env: MujocoGo1Env,
) -> None:

    if not cfg.debug:
        print(
            f"{t:05d} cmd=({commander.state.vx:+.2f},"
            f"{commander.state.vy:+.2f},"
            f"{commander.state.wyaw:+.2f}) "
            f"act(mean,std)=({np.mean(action):+.3f},{np.std(action):.3f})"
        )
        return

    dbg = env.get_debug()
    if not dbg:
        print(
            f"{t:05d} cmd=({commander.state.vx:+.2f},"
            f"{commander.state.vy:+.2f},"
            f"{commander.state.wyaw:+.2f}) "
            f"act(mean,std)=({np.mean(action):+.3f},{np.std(action):.3f})"
        )
        return

    v = dbg["base_lin_vel_body"]
    w = dbg["base_ang_vel_body"]
    pg = dbg["projected_gravity"]

    # ===== 新增：heading 相关 =====
    yaw = dbg.get("yaw", 0.0)                    # 当前机体 yaw
    cmd_yaw_rate = dbg.get("cmd_yaw_rate", commander.state.wyaw)  # 实际送入网络的 yaw_rate
    # ==============================

    print(
        f"{t:05d} "
        f"cmd_raw=({commander.state.vx:+.2f},"
        f"{commander.state.vy:+.2f},"
        f"{commander.state.wyaw:+.2f}) "
        f"yaw={yaw:+.2f} "
        f"cmd_yaw_rate={cmd_yaw_rate:+.2f} "
        f"v_body=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) "
        f"w_body=({w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}) "
        f"pg=({pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f}) "
        f"act(mean,std)=({np.mean(action):+.3f},{np.std(action):.3f}) "
        f"act_sat={dbg['action_sat_frac']:.2f} "
        f"tau_sat={dbg['tau_sat_frac']:.2f} "
        f"tau_max_ratio={dbg['tau_max_ratio']:.2f}"
    )


def run(cfg: MujocoSimConfig) -> None:
    env = MujocoGo1Env(cfg)
    policy = PolicyRunner(policy_path=cfg.policy_path, device=cfg.device)

    commander = KeyboardCommander(cfg.cmd_x, cfg.cmd_y, cfg.cmd_yaw)
    # 关键：把 CLI 传入的 heading 写进去
    commander.state.heading = cfg.cmd_heading

    obs = env.reset()

    # make sure action is always defined (useful for safety if you change branches later)
    action = np.zeros(env.pcfg.num_actions, dtype=np.float32)

    if cfg.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=commander.on_key) as viewer:
            for t in range(cfg.steps):
                # 1) set command (heading mode supported)
                if cfg.heading_command:
                    env.set_command(commander.state.vx, commander.state.vy, commander.state.wyaw, heading=commander.state.heading)
                else:
                    env.set_command(commander.state.vx, commander.state.vy, commander.state.wyaw)

                # 2) compute action (ALWAYS happens)
                action = policy(obs[None, :])  # PolicyRunner returns (12,) for batch=1

                # 3) step
                obs = env.step(action)

                if t % cfg.print_every == 0 or t == cfg.steps - 1:
                    _print_status(t, cfg, commander, action, env)
                viewer.sync()
    else:
        for t in range(cfg.steps):
            if cfg.heading_command:
                env.set_command(commander.state.vx, commander.state.vy, commander.state.wyaw, heading=commander.state.heading)
            else:
                env.set_command(commander.state.vx, commander.state.vy, commander.state.wyaw)

            action = policy(obs[None, :])
            obs = env.step(action)

            if t % cfg.print_every == 0 or t == cfg.steps - 1:
                _print_status(t, cfg, commander, action, env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Go1 HIM policy.pt MuJoCo sim2sim demo")
    p.add_argument("--policy", type=Path, required=True, help="Path to exported policy.pt")
    p.add_argument("--xml", type=Path, default=Path("legged_gym/resources/robots/go1/xml/go1.xml"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--cmd-x", type=float, default=1.0)
    p.add_argument("--cmd-y", type=float, default=0.0)
    p.add_argument("--cmd-yaw", type=float, default=0.0)
    p.add_argument("--heading-command", action="store_true", default=False,
                   help="Use heading command mode (training setting).")
    p.add_argument("--cmd-heading", type=float, default=0.0,
                   help="Target heading (rad) when --heading-command is set.")
    p.add_argument("--headless", action="store_true", default=False)

    # debug
    p.add_argument("--debug", action="store_true", default=False, help="Enable extra debug prints")
    p.add_argument("--print-every", type=int, default=20, help="Print interval (steps)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = MujocoSimConfig(
        model_xml=args.xml,
        policy_path=args.policy,
        device=args.device,
        viewer=not args.headless,
        steps=args.steps,
        debug=args.debug,
        print_every=args.print_every,
        cmd_x=args.cmd_x,
        cmd_y=args.cmd_y,
        cmd_yaw=args.cmd_yaw,
        heading_command=args.heading_command,
        cmd_heading=args.cmd_heading,
    )
    run(cfg)
