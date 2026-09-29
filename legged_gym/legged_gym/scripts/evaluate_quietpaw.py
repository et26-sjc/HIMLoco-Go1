"""Paired-seed epsilon sweep. A fresh simulator process is used per epsilon.

Run --help for options. Training environment settings are not edited.
Evaluation fixes commands and freezes curricula, retaining physics/randomization.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Corrected Stage 1 checkpoint path")
    p.add_argument("--output", required=True)
    p.add_argument("--epsilons", nargs="+", type=float, default=[0., .1, .2, .3, .4, .6, .8, 1.])
    p.add_argument("--num-envs", type=int, default=160)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--command", nargs=3, type=float, default=[1., 0., 0.], metavar=("VX", "VY", "YAW"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--worker", type=float, help=argparse.SUPPRESS)
    return p


def worker(args):
    import isaacgym
    import torch
    from legged_gym.envs.go1.quietpaw_go1_config import QuietPawStage1Go1Cfg
    from legged_gym.envs.go1.quietpaw_legged_robot import QuietPawLeggedRobot
    from legged_gym.utils import get_args, task_registry
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.modules.quietpaw_conditional_actor_critic import QuietPawConditionalActorCritic
    from rsl_rl.utils.quietpaw_evaluation import EpsilonEvaluation

    class FixedEvaluationRobot(QuietPawLeggedRobot):
        def fixed_command(self):
            self.commands[:, :3] = torch.tensor(args.command, device=self.device)

        def _resample_commands(self, env_ids):
            super()._resample_commands(env_ids)
            self.fixed_command()

        def _post_physics_step_callback(self):
            super()._post_physics_step_callback()
            self.fixed_command()  # override heading feedback BEFORE reward/obs

        def _update_terrain_curriculum(self, env_ids):
            pass  # evaluate one fixed initial terrain distribution

        def update_command_curriculum(self, env_ids):
            pass

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    contract = checkpoint.get("quietpaw", {}).get("contract", {})
    if contract.get("version") != 2:
        raise ValueError("Evaluation requires corrected Stage 1 contract v2")
    cfg = QuietPawStage1Go1Cfg()
    if contract["noise_definition"] != cfg.noise_cost.definition or any(
            getattr(cfg.noise_cost, k) != v for k, v in contract["noise_parameters"].items()):
        raise ValueError("Checkpoint and evaluation noise definitions differ")
    # Use standard Isaac Gym argument parsing without changing shared helpers.
    sys.argv = [sys.argv[0], "--headless", "--num_envs", str(args.num_envs),
                "--seed", str(args.seed), "--rl_device", args.device]
    gym_args = get_args()
    task_registry.task_classes["quietpaw_eval"] = FixedEvaluationRobot
    env, _ = task_registry.make_env("quietpaw_eval", args=gym_args, env_cfg=cfg)
    model = QuietPawConditionalActorCritic(270, 238, 45, 12).to(env.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # Same reset sequence as BaseTask.reset, but fingerprint the assigned
    # state BEFORE its zero-action physics step (GPU PhysX is not bitwise).
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    initial = dict(terrain_types=env.terrain_types.cpu().tolist(),
                   terrain_levels=env.terrain_levels.cpu().tolist(),
                   root_states=env.root_states.cpu().tolist(),
                   dof_pos=env.dof_pos.cpu().tolist())
    for name in ("friction_coeffs", "payload", "Kp_factors", "Kd_factors", "motor_strength_factors"):
        value = getattr(env, name, None)
        if torch.is_tensor(value):
            initial[name] = value.cpu().tolist()
    fingerprint = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
    obs, _, _, _, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    env.episode_length_buf.zero_()
    epsilon = torch.full((env.num_envs, 1), args.worker, device=env.device)
    stats = EpsilonEvaluation(env.num_envs, args.worker, contract["algorithm"]["gamma"], env.device)
    with torch.no_grad():
        for step in range(args.steps):
            actions = model.act_inference(obs, epsilon)
            obs, critic, rewards, dones, infos, ids, terminal = env.step(actions)
            next_critic = critic.clone()
            next_critic[ids] = terminal
            values = model.evaluate_cost(next_critic, epsilon)
            stats.add(rewards, dones, infos["quietpaw"], infos["time_outs"], values,
                      last_step=step == args.steps - 1)
            if not stats.active.any():
                break
    result = dict(summary=stats.summary(), episodes=stats.episodes(), initial=initial,
                  initial_sha256=fingerprint, seed=args.seed, command=args.command,
                  horizon_steps=args.steps, dt=env.dt, checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  contract=contract, environment_config=class_to_dict(cfg))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    env.gym.destroy_sim(env.sim)


def sweep(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, epsilon in enumerate(args.epsilons):
        path = output / "epsilon_{:02d}.json".format(index)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", args.checkpoint,
               "--output", str(path), "--worker", str(epsilon), "--seed", str(args.seed),
               "--num-envs", str(args.num_envs), "--steps", str(args.steps), "--device", args.device,
               "--command", *map(str, args.command)]
        with (output / "epsilon_{:02d}.log".format(index)).open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        results.append(json.loads(path.read_text()))
        print("Evaluated epsilon", epsilon, flush=True)
    if len({r["initial_sha256"] for r in results}) != 1:
        raise RuntimeError("Initial terrain/robot states differ across epsilon; no paired summary produced")
    rows = [r["summary"] for r in results]
    for row in rows:
        row["pareto_reward_noise"] = not any(
            other["noise_cost"] <= row["noise_cost"] and other["locomotion_reward"] >= row["locomotion_reward"]
            and (other["noise_cost"] < row["noise_cost"] or other["locomotion_reward"] > row["locomotion_reward"])
            for other in rows)
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for row in rows:
        ax.scatter(row["noise_cost"], row["locomotion_reward"],
                   marker="o" if row["pareto_reward_noise"] else "x")
        ax.annotate(str(row["epsilon"]), (row["noise_cost"], row["locomotion_reward"]))
    frontier = sorted((r for r in rows if r["pareto_reward_noise"]), key=lambda r: r["noise_cost"])
    ax.plot([r["noise_cost"] for r in frontier], [r["locomotion_reward"] for r in frontier], "--")
    ax.set(xlabel="Mean sampled noise proxy", ylabel="First-episode locomotion return",
           title="Empirical reward/noise trade-off (labels: epsilon)")
    fig.tight_layout()
    fig.savefig(output / "reward_noise.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    args = parser().parse_args()
    levels = args.epsilons if args.worker is None else [args.worker]
    if args.steps < 1 or args.num_envs < 1 or any(not 0 <= value <= 1 for value in levels):
        raise ValueError("Positive horizon/env count and finite epsilon in [0,1] required")
    worker(args) if args.worker is not None else sweep(args)
