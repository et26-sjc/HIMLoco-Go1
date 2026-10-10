"""Paired, first-episode-only evaluation for frozen-policy quiet adaptation.

Run from the repository with the Isaac Gym Python environment. This entrypoint
does not create an RL runner or initialize Weights & Biases.
"""
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'rsl_rl'))
sys.path.insert(0, str(REPO / 'legged_gym'))


def _clone_config(value):
    """Copy nested config classes while retaining data mappings as mappings."""
    if isinstance(value, dict):
        return {key: _clone_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_config(item) for item in value)
    if isinstance(value, type) or hasattr(value, "__dict__"):
        return SimpleNamespace(**{key: _clone_config(getattr(value, key))
                                  for key in dir(value) if not key.startswith("_")})
    return copy.deepcopy(value)


def _json_write(path, report):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def load_policy(path, device="cpu", adapter_enabled=True):
    """Load either an audited Quiet checkpoint or the original Go1 HIM model."""
    import torch
    from rsl_rl.modules.quiet_actor_critic import QuietHIMActorCritic
    from rsl_rl.utils.quiet_checkpoint import load_base_checkpoint, load_quiet_policy
    payload = torch.load(path, map_location=device)
    if "quiet_metadata" in payload:
        return load_quiet_policy(path, device=device, adapter_enabled=adapter_enabled)
    model = QuietHIMActorCritic(270, 238, 45, 12, adaptation="frozen").to(device)
    metadata = load_base_checkpoint(model, path, map_location=device)
    model.set_adapter_enabled(adapter_enabled)
    model.eval()
    return model, {"quiet_metadata": metadata, "iter": payload.get("iter", 0)}


def _configure_condition(cfg, condition, episodes, duration, seed):
    cfg.env.num_envs = episodes
    # Stop explicitly at the requested horizon. The legacy environment uses
    # episode_length > max_episode_length; do not mislabel horizon as a fall.
    dt = cfg.sim.dt * cfg.control.decimation
    cfg.env.episode_length_s = duration + 2.0 * dt
    cfg.seed = seed
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = duration + 1.0
    cfg.terrain.curriculum = False
    cfg.terrain.selected = False
    cfg.terrain.mesh_type = condition["terrain"]
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.disturbance = False
    cfg.quiet.touchdown_reference = None
    cfg.quiet.target_weight = 0.0
    if condition["terrain"] != "plane":
        cfg.terrain.num_rows = 1
        cfg.terrain.num_cols = 1
        cfg.terrain.terrain_length = 32.0
        cfg.terrain.terrain_width = 32.0
        cfg.terrain.max_init_terrain_level = 0
        cfg.terrain.terrain_proportions = [.2, .2, .2, .2, .2, 0., 0.]
        cfg.terrain.evaluation_choice = condition["choice"]
        cfg.terrain.evaluation_difficulty = condition["difficulty"]


def evaluate_policy(model, args, suite="all", split="validation", episodes=32,
                    duration=20.0, conditions=None, progress=None):
    """Evaluate each condition with paired seeds and preserve the live model.

    Every parallel environment contributes its first episode. Once it fails,
    later auto-reset episodes are masked. The caller may use this during training
    without stepping or reconfiguring its training environment.
    """
    import torch
    from isaacgym import gymtorch
    from legged_gym.envs import task_registry
    from legged_gym.utils.quiet_evaluation import (
        QUIET_EVALUATION_FORMAT_VERSION, QuietEpisodeMetrics, condition_seed,
        evaluation_conditions, validate_evaluation_schedule)
    if episodes < 1 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Evaluation needs positive episodes and duration")
    selected = evaluation_conditions(suite) if conditions is None else list(conditions)
    cfg, _ = task_registry.get_cfgs("go1_quiet")
    template_cfg = _clone_config(cfg)
    # Validate all conditions, including the last push bucket, before creating
    # the first simulator. Invalid short suites cannot leave a partial report.
    num_steps, push_steps = validate_evaluation_schedule(
        selected, duration, template_cfg.sim.dt * template_cfg.control.decimation)
    report = {"format_version": QUIET_EVALUATION_FORMAT_VERSION, "suite": suite, "split": split,
              "episodes_per_condition": episodes, "duration_s": duration,
              "conditions": [], "action_mode": "deterministic_mean",
              "quiet_measurement": "world-vertical foot speed sampled 5 ms before contact",
              "impact_measurement": "20 ms contact-load window; not an acoustic model",
              "seed_offset": int(getattr(args, "seed_offset", 0))}
    original_device = next(model.parameters()).device
    original_training = model.training
    original_adapter = model.adapter_enabled
    import numpy as np
    rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        model.eval()
        for condition in selected:
            cfg = _clone_config(template_cfg)
            seed = condition_seed(condition["name"], split) + report["seed_offset"]
            _configure_condition(cfg, condition, episodes, duration, seed)
            condition_args = copy.copy(args)
            condition_args.task = "go1_quiet"
            condition_args.num_envs = episodes
            condition_args.seed = seed
            env = None
            try:
                env, _ = task_registry.make_env("go1_quiet", condition_args, env_cfg=cfg)
                model.to(env.device)
                env.set_evaluation_command(condition["command"])
                obs, _ = env.reset()
                # Ensure every history begins with the prescribed command.
                # reset() already calls compute_observations after resampling.
                env.set_evaluation_command(condition["command"])
                env.compute_observations()
                obs = env.get_observations()
                # reset() includes one zero-action simulation step. It provides
                # initialized observation history, but is outside this test.
                env.episode_length_buf.zero_()
                env.quiet_episode_time.zero_()
                env.quiet_tracker.reset(torch.arange(episodes, device=env.device))
                names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
                foot_names = [names[int(index)] for index in env.feet_indices.cpu().tolist()]
                metrics = QuietEpisodeMetrics(condition, episodes, env.dt, duration, foot_names,
                                              env.torque_limits.cpu().numpy(),
                                              action_scale=cfg.control.action_scale,
                                              startup_seconds=cfg.quiet.startup_time,
                                              action_clip_limit=cfg.normalization.clip_actions,
                                              deployment_action_limit=env.deployment_action_limit)
                metrics.set_initial_position(env.root_states[:, :3].cpu().numpy())
                platform_exit = np.zeros(episodes, dtype=bool)
                terrain_centers = env.env_origins[:, :2].cpu().numpy().copy()
                push_applied = False
                push_episode_count = 0
                push_step = push_steps.get(condition["name"], -1)
                with torch.no_grad():
                    for step in range(num_steps):
                        if step == push_step:
                            active = torch.as_tensor(metrics.active, device=env.device)
                            env.root_states[active, 8] += condition["push_velocity_y"]
                            env.gym.set_actor_root_state_tensor(
                                env.sim, gymtorch.unwrap_tensor(env.root_states))
                            push_episode_count = int(metrics.active.sum())
                            push_applied = push_episode_count > 0
                        actions = model.act_inference(obs)
                        drift = None
                        if original_adapter:
                            model.set_adapter_enabled(False)
                            base_actions = model.act_inference(obs)
                            model.set_adapter_enabled(True)
                            drift = (actions - base_actions).cpu().numpy()
                        result = env.step(actions)
                        obs = result[0]
                        snapshot = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
                                    for key, value in result[4]["quiet"].items()}
                        if condition["terrain"] != "plane":
                            outside = np.max(np.abs(snapshot["root_position"][:, :2]
                                                    - terrain_centers), axis=1) > 1.5
                            platform_exit |= metrics.active & outside
                        if step == num_steps - 1:
                            # The explicit horizon does not call reset_idx.
                            # Count windows that cannot finish within the test.
                            pending = (env.quiet_tracker.window_remaining > 0).cpu().numpy()
                            snapshot["impact_invalid"] += pending * metrics.active[:, None]
                        metrics.add(snapshot, action_drift=drift,
                                    raw_actions=actions.cpu().numpy())
                        if not metrics.active.any():
                            break
                summary = metrics.finish()
                summary["seed"] = seed
                summary["physical_dt_s"] = cfg.sim.dt
                summary["control_dt_s"] = env.dt
                summary["push_applied"] = push_applied
                summary["push_episode_count"] = push_episode_count
                summary["terrain_platform_exit_count"] = (int(platform_exit.sum())
                                                            if condition["terrain"] != "plane" else None)
                summary["terrain_exposure_note"] = (
                    "Central platform width is 3 m. A short run may not traverse the terrain."
                    if condition["terrain"] != "plane" else None)
                report["conditions"].append(summary)
                if progress is not None:
                    progress(summary)
            finally:
                if env is not None:
                    if env.viewer is not None:
                        env.gym.destroy_viewer(env.viewer)
                    env.gym.destroy_sim(env.sim)
                del env
    finally:
        model.to(original_device)
        model.set_adapter_enabled(original_adapter)
        model.train(original_training)
        random.setstate(rng_state[0])
        np.random.set_state(rng_state[1])
        torch.set_rng_state(rng_state[2])
        if rng_state[3] is not None:
            torch.cuda.set_rng_state_all(rng_state[3])
    return report


def _parse_args():
    from isaacgym import gymutil
    parameters = [
        {"name": "--checkpoint", "type": str, "required": True},
        {"name": "--baseline_checkpoint", "type": str},
        {"name": "--suite", "type": str, "choices": ["flat", "rough", "push", "all"], "default": "all"},
        {"name": "--split", "type": str, "choices": ["validation", "test"], "default": "validation"},
        {"name": "--episodes", "type": int, "default": 32},
        {"name": "--duration", "type": float, "default": 20.0},
        {"name": "--seed_offset", "type": int, "default": 0},
        {"name": "--output", "type": str, "default": "quiet_evaluation.json"},
        {"name": "--adapter", "type": str, "choices": ["enabled", "disabled"], "default": "enabled"},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
    ]
    args = gymutil.parse_arguments(description="Quiet policy evaluation", custom_parameters=parameters)
    # gymutil forwards only name/type/default/help to argparse in this version.
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    if args.suite not in ("flat", "rough", "push", "all"):
        raise ValueError("--suite must be flat, rough, push or all")
    if args.split not in ("validation", "test") or args.adapter not in ("enabled", "disabled"):
        raise ValueError("Invalid split or adapter selection")
    args.sim_device = args.rl_device
    args.num_envs = args.episodes
    args.seed = None
    args.task = "go1_quiet"
    return args


def main():
    # Isaac Gym must be imported before torch in its supported Python runtime.
    import isaacgym  # noqa: F401
    args = _parse_args()
    from rsl_rl.utils.quiet_checkpoint import file_sha256
    # Import envs before utils to avoid the package's legacy registry cycle.
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.quiet_evaluation import select_candidate
    model, checkpoint = load_policy(args.checkpoint, device=args.rl_device,
                                    adapter_enabled=args.adapter == "enabled")
    report = evaluate_policy(model, args, args.suite, args.split, args.episodes, args.duration,
                             progress=lambda item: print("Evaluated " + item["name"], flush=True))
    report["checkpoint"] = os.path.abspath(args.checkpoint)
    report["checkpoint_sha256"] = file_sha256(args.checkpoint)
    report["iteration"] = int(checkpoint.get("iter", 0))
    if args.baseline_checkpoint:
        baseline, _ = load_policy(args.baseline_checkpoint, device=args.rl_device, adapter_enabled=False)
        baseline_report = evaluate_policy(baseline, args, args.suite, args.split, args.episodes, args.duration)
        baseline_report["checkpoint_sha256"] = file_sha256(args.baseline_checkpoint)
        report["baseline"] = baseline_report
        report["selection"] = select_candidate(report, baseline_report)
    _json_write(args.output, report)
    print("Saved evaluation to " + os.path.abspath(args.output))


if __name__ == "__main__":
    main()
