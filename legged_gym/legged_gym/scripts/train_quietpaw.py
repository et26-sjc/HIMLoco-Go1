"""Stage 1 training entry point that honors explicit --resume arguments."""

import os
import math
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.algorithms.quietpaw_assignment import validate_epsilon_range


def configure_experiment(env_cfg, train_cfg, args):
    """Explicit experiment overrides; omitted arguments preserve old contracts."""
    fixed = getattr(args, "epsilon_fixed", None)
    if fixed is not None:
        if not math.isfinite(fixed) or not 0 <= fixed <= 1:
            raise ValueError("--epsilon_fixed must be finite and within [0,1]")
        if any(getattr(args, name, None) is not None for name in ("epsilon_min", "epsilon_max")):
            raise ValueError("Use --epsilon_fixed or an epsilon range, not both")
        train_cfg.algorithm.epsilon_fixed = fixed
        train_cfg.algorithm.epsilon_min = fixed
        train_cfg.algorithm.epsilon_max = fixed
    for name in ("epsilon_min", "epsilon_max"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(train_cfg.algorithm, name, value)
    if fixed is None:
        validate_epsilon_range(train_cfg.algorithm.epsilon_min, train_cfg.algorithm.epsilon_max)
    gain = getattr(args, "pid_gain_scale", None)
    if gain is not None:
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("--pid_gain_scale must be finite and nonnegative")
        for name in ("kp", "ki", "kd"):
            setattr(train_cfg.algorithm.pid, name, gain * getattr(train_cfg.algorithm.pid, name))
        if gain == 0:
            train_cfg.algorithm.pid.initial = 0.0
    interval = getattr(args, "constraint_diagnostics_interval", None)
    if interval is not None:
        if interval < 0:
            raise ValueError("--constraint_diagnostics_interval must be nonnegative")
        train_cfg.algorithm.diagnostics_interval = interval
    force_weight = getattr(args, "noise_force_weight", None)
    if force_weight is not None:
        if not math.isfinite(force_weight) or not 0 <= force_weight <= 1:
            raise ValueError("--noise_force_weight must be finite and within [0,1]")
        env_cfg.noise_cost.force_weight = force_weight
        env_cfg.noise_cost.impact_weight = 1 - force_weight


def train(args):
    if args.task != "go1_quietpaw_stage1":
        raise ValueError("Use --task go1_quietpaw_stage1 with train_quietpaw.py")
    him_checkpoint = getattr(args, "him_checkpoint", None)
    if args.resume and him_checkpoint:
        raise ValueError("--resume and --him_checkpoint cannot be combined")
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    configure_experiment(env_cfg, train_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, cfg = task_registry.make_alg_runner(env=env, args=args, train_cfg=train_cfg)
    if him_checkpoint:
        print("HIM weight-only warm-start:", runner.initialize_from_him(him_checkpoint))
        if runner.log_dir is not None:
            runner.save(os.path.join(runner.log_dir, "model_him_init.pt"))
    runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train(get_args([
        {"name": "--him_checkpoint", "type": str,
         "help": "Original HIM checkpoint for weight-only Stage 1 initialization"},
        {"name": "--epsilon_min", "type": float,
         "help": "Minimum actual cost budget in the epsilon grid (default 0.2)"},
        {"name": "--epsilon_max", "type": float,
         "help": "Maximum actual cost budget in the epsilon grid (default 1.0)"},
        {"name": "--noise_force_weight", "type": float,
         "help": "Force cost weight; touchdown weight is 1 minus this value (default 0.2)"},
        {"name": "--pid_gain_scale", "type": float,
         "help": "Multiply Kp/Ki/Kd by this factor; zero gives the lambda=0 control"},
        {"name": "--epsilon_fixed", "type": float,
         "help": "Assign this actual cost budget to every environment for a single-condition control"},
        {"name": "--constraint_diagnostics_interval", "type": int,
         "help": "Enable rollout diagnostics and sample actor gradients every N updates; 0 disables"},
    ]))
