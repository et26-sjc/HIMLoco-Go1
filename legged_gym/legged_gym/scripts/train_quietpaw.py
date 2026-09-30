"""Stage 1 training entry point that honors explicit --resume arguments."""

import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train(args):
    if args.task != "go1_quietpaw_stage1":
        raise ValueError("Use --task go1_quietpaw_stage1 with train_quietpaw.py")
    him_checkpoint = getattr(args, "him_checkpoint", None)
    if args.resume and him_checkpoint:
        raise ValueError("--resume and --him_checkpoint cannot be combined")
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    if him_checkpoint:
        print("HIM weight-only warm-start:", runner.initialize_from_him(him_checkpoint))
        if runner.log_dir is not None:
            runner.save(os.path.join(runner.log_dir, "model_him_init.pt"))
    runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train(get_args([{"name": "--him_checkpoint", "type": str,
                    "help": "Original HIM checkpoint for weight-only Stage 1 initialization"}]))
