"""Stage 1 training entry point that honors explicit --resume arguments."""

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train(args):
    if args.task != "go1_quietpaw_stage1":
        raise ValueError("Use --task go1_quietpaw_stage1 with train_quietpaw.py")
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    runner.learn(num_learning_iterations=cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train(get_args())
