"""Short lambda=0 stability control; only the initial shared LR differs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/quietpaw_fixed_budget_20261009'))


def main(args):
    import isaacgym
    import worker
    from legged_gym.envs.go1.quietpaw_go1_config import QuietPawStage1Go1CfgPPO
    QuietPawStage1Go1CfgPPO.algorithm.learning_rate = args.lr
    worker.main(argparse.Namespace(mode='train', output=args.output, him_checkpoint=str(ROOT /
        'legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_8000.pt'),
        checkpoint=None, expected_initial=None, seed=args.seed, num_envs=4096,
        iterations=20, epsilon=.037646, arm='zero'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lr', type=float, choices=[1e-3,1e-5], required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
