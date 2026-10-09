"""E3 training wrapper: low initial LR and explicit PID gain factor."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/quietpaw_fixed_budget_20261009'))


def main(args):
    import isaacgym
    import worker
    from instrumentation import install
    from legged_gym.envs.go1.quietpaw_go1_config import QuietPawStage1Go1CfgPPO
    from rsl_rl.runners.quietpaw_stage1_runner import QuietPawStage1Runner
    cfg=QuietPawStage1Go1CfgPPO.algorithm
    cfg.learning_rate=1e-5
    for name in ('kp','ki','kd'):
        setattr(cfg.pid,name,getattr(cfg.pid,name)*args.gain)
    cfg.pid.initial=0.
    original_learn=QuietPawStage1Runner.learn
    def learn(runner,*a,**kw):
        install(runner.alg,args.output,args.iterations)
        return original_learn(runner,*a,**kw)
    QuietPawStage1Runner.learn=learn
    worker.main(argparse.Namespace(mode='train',output=args.output,him_checkpoint=str(ROOT /
        'legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_8000.pt'),
        checkpoint=None,expected_initial=args.expected_initial,seed=args.seed,num_envs=args.num_envs,
        iterations=args.iterations,epsilon=.037646,arm='pid'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--gain',type=int,choices=[0,1,100,1000],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--expected-initial')
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--iterations',type=int,default=300)
    main(parser.parse_args())
