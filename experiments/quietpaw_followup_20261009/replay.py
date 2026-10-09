"""Replay one identical rollout at two initial learning rates; no core edits."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/quietpaw_fixed_budget_20261009'))


def main(args):
    import isaacgym
    import torch
    import worker
    from rsl_rl.algorithms.quietpaw_constrained_ppo import QuietPawConstrainedPPO
    base_update = QuietPawConstrainedPPO.update

    def replay(alg):
        ac, st = alg.actor_critic, alg.storage
        weights = copy.deepcopy(ac.state_dict())
        opt = copy.deepcopy(alg.optimizer.state_dict())
        himopt = copy.deepcopy(ac.estimator.optimizer.state_dict())
        tensors = {k: v.clone() for k, v in vars(alg).items() if torch.is_tensor(v)}
        metrics = copy.deepcopy(alg.metric_sums)
        metric_steps = alg.metric_steps
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        indices = torch.linspace(0, st.observations.flatten(0, 1).shape[0]-1,
                                 8192, device=alg.device).long()
        obs = st.observations.flatten(0, 1)[indices]
        eps = st.epsilon.flatten(0, 1)[indices]
        old_mu, old_std = st.mu.flatten(0, 1)[indices], st.sigma.flatten(0, 1)[indices]

        @torch.no_grad()
        def policy():
            return ac.act_inference(obs, eps).clone(), ac.std.expand_as(old_mu).clone()

        @torch.no_grad()
        def distance(left, right):
            mu0, s0 = left
            mu1, s1 = right
            kl = (torch.log(s1/s0) + (s0.square() + (mu0-mu1).square()) /
                  (2*s1.square()) - .5).sum(-1)
            return dict(kl_mean=kl.mean().item(), kl_p95=torch.quantile(kl,.95).item(),
                        standardized_mean_rms=((mu0-mu1)/s0).square().mean().sqrt().item())

        records = {}
        for lr in (1e-3, 1e-5):
            ac.load_state_dict(weights)
            alg.optimizer.load_state_dict(opt)
            ac.estimator.optimizer.load_state_dict(himopt)
            for k,v in tensors.items():
                setattr(alg,k,v.clone())
            alg.metric_sums = copy.deepcopy(metrics)
            alg.metric_steps = metric_steps
            alg.iterations = 0
            alg.learning_rate = lr
            for group in alg.optimizer.param_groups:
                group['lr'] = lr
            alg.diagnostics_interval = 0
            st.step = st.num_transitions_per_env
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state_all(cuda)
            record = dict(initial_lr=lr, initial=distance((old_mu,old_std),policy()), steps=[])
            original_ppo, original_him = alg.optimizer.step, ac.estimator.optimizer.step

            def wrap_step(original, optimizer, kind):
                def step(*a, **kw):
                    before = policy()
                    result = original(*a, **kw)
                    after = policy()
                    record['steps'].append(dict(kind=kind, lr=optimizer.param_groups[0]['lr'],
                        local=distance(before,after), collection=distance((old_mu,old_std),after)))
                    return result
                return step

            alg.optimizer.step = wrap_step(original_ppo, alg.optimizer, 'ppo')
            ac.estimator.optimizer.step = wrap_step(original_him, ac.estimator.optimizer, 'him')
            try:
                result = base_update(alg)
            finally:
                alg.optimizer.step = original_ppo
                ac.estimator.optimizer.step = original_him
            record['final'] = distance((old_mu,old_std),policy())
            record['losses'] = result
            record['parameter_delta'] = {
                prefix: sum((v-weights[k]).square().sum() for k,v in ac.state_dict().items()
                            if k.startswith(prefix)).sqrt().item()
                for prefix in ['actor.', 'estimator.', 'critic.', 'cost_critic.', 'std']}
            record['optimizer_steps'] = {
                name: sorted(set(int(v['step']) for v in optimizer.state.values()))
                for name,optimizer in [('ppo',alg.optimizer),('him',ac.estimator.optimizer)]}
            assert record['optimizer_steps'] == {'ppo':[20], 'him':[20]}
            records[str(lr)] = record
        Path(args.output,'replay.json').write_text(json.dumps(records,indent=2,allow_nan=False))
        print('REPLAY_RESULT',json.dumps({k:v['final'] for k,v in records.items()}),flush=True)
        return result

    QuietPawConstrainedPPO.update = replay
    worker.main(argparse.Namespace(mode='train',output=args.output,him_checkpoint=str(ROOT /
        'legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_8000.pt'),
        checkpoint=None,expected_initial=None,seed=args.seed,num_envs=4096,iterations=1,
        epsilon=.037646,arm='zero'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
