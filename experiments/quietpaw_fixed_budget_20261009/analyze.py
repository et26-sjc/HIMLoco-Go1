"""Report the fixed primary endpoint only; refuses incomplete studies."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'legged_gym/logs/quietpaw_fixed_budget_20261009'


def main():
    import torch
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise RuntimeError('Study incomplete: do not select interim checkpoints')
    records, checks, contrasts = {}, {}, {}
    metrics = ['observed_discounted_cost', 'pid_bootstrapped_cost', 'signed_violation',
               'survival', 'alive_seconds', 'cost_per_alive_second', 'touchdown_rate',
               'tracking', 'initial_cost_bias', 'initial_cost_rmse', 'timeout_tail_mean']
    for seed in (4, 5):
        initial = []
        for arm in ('zero', 'pid'):
            name = 'seed{}_{}'.format(seed, arm)
            directory = OUT / name
            initial.append(json.loads((directory / 'initial.json').read_text()))
            checkpoint = torch.load(directory / 'model_300.pt', map_location='cpu')
            assert checkpoint['iter'] == 300
            assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
            steps = {}
            for key in ('optimizer_state_dict', 'estimator_optimizer_state_dict'):
                steps[key] = sorted(set(int(s['step']) for s in checkpoint[key]['state'].values()))
                assert steps[key] == [6000], (name, key, steps[key])
            pid = checkpoint['quietpaw']['pid']
            multiplier = pid['multiplier']
            if arm == 'zero':
                assert not multiplier.any()
            checks[name] = dict(optimizer_steps=steps, lambda_mean=multiplier.mean().item(),
                                lambda_max=multiplier.max().item(),
                                pid_updates_mean=pid['updates'].float().mean().item())
            evaluations = [json.loads((OUT / (name + '_eval' + str(s)) / 'result.json').read_text())
                           for s in (111, 112)]
            records[name] = {m: statistics.mean(e['summary'][m] for e in evaluations) for m in metrics}
            records[name]['deployment'] = json.loads((OUT / (name + '_deploy11') / 'result.json').read_text())['summary']
            records[name]['gradients'] = [json.loads(line) for line in (directory / 'gradients.jsonl').read_text().splitlines()]
        assert initial[0] == initial[1], 'Paired initialization mismatch'
        contrasts[str(seed)] = {m: records['seed{}_pid'.format(seed)][m] - records['seed{}_zero'.format(seed)][m]
                                for m in metrics}
    report = dict(epsilon=manifest['epsilon'], calibration_means=manifest['calibration_means'],
                  checks=checks, records=records, pid_minus_zero=contrasts,
                  interpretation='Two paired training seeds: mechanism diagnostic, not a robust algorithm benefit claim.')
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ['# Fixed-budget mechanism test', '',
             'Budget: `{}`. Primary endpoint: 300 updates; 122,880,000 transitions per arm.'.format(manifest['epsilon']), '',
             '| Training seed / arm | Discounted cost | Jc − epsilon | Survival | Cost / alive second | Tracking error |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, record in records.items():
        lines.append('| {} | {:.6f} | {:+.6f} | {:.2%} | {:.4f} | {:.4f} |'.format(
            name, record['observed_discounted_cost'], record['signed_violation'], record['survival'],
            record['cost_per_alive_second'], record['tracking']))
    lines += ['', 'Paired initialization hashes match. Both optimizers took 6000 steps in every arm.', '',
              'The JSON report contains paired contrasts, cost-critic calibration, lambda, gradient probes and deployment physical metrics.', '',
              'Two training seeds support mechanism diagnosis only. Lower cost must be interpreted alongside survival and alive-time cost.']
    (OUT / 'analysis.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
