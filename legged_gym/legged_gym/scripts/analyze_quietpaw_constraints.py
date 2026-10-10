"""Read-only analysis of the matched Stage 1 constraint validation study."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[3]


def paired_interval(reference, candidate):
    """Conditional environment-pair interval, not training-seed uncertainty."""
    difference = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    mean = float(difference.mean())
    half_width = float(1.96 * difference.std(ddof=1) / np.sqrt(len(difference)))
    return dict(change=mean, conditional_environment_pair_95ci=[mean-half_width, mean+half_width])


def evaluation_records(directory):
    records = {}
    for path in Path(directory).glob('epsilon_*.json'):
        record = json.loads(path.read_text())
        records[round(record['summary']['epsilon'], 8)] = record
    return records


def compare(reference, candidate, criteria):
    if reference['initial_sha256'] != candidate['initial_sha256']:
        raise ValueError('Initial state fingerprints differ; paired comparison rejected')
    r, c = reference['summary'], candidate['summary']
    cost_reduction = 1 - c['observed_discounted_cost'] / r['observed_discounted_cost']
    raw_reduction = 1 - c['noise_cost'] / r['noise_cost']
    tracking_change = c['velocity_tracking_error_mean'] / r['velocity_tracking_error_mean'] - 1
    survival_change = 100 * (c['survival_success_rate'] - r['survival_success_rate'])
    result = dict(cost_reduction_fraction=cost_reduction, raw_cost_reduction_fraction=raw_reduction,
                  survival_change_percentage_points=survival_change,
                  tracking_error_change_fraction=tracking_change,
                  locomotion_return_change=c['locomotion_reward']-r['locomotion_reward'],
                  reward_per_step_change=c['reward_per_step']-r['reward_per_step'],
                  screening_pass=(cost_reduction >= criteria['cost_reduction_relative_to_lambda0']
                                  and raw_reduction > 0
                                  and survival_change >= -criteria['maximum_survival_drop_percentage_points']
                                  and tracking_change <= criteria['maximum_tracking_error_increase_fraction']))
    for key in ('observed_discounted_cost', 'observed_discounted_reward', 'reward', 'steps'):
        if key in reference['episodes'] and key in candidate['episodes']:
            result[key] = paired_interval(reference['episodes'][key], candidate['episodes'][key])
    result['survival'] = paired_interval(1-np.asarray(reference['episodes']['failure'], dtype=float),
                                         1-np.asarray(candidate['episodes']['failure'], dtype=float))
    return result


def training_curves(run):
    event = EventAccumulator(str(next(Path(run).glob('events.out.tfevents.*'))),
                             size_guidance={'scalars': 0}).Reload()
    global_tags = {'Train/mean_reward', 'Train/mean_episode_length', 'Loss/value_function',
                   'QuietPaw/noise_cost_mean', 'QuietPaw/completed_cost_mean',
                   'QuietPaw/cost_value_loss', 'QuietPaw/lambda_next_mean',
                   'QuietPaw/lambda_next_max', 'QuietPaw/budget_violation_mean'}
    selected = [tag for tag in event.Tags()['scalars'] if tag in global_tags
                or tag.startswith('QuietPaw/epsilon_bin_0/')
                or (tag.startswith('QuietPaw/epsilon_bin_') and '/actor_' in tag)]
    return {tag: {'steps': [r.step for r in event.Scalars(tag)],
                  'values': [r.value for r in event.Scalars(tag)]}
            for tag in selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', default=str(ROOT / 'docs/quietpaw_constraint_validation'))
    args = parser.parse_args()
    study = Path(args.study).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((study/'manifest.json').read_text())
    arms = manifest['arms']
    evaluated = {a: evaluation_records(r['evaluation_directory'])
                 for a, r in arms.items() if r.get('status') == 'evaluated'}
    if evaluated and len({r['initial_sha256'] for records in evaluated.values()
                          for r in records.values()}) != 1:
        raise ValueError('Cross-arm initial evaluation states differ')
    complete = all(arms.get(a, {}).get('status') == 'evaluated'
                   for a in manifest['configuration']['arms'])
    hashes = {r['verification']['initial_model_tensor_sha256']
              for r in arms.values() if 'verification' in r}
    if len(hashes) > 1:
        raise ValueError('Initial network weights differ')
    comparisons = {}
    control = evaluated.get('grid_g0', {})
    criteria = manifest['protocol']['pilot_screening']
    for arm, records in evaluated.items():
        if arm == 'grid_g0':
            continue
        comparisons[arm] = {str(e): compare(control[e], r, criteria)
                            for e, r in sorted(records.items()) if e in control}
    curves = {a: training_curves(r['run_directory']) for a, r in arms.items()
              if 'run_directory' in r}
    conditional = {a: compare(records[min(records)], records[max(records)], criteria)
                   for a, records in evaluated.items() if len(records) > 1}
    source_files = ('legged_gym/legged_gym/scripts/evaluate_quietpaw.py',
                    'rsl_rl/rsl_rl/utils/quietpaw_evaluation.py',
                    'legged_gym/legged_gym/scripts/analyze_quietpaw_constraints.py')
    result = dict(complete=complete, manifest=str(study/'manifest.json'),
                  configuration=manifest['configuration'], protocol=manifest['protocol'],
                  initial_model_tensor_hashes=sorted(hashes),
                  evaluation_source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                                             for f in source_files},
                  comparisons_to_lambda0=comparisons, within_policy_endpoint_changes=conditional,
                  summaries={a: [r['summary'] for e, r in sorted(records.items())]
                             for a, records in evaluated.items()}, training_curves=curves,
                  training_summary_last50={a:r['verification']['last50_training_means']
                                           for a,r in arms.items() if 'verification' in r})
    output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    rows = [dict(arm=a, **r['summary']) for a, records in evaluated.items()
            for e, r in sorted(records.items())]
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with output.with_suffix('.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    specs = [('QuietPaw/noise_cost_mean', 'Training raw cost'),
             ('QuietPaw/epsilon_bin_0/lambda_mean', 'Strict-bin lambda'),
             ('QuietPaw/epsilon_bin_0/actor_cost_reward_gradient_ratio', 'Strict-bin cost/reward gradient ratio'),
             ('Train/mean_reward', 'Training episode reward'),
             ('QuietPaw/epsilon_bin_0/ppo_clip_fraction', 'Strict-bin probability ratio clip fraction')]
    for ax, (tag, title) in zip(axes.flatten(), specs):
        for arm, tags in curves.items():
            record = tags.get(tag)
            if record:
                values = np.asarray(record['values'])
                if len(values) > 20:
                    values = np.convolve(values, np.ones(20)/20, mode='valid')
                    steps = record['steps'][19:]
                else:
                    steps = record['steps']
                ax.plot(steps, values, label=arm)
        ax.set(title=title, xlabel='PPO update')
        ax.grid(alpha=.2)
    if curves:
        axes[0, 0].legend(fontsize=7)
    ax = axes[1, 2]
    for arm, records in evaluated.items():
        ordered = [r['summary'] for e, r in sorted(records.items())]
        ax.plot([r['observed_discounted_cost'] for r in ordered],
                [r['observed_discounted_reward'] for r in ordered], 'o-', label=arm)
        for row in ordered:
            ax.annotate(str(row['epsilon']), (row['observed_discounted_cost'],
                                             row['observed_discounted_reward']), fontsize=6)
    ax.set(title='Observed finite-horizon discounted returns',
           xlabel='Normalized discounted cost', ylabel='Discounted original reward')
    ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output.with_suffix('.png'), dpi=170)
    plt.close(figure)
    lines = ['# Stage 1 constraint validation results', '',
             'Status: **{}**.'.format('Complete pilot' if complete else 'Experiment still running'), '',
             'All arms use the same trained HIM initialization and matched transitions. '
             'The original HIM, locomotion reward, cost, physics, GAE and combined-advantage semantics are preserved.', '',
             '## Evaluation: command (1,0,0), seed 11', '',
             '| Arm | Epsilon | Discounted cost | Violation | Raw cost | Return | Tracking error | Survival | Vc vs deployment RMSE |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| {arm} | {epsilon:.3f} | {observed_discounted_cost:.6f} | '
                     '{observed_budget_violation:.6f} | {noise_cost:.6f} | {locomotion_reward:.3f} | '
                     '{velocity_tracking_error_mean:.4f} | {survival_success_rate:.2%} | '
                     '{cost_value_observed_rmse:.6f} |'.format(**row))
    lines += ['', '## Same-epsilon change relative to lambda-zero continuation', '',
              '| Arm | Epsilon | Discounted cost reduction | Raw cost reduction | Survival change (pp) | '
              'Tracking error change | Pilot screen |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for arm, records in comparisons.items():
        for epsilon, row in records.items():
            lines.append('| {} | {} | {:.2%} | {:.2%} | {:+.2f} | {:+.2%} | {} |'.format(
                arm, epsilon, row['cost_reduction_fraction'], row['raw_cost_reduction_fraction'],
                row['survival_change_percentage_points'], row['tracking_error_change_fraction'],
                'Pass' if row['screening_pass'] else 'Not passed'))
    lines += ['', '## Sampled physical diagnostics', '',
              '| Arm | Epsilon | Force cost | Impact cost | Fz peak mean (N) | '
              'dF peak mean (N/s) | Base acceleration (m/s²) | Touchdowns per alive step |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| {arm} | {epsilon:.3f} | {noise_force_cost_mean:.6f} | '
                     '{noise_impact_cost_mean:.6f} | {fz_peak_mean:.2f} | {df3d_peak_mean:.2f} | '
                     '{base_acceleration_mean:.3f} | {touchdown_events_mean:.4f} |'.format(**row))
    lines += ['', '## Interpretation limits', '',
              '- The screen is a declared pilot criterion, not a QuietPaw reproduction claim.',
              '- Confidence intervals in the JSON describe paired environment outcomes at one trained seed. '
              'They do not quantify uncertainty across independent training runs.',
              '- Fixed-budget arms receive more samples at their one condition. A difference from the grid arm '
              'does not uniquely identify a representation problem.',
              '- Discounted reward/cost omit the evaluation-horizon tail; cost critic calibration uses those '
              'physical observed returns. Bootstrapped cost estimates are reported separately.',
              '- The critic is trained for a stochastic Gaussian policy; deployment evaluation uses its deterministic '
              'mean action. Vc vs deployment RMSE therefore includes a policy mismatch and is not a pure '
              'critic calibration error for the training policy.',
              '- Lower discounted cost accompanied by lower survival needs explicit failure analysis. '
              'Per-alive-step cost, contact events, tracking, forces, and return are recorded alongside it.',
              '- 300 warm-start PPO updates do not establish convergence or infeasibility of a budget.', '',
              '- Force peaks are sampled at the 50-Hz control rate. The proxy is not an acoustic measurement, '
              'and per-foot monotonicity does not guarantee monotonicity of aggregate force cost under changes in contact count.', '',
              '![Training and evaluation](./{}.png)'.format(output.name), '',
              'Protocol: [quietpaw_constraint_validation_protocol.md](./quietpaw_constraint_validation_protocol.md).',
              'Machine-readable results: [{}](./{}).'.format(output.with_suffix('.json').name,
                                                             output.with_suffix('.json').name)]
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete': complete, 'evaluated_arms': list(evaluated),
                      'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
