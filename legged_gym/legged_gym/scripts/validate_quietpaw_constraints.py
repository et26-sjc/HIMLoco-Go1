"""Sequential, matched-transition Stage 1 PID and fixed-budget validation.

All runs start from the same original HIM checkpoint. The study writes a
manifest before training and resumes only arms with a verified final model.
GPU children run sequentially; no baseline artifacts are moved or overwritten.
"""

import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / 'legged_gym/logs/quietpaw_go1_stage1'
FROZEN = (
    'rsl_rl/rsl_rl/modules/him_estimator.py',
    'rsl_rl/rsl_rl/modules/him_actor_critic.py',
    'rsl_rl/rsl_rl/algorithms/him_ppo.py',
    'rsl_rl/rsl_rl/storage/him_rollout_storage.py',
    'rsl_rl/rsl_rl/storage/quietpaw_cmdp_storage.py',
    'legged_gym/legged_gym/envs/base/legged_robot.py',
    'legged_gym/legged_gym/envs/go1/go1_config.py',
    'legged_gym/legged_gym/envs/go1/quietpaw_legged_robot.py',
    'legged_gym/legged_gym/envs/go1/quietpaw_noise.py',
    'legged_gym/legged_gym/envs/go1/quietpaw_go1_config.py',
    'rsl_rl/rsl_rl/algorithms/quietpaw_constrained_ppo.py',
    'rsl_rl/rsl_rl/runners/quietpaw_stage1_runner.py',
    'rsl_rl/rsl_rl/utils/quietpaw_diagnostics.py',
    'legged_gym/legged_gym/scripts/train_quietpaw.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(path, manifest):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    temporary.replace(path)


def latest_progress(log_path):
    with log_path.open('rb') as log:
        log.seek(0, 2)
        log.seek(max(0, log.tell() - 200000))
        tail = log.read().decode('utf-8', errors='replace')
    iterations = re.findall(r'Learning iteration (\d+)/(\d+)', tail)
    metrics = re.findall(r'^QuietPaw: (\{.*\})$', tail, flags=re.MULTILINE)
    result = {}
    if iterations:
        result['iteration'] = int(iterations[-1][0])
    if metrics:
        try:
            values = ast.literal_eval(metrics[-1])
            for key in ('noise_cost_mean', 'completed_cost_mean', 'lambda_next_mean',
                        'lambda_next_max', 'budget_violation_mean'):
                result[key] = values[key]
        except (ValueError, SyntaxError, KeyError):
            pass
    return result


def child(command, log_path, label):
    env = os.environ.copy()
    env['WANDB_MODE'] = 'disabled'
    env['MPLCONFIGDIR'] = '/tmp/quietpaw_validation_mpl'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(json.dumps({'event': 'progress', 'arm': label,
                                      **latest_progress(log_path)}), flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode:
        raise RuntimeError('{} failed with exit {}: {}'.format(label, process.returncode, log_path))


def model_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str((str(value.dtype), tuple(value.shape))).encode())
        h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_training(run, iterations, env_count):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    state = torch.load(run / ('model_{}.pt'.format(iterations)), map_location='cpu')
    if state['iter'] != iterations:
        raise ValueError('Checkpoint iteration mismatch')
    if any(not torch.isfinite(v).all() for v in state['model_state_dict'].values()):
        raise ValueError('Non-finite final model')
    event = EventAccumulator(str(next(run.glob('events.out.tfevents.*'))),
                             size_guidance={'scalars': 0}).Reload()
    records = event.Scalars('QuietPaw/him_updates')
    if len(records) != iterations or records[-1].step != iterations - 1:
        raise ValueError('Incomplete training log')
    if any(record.value != 20 for record in records):
        raise ValueError('HIM auxiliary update count changed')
    if any(not math.isfinite(s.value) for tag in event.Tags()['scalars'] for s in event.Scalars(tag)):
        raise ValueError('Non-finite training metric')
    summary = {}
    for tag in event.Tags()['scalars']:
        values = [s.value for s in event.Scalars(tag) if s.step >= iterations - 50]
        if values:
            summary[tag] = sum(values) / len(values)
    initial = torch.load(run / 'model_him_init.pt', map_location='cpu')
    return dict(iterations=iterations, transitions=iterations * env_count * 100,
                initial_model_tensor_sha256=model_digest(initial['model_state_dict']),
                final_model_tensor_sha256=model_digest(state['model_state_dict']),
                contract=state['quietpaw']['contract'],
                initialization=state['quietpaw']['initialization'],
                last50_training_means=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--him-checkpoint', required=True)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--arms', nargs='+', default=['grid_g0', 'grid_g1', 'grid_g10',
                                                    'grid_g100', 'fixed085_g100', 'fixed110_g100'])
    parser.add_argument('--resume-study', action='store_true')
    args = parser.parse_args()
    definitions = {
        'grid_g0': (0., None), 'grid_g1': (1., None),
        'grid_g10': (10., None), 'grid_g100': (100., None),
        'fixed085_g100': (100., .085), 'fixed110_g100': (100., .11),
    }
    if any(a not in definitions for a in args.arms) or args.iterations < 1:
        raise ValueError('Unknown arm or nonpositive iterations')
    out = Path(args.output).resolve()
    path = out / 'manifest.json'
    source = Path(args.him_checkpoint).resolve(strict=True)
    configuration = {'iterations': args.iterations, 'num_envs': args.num_envs,
                     'training_seed': args.seed, 'arms': args.arms,
                     'source': str(source), 'source_sha256': digest(source)}
    if path.exists():
        if not args.resume_study:
            raise ValueError('Study exists; use --resume-study')
        manifest = json.loads(path.read_text())
        if manifest['configuration'] != configuration:
            raise ValueError('Study configuration mismatch')
    else:
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            'configuration': configuration,
            'started_at': datetime.now().isoformat(),
            'frozen_source_sha256': {f: digest(ROOT / f) for f in FROZEN},
            'protocol': {
                'cost': 'unchanged normalized_complement_v2, weights 0.2/0.8',
                'grid': [.08, .12], 'fixed_budgets': [.085, .11],
                'diagnostics_interval': 50,
                'primary_evaluation': {'seed': 11, 'command': [1, 0, 0], 'steps': 1000,
                                       'epsilons_grid': [.08, .085, .095, .11, .12]},
                'pilot_screening': {'cost_reduction_relative_to_lambda0': .03,
                                    'maximum_survival_drop_percentage_points': 5,
                                    'maximum_tracking_error_increase_fraction': .15},
                'interpretation': 'Pilot, one training seed; thresholds are declared screening criteria, not paper claims.'},
            'arms': {},
        }
        write_manifest(path, manifest)
    for arm in args.arms:
        if any(digest(ROOT / f) != value for f, value in manifest['frozen_source_sha256'].items()):
            raise ValueError('Frozen training/measurement source changed during the study')
        gain, fixed = definitions[arm]
        record = manifest['arms'].setdefault(arm, {})
        if record.get('status') == 'evaluated':
            continue
        if record.get('status') != 'trained':
            run_name = '{}_{}_seed{}_{}'.format(out.name, arm, args.seed, datetime.now().strftime('%H%M%S'))
            command = [sys.executable, '-u', str(ROOT / 'legged_gym/legged_gym/scripts/train_quietpaw.py'),
                       '--task', 'go1_quietpaw_stage1', '--headless', '--num_envs', str(args.num_envs),
                       '--max_iterations', str(args.iterations), '--seed', str(args.seed),
                       '--run_name', run_name, '--him_checkpoint', str(source), '--noise_force_weight', '.2',
                       '--pid_gain_scale', str(gain), '--constraint_diagnostics_interval', '50']
            if fixed is None:
                command += ['--epsilon_min', '.08', '--epsilon_max', '.12']
            else:
                command += ['--epsilon_fixed', str(fixed)]
            record.update(status='training', command=command, started_at=datetime.now().isoformat(),
                          gain_scale=gain, fixed_epsilon=fixed)
            write_manifest(path, manifest)
            print(json.dumps({'event': 'training_started', 'arm': arm, 'iterations': args.iterations}), flush=True)
            child(command, out / (arm + '_training.log'), arm)
            matches = list(LOGS.glob('*_' + run_name))
            if len(matches) != 1:
                raise ValueError('Training run directory is not unique')
            record.update(status='trained', run_directory=str(matches[0]),
                          verification=verify_training(matches[0], args.iterations, args.num_envs))
            if record['verification']['initialization']['source_sha256'] != configuration['source_sha256']:
                raise ValueError('HIM source mismatch')
            initial_hashes = {r['verification']['initial_model_tensor_sha256']
                              for r in manifest['arms'].values() if 'verification' in r}
            if len(initial_hashes) != 1:
                raise ValueError('Initial network tensors differ across experiments')
            write_manifest(path, manifest)
        checkpoint = Path(record['run_directory']) / ('model_{}.pt'.format(args.iterations))
        epsilon = [.08, .085, .095, .11, .12] if fixed is None else [fixed]
        output = out / (arm + '_eval_cmd1_seed11')
        command = [sys.executable, '-u', str(ROOT / 'legged_gym/legged_gym/scripts/evaluate_quietpaw.py'),
                   '--checkpoint', str(checkpoint), '--output', str(output), '--epsilons', *map(str, epsilon),
                   '--num-envs', str(args.num_envs), '--steps', '1000', '--seed', '11', '--command', '1', '0', '0']
        record['evaluation_command'] = command
        write_manifest(path, manifest)
        child(command, out / (arm + '_evaluation.log'), arm + '_evaluation')
        record.update(status='evaluated', evaluation_directory=str(output),
                      summaries=json.loads((output / 'summary.json').read_text()),
                      completed_at=datetime.now().isoformat())
        write_manifest(path, manifest)
        print(json.dumps({'event': 'evaluation_complete', 'arm': arm,
                          'summaries': [{k: r[k] for k in ('epsilon', 'noise_cost', 'observed_discounted_cost',
                                                          'locomotion_reward', 'survival_success_rate',
                                                          'velocity_tracking_error_mean')}
                                        for r in record['summaries']]}), flush=True)
    manifest['completed_at'] = datetime.now().isoformat()
    write_manifest(path, manifest)
    print('STUDY_COMPLETE ' + str(path), flush=True)


if __name__ == '__main__':
    main()
