"""Sequential execution of the preregistered fixed-budget mechanism study."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'legged_gym/logs/quietpaw_fixed_budget_20261009'
WORKER = Path(__file__).resolve().with_name('worker.py')
HIM = ROOT / 'legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_8000.pt'
PROTOCOL = ROOT / 'docs/quietpaw_fixed_budget_protocol_20261009.md'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.json'
    sources = list((ROOT / 'rsl_rl/rsl_rl').rglob('*.py')) + list((ROOT / 'legged_gym/legged_gym').rglob('*.py'))
    sources += list(WORKER.parent.glob('*.py'))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = dict(protocol_sha256=digest(PROTOCOL), him_sha256=digest(HIM),
                        protocol_date='2026-10-09',
                        sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
                        iterations=300, num_envs=4096, seeds=[4, 5], jobs={})

    def save():
        temp = manifest_path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        temp.replace(manifest_path)

    def check_frozen():
        if digest(PROTOCOL) != manifest['protocol_sha256'] or digest(HIM) != manifest['him_sha256']:
            raise RuntimeError('Protocol/source checkpoint changed')
        for path, expected in manifest['sources'].items():
            if digest(ROOT / path) != expected:
                raise RuntimeError('Frozen source changed: ' + path)

    def job(name, mode, seed, epsilon=0., arm='pid', checkpoint=None):
        check_frozen()
        folder = OUT / name
        expected = folder / ('model_300.pt' if mode == 'train' else 'result.json')
        record = manifest['jobs'].setdefault(name, {})
        command = [sys.executable, '-u', str(WORKER), '--mode', mode, '--output', str(folder),
                   '--him-checkpoint', str(HIM), '--seed', str(seed), '--epsilon', str(epsilon), '--arm', arm]
        if checkpoint:
            command += ['--checkpoint', str(checkpoint)]
        if mode == 'train' and arm == 'pid':
            command += ['--expected-initial', str(OUT / ('seed{}_zero'.format(seed)) / 'initial.json')]
        if record.get('complete'):
            if not expected.exists():
                raise RuntimeError('Missing completed artifact: ' + name)
            return folder
        # Calibration101 can have been launched during instrumentation checks.
        if expected.exists() and name == 'calibration101':
            data = json.loads(expected.read_text())
            if data['initial']['source_sha256'] != manifest['him_sha256'] or data['seed'] != 101:
                raise RuntimeError('Calibration provenance mismatch')
        else:
            if expected.exists():
                raise RuntimeError('Unregistered output exists: ' + name)
            record.update(command=command, start_time=time.time())
            save()
            print('START', name, flush=True)
            with (OUT / (name + '.log')).open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'WANDB_MODE': 'disabled'})
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print('RUNNING', name, 'seconds', int(time.time() - record['start_time']), flush=True)
                if process.returncode:
                    raise RuntimeError('Worker failed: ' + name)
        record.update(complete=True, end_time=time.time(), output=str(folder), command=command)
        save()
        print('COMPLETE', name, flush=True)
        return folder

    save()
    calibrations = []
    for seed in (101, 102):
        folder = job('calibration' + str(seed), 'calibrate', seed)
        result = json.loads((folder / 'result.json').read_text())
        if result['summary']['tail_bound_max'] > 1e-4:
            raise RuntimeError('Calibration timeout tail exceeds preregistered tolerance')
        calibrations.append(result['summary']['observed_discounted_cost'])
    budget = round(.9 * sum(calibrations) / 2, 6)
    if 'epsilon' in manifest and manifest['epsilon'] != budget:
        raise RuntimeError('Budget changed')
    manifest['epsilon'] = budget
    manifest['calibration_means'] = calibrations
    save()
    print('FROZEN_BUDGET', budget, calibrations, flush=True)

    # Predeclared initial stochastic-policy reference evaluations.
    for eval_seed in (111, 112):
        job('initial_eval' + str(eval_seed), 'evaluate', eval_seed, budget)
    job('initial_deploy11', 'deploy', 11, budget)

    for seed in (4, 5):
        initial = None
        for arm in ('zero', 'pid'):
            folder = job('seed{}_{}'.format(seed, arm), 'train', seed, budget, arm)
            current = json.loads((folder / 'initial.json').read_text())
            if initial is not None and current != initial:
                raise RuntimeError('Paired initial weights/optimizer/simulator/RNG mismatch')
            initial = current
            checkpoint = folder / 'model_300.pt'
            for eval_seed in (111, 112):
                job('seed{}_{}_eval{}'.format(seed, arm, eval_seed), 'evaluate', eval_seed,
                    budget, arm, checkpoint)
            job('seed{}_{}_deploy11'.format(seed, arm), 'deploy', 11, budget, arm, checkpoint)
    manifest['complete'] = True
    save()
    print('STUDY_COMPLETE', str(manifest_path), flush=True)


if __name__ == '__main__':
    main()
