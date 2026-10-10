"""Train the V1 quiet adapter. Use the local packages, not an installed checkout."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'rsl_rl'))
sys.path.insert(0, str(REPO / 'legged_gym'))

import isaacgym  # Isaac Gym must be imported before torch.
from isaacgym import gymutil
import copy
import hashlib
import json
import random
import subprocess
import tempfile
import time
from types import SimpleNamespace
import numpy as np
import torch

from legged_gym.envs import task_registry
from legged_gym.envs.go1.quiet_config import Go1QuietCfg, Go1QuietCfgPPO
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.algorithms import HIMPPO
from rsl_rl.modules import QuietHIMActorCritic
from rsl_rl.utils.quiet_checkpoint import (load_base_checkpoint, load_quiet_checkpoint,
                                          save_quiet_checkpoint, file_sha256)


def get_quiet_args():
    parameters = [
        {'name': '--base_checkpoint', 'type': str},
        {'name': '--resume_checkpoint', 'type': str},
        {'name': '--adaptation', 'type': str, 'default': 'lora',
         'choices': ['frozen', 'head', 'lora', 'full']},
        {'name': '--num_envs', 'type': int, 'default': 256},
        {'name': '--max_iterations', 'type': int, 'default': 200},
        {'name': '--seed', 'type': int, 'default': 1},
        {'name': '--rl_device', 'type': str, 'default': 'cuda:0'},
        {'name': '--headless', 'action': 'store_true'},
        {'name': '--output', 'type': str},
        {'name': '--eval_interval', 'type': int, 'default': 50},
        {'name': '--eval_episodes', 'type': int, 'default': 32},
        {'name': '--eval_duration', 'type': float, 'default': 20.0},
        {'name': '--calibration_steps', 'type': int, 'default': 1000},
        {'name': '--quiet_fraction', 'type': float, 'default': 0.1},
        {'name': '--ramp_iterations', 'type': int, 'default': 50},
        {'name': '--skip_evaluation', 'action': 'store_true',
         'help': 'Diagnostic smoke run only; produces no selected policy.'},
    ]
    args = gymutil.parse_arguments(description='Frozen HIM quiet adaptation',
                                   custom_parameters=parameters)
    args.explicit_options = {argument.split('=', 1)[0] for argument in sys.argv[1:]
                             if argument.startswith('--')}
    if bool(args.base_checkpoint) == bool(args.resume_checkpoint):
        raise ValueError('Specify exactly one of --base_checkpoint and --resume_checkpoint')
    if min(args.num_envs, args.max_iterations, args.eval_interval,
           args.eval_episodes, args.calibration_steps, args.ramp_iterations) <= 0:
        raise ValueError('Environment counts, steps and intervals must be positive')
    if args.eval_duration <= 0 or args.quiet_fraction < 0:
        raise ValueError('Duration must be positive and quiet fraction nonnegative')
    args.sim_device = args.rl_device
    # TaskRegistry accepts the standard runner flags. Quiet initialization is
    # explicit and never goes through the legacy implicit resume mechanism.
    for key in ('experiment_name', 'run_name', 'load_run', 'checkpoint'):
        setattr(args, key, None)
    args.resume = False
    return args


def json_write(path, value):
    with open(str(path), 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def implementation_fingerprint():
    """Include new, uncommitted source files in the experiment identity."""
    digest = hashlib.sha256()
    for directory in ('legged_gym/legged_gym', 'rsl_rl/rsl_rl'):
        for path in sorted((REPO / directory).rglob('*.py')):
            digest.update(str(path.relative_to(REPO)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def apply_saved_config(target, values):
    """Restore BaseConfig instances without replacing their nested objects."""
    for name, value in values.items():
        existing = getattr(target, name, None)
        # class_to_dict includes the public BaseConfig initialization method as
        # an empty dict. Keep that method callable when restoring its config.
        if callable(existing) and value == {}:
            continue
        if isinstance(value, dict) and not isinstance(existing, dict):
            if existing is None or not hasattr(existing, '__dict__'):
                existing = SimpleNamespace()
                setattr(target, name, existing)
            apply_saved_config(existing, value)
        else:
            setattr(target, name, copy.deepcopy(value))


def restore_resume_configuration(args, cfg, train_cfg, metadata):
    """Restore the learning experiment before constructing simulation or PPO.

    A resume can change hardware, environment count, output location and the
    number of additional updates. It cannot silently change the learning task.
    Older V1 files omitted runner/evaluation settings; use the original V1
    defaults for those fields and record that compatibility choice.
    """
    required = {'environment_config', 'algorithm_config', 'calibration', 'seed',
                'quiet_fraction', 'ramp_iterations', 'policy'}
    if not required.issubset(metadata):
        raise ValueError('Resume metadata is missing: ' + str(sorted(required - set(metadata))))
    apply_saved_config(cfg, metadata['environment_config'])
    apply_saved_config(train_cfg.algorithm, metadata['algorithm_config'])
    if 'runner_config' in metadata:
        apply_saved_config(train_cfg.runner, metadata['runner_config'])
    else:
        # The original V1 rollout contains exactly 100 transitions per robot.
        train_cfg.runner.num_steps_per_env = 100
    protocol = metadata.get('evaluation_protocol', {
        'eval_interval': 50, 'eval_episodes': 32, 'eval_duration': 20.0,
        'skip_evaluation': bool(metadata.get('diagnostic_only', False)),
    })
    saved_options = dict(protocol)
    saved_options.update({
        'seed': metadata['seed'],
        'adaptation': metadata['policy']['adaptation'],
        'quiet_fraction': metadata['quiet_fraction'],
        'ramp_iterations': metadata['ramp_iterations'],
        'calibration_steps': metadata['calibration']['steps'],
    })
    explicit = getattr(args, 'explicit_options', set())
    for name, saved_value in saved_options.items():
        if '--' + name in explicit and getattr(args, name) != saved_value:
            raise ValueError('Resume must preserve --' + name + '; saved value is ' + str(saved_value))
        setattr(args, name, saved_value)
    if '--num_envs' not in explicit:
        args.num_envs = cfg.env.num_envs
    if cfg.seed != args.seed:
        raise ValueError('Saved environment seed differs from experiment seed')
    if class_to_dict(cfg) != metadata['environment_config']:
        raise ValueError('Saved environment config cannot be restored exactly')
    if class_to_dict(train_cfg.algorithm) != metadata['algorithm_config']:
        raise ValueError('Saved algorithm config cannot be restored exactly')
    cfg.env.num_envs = args.num_envs
    return protocol


def calibrate(env, model, steps):
    model.set_adapter_enabled(False)
    env.reset()
    total_cost = 0.0
    tracking = 0.0
    exposure = 0.0
    count = 0
    with torch.no_grad():
        for _ in range(steps):
            obs = env.get_observations()
            _, _, _, _, infos, _, _ = env.step(model.act_inference(obs))
            data = infos['quiet']
            valid = data['episode_time'] >= env.cfg.quiet.startup_time
            total_cost += data['touchdown_cost'][valid].sum().item()
            tracking += data['positive_tracking_rate'][valid].sum().item() * env.dt
            exposure += valid.sum().item() * env.dt
            count += data['touchdown_count'][valid].sum().item()
    model.set_adapter_enabled(True)
    if count < 32 or exposure <= 0 or total_cost <= 0:
        raise RuntimeError('Baseline calibration failed: insufficient valid touchdown events')
    return {'touchdown_reference': total_cost / exposure,
            'positive_tracking_reference': tracking / exposure,
            'events': int(count), 'exposure_seconds': exposure, 'steps': steps}


def rng_state():
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
            'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def evaluate(model, args, adapter_enabled, split='validation'):
    """Keep evaluation PhysX in a separate process from the live training sim.

    Concurrent GPU simulations in one Isaac Gym process can crash its native
    runtime. A snapshot also prevents evaluation from changing training RNG,
    model mode, device or observation history.
    """
    with tempfile.TemporaryDirectory(prefix='him_quiet_evaluation_') as directory:
        checkpoint_path = Path(directory) / 'policy.pt'
        report_path = Path(directory) / 'report.json'
        save_quiet_checkpoint(model, str(checkpoint_path))
        command = [sys.executable, '-u', str(REPO / 'legged_gym' / 'legged_gym'
                                            / 'scripts' / 'evaluate_quiet.py'),
                   '--checkpoint', str(checkpoint_path), '--output', str(report_path),
                   '--adapter', 'enabled' if adapter_enabled else 'disabled',
                   '--suite', 'all', '--split', split,
                   '--episodes', str(args.eval_episodes),
                   '--duration', str(args.eval_duration),
                   '--rl_device', args.rl_device]
        if args.headless:
            command.append('--headless')
        subprocess.run(command, cwd=str(REPO), check=True)
        with report_path.open() as stream:
            report = json.load(stream)
        # The temporary file is removed. Keep its content hash as provenance.
        report['checkpoint'] = 'training_policy_snapshot'
        report['adapter_enabled'] = bool(adapter_enabled)
        return report


def train(args):
    output = Path(args.output or REPO / 'legged_gym' / 'logs' / 'quiet_go1'
                  / time.strftime('%Y%m%d_%H%M%S'))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory so prior experiments are preserved')
    output.mkdir(parents=True, exist_ok=True)
    cfg, train_cfg = Go1QuietCfg(), Go1QuietCfgPPO()
    saved = None
    if args.resume_checkpoint:
        saved = torch.load(args.resume_checkpoint, map_location='cpu')
        restore_resume_configuration(args, cfg, train_cfg, saved['quiet_metadata'])
    else:
        cfg.env.num_envs = args.num_envs
        cfg.seed = args.seed
    if not args.skip_evaluation:
        from legged_gym.utils.quiet_evaluation import (
            evaluation_conditions, validate_evaluation_schedule)
        validate_evaluation_schedule(evaluation_conditions('all'), args.eval_duration,
                                     cfg.sim.dt * cfg.control.decimation)
    env, _ = task_registry.make_env('go1_quiet', args=args, env_cfg=cfg)
    model_args = dict(num_actor_obs=env.num_obs, num_critic_obs=env.num_privileged_obs,
                      num_one_step_obs=env.num_one_step_obs, num_actions=env.num_actions,
                      adaptation=args.adaptation)
    if saved is not None:
        model_args = saved['quiet_metadata']['policy']
    model = QuietHIMActorCritic(**model_args).to(env.device)
    algorithm_cfg = class_to_dict(train_cfg.algorithm)
    if algorithm_cfg.get('update_estimator') is not False:
        raise ValueError('Quiet adaptation must disable the independent estimator optimizer')
    alg = HIMPPO(model, device=env.device, **algorithm_cfg)
    alg.init_storage(env.num_envs, train_cfg.runner.num_steps_per_env,
                     [env.num_obs], [env.num_privileged_obs], [env.num_actions])
    if args.resume_checkpoint:
        checkpoint = load_quiet_checkpoint(model, args.resume_checkpoint, alg.optimizer)
        metadata = copy.deepcopy(checkpoint['quiet_metadata'])
        start_iteration = checkpoint['iter']
        calibration = metadata['calibration']
        if (checkpoint.get('infos') or {}).get('rng_state') is not None:
            restore_rng(checkpoint['infos']['rng_state'])
    else:
        metadata = load_base_checkpoint(model, args.base_checkpoint)
        start_iteration = 0
        calibration = calibrate(env, model, args.calibration_steps)
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip()
    if saved is None:
        metadata.update({
            'calibration': calibration, 'quiet_fraction': args.quiet_fraction,
            'ramp_iterations': args.ramp_iterations, 'seed': args.seed,
            'environment_config': class_to_dict(cfg), 'algorithm_config': algorithm_cfg,
            'runner_config': class_to_dict(train_cfg.runner),
            'code_commit': code_commit, 'implementation_sha256': implementation_fingerprint(),
            'torch_version': torch.__version__, 'python_version': sys.version,
            'diagnostic_only': args.skip_evaluation,
        })
    # Preserve original provenance on resume. Only append runtime information.
    metadata.setdefault('runner_config', class_to_dict(train_cfg.runner))
    metadata.setdefault('evaluation_protocol', {
        'eval_interval': args.eval_interval, 'eval_episodes': args.eval_episodes,
        'eval_duration': args.eval_duration, 'skip_evaluation': args.skip_evaluation,
    })
    runtime = {
        'output': str(output.resolve()), 'num_envs': env.num_envs,
        'device': str(env.device), 'start_iteration': start_iteration,
        'additional_iterations': args.max_iterations, 'code_commit': code_commit,
        'torch_version': torch.__version__, 'implementation_sha256': implementation_fingerprint(),
    }
    if saved is not None:
        runtime.update({'resume_checkpoint': str(Path(args.resume_checkpoint).resolve()),
                        'resume_checkpoint_sha256': file_sha256(args.resume_checkpoint),
                        'simulator_state_restored': False})
        missing_legacy = [field for field in ('runner_config', 'evaluation_protocol')
                          if field not in saved['quiet_metadata']]
        if missing_legacy:
            runtime['legacy_v1_defaults'] = missing_legacy
    metadata.setdefault('runtime_segments', []).append(runtime)
    env.set_quiet_reward(calibration['touchdown_reference'],
                         args.quiet_fraction * calibration['positive_tracking_reference'], 0.0)
    baseline = None
    if not args.skip_evaluation:
        from legged_gym.utils.quiet_evaluation import QUIET_EVALUATION_FORMAT_VERSION
        baseline = copy.deepcopy(metadata.get('baseline_validation'))
        if baseline is not None and baseline.get('format_version') != QUIET_EVALUATION_FORMAT_VERSION:
            # Reports from the old ambiguous action-saturation metric cannot
            # supply a paired baseline for the new explicit action limits.
            baseline = None
        if baseline is None:
            baseline = evaluate(model, args, False)
            metadata['baseline_validation'] = copy.deepcopy(baseline)
        json_write(output / 'baseline_validation.json', baseline)
    json_write(output / 'config.json', metadata)
    env.reset()
    obs, critic_obs = env.get_observations(), env.get_privileged_observations()
    frozen = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
    initial_adapters = {k: p.detach().clone() for k, p in model.actor.named_parameters() if p.requires_grad}
    best_cost = float('inf')
    if saved is not None and baseline is not None:
        # A new output directory has no archived selected weights. Screen the
        # resumed model itself instead of carrying an unbacked historical score.
        from legged_gym.utils.quiet_evaluation import select_candidate
        resume_report = evaluate(model, args, True)
        resume_screening = select_candidate(resume_report, baseline)
        json_write(output / 'resume_validation.json',
                   {'evaluation': resume_report, 'screening': resume_screening})
        if resume_screening['accepted']:
            best_cost = resume_screening['score']
            save_quiet_checkpoint(model, str(output / 'selected.pt'), alg.optimizer,
                                  start_iteration, metadata,
                                  {'screening': resume_screening, 'rng_state': rng_state()})
    history = []
    for iteration in range(start_iteration, start_iteration + args.max_iterations):
        model.train()
        env.set_quiet_progress(min(1.0, (iteration + 1) / float(args.ramp_iterations)))
        started = time.time()
        cost_sum, touchdowns = 0.0, 0
        with torch.inference_mode():
            for _ in range(train_cfg.runner.num_steps_per_env):
                actions = alg.act(obs, critic_obs)
                obs, critic_obs, rewards, dones, infos, ids, terminal_obs = env.step(actions)
                if not torch.isfinite(rewards).all():
                    raise RuntimeError('Non-finite quiet reward')
                next_critic_obs = critic_obs.clone()
                next_critic_obs[ids] = terminal_obs
                alg.process_env_step(rewards, dones, infos, next_critic_obs)
                cost_sum += infos['quiet']['touchdown_cost'].sum().item()
                touchdowns += infos['quiet']['touchdown_count'].sum().item()
            alg.compute_returns(critic_obs)
        if not torch.isfinite(alg.storage.returns).all() or not torch.isfinite(alg.storage.advantages).all():
            raise RuntimeError('Non-finite PPO returns or advantages')
        losses = alg.update()
        for key, parameter in model.named_parameters():
            if key in frozen and not torch.equal(parameter, frozen[key]):
                raise RuntimeError('Frozen parameter changed: ' + key)
            if not torch.isfinite(parameter).all():
                raise RuntimeError('Non-finite model parameter: ' + key)
        if (model.std <= 0).any():
            raise RuntimeError('PPO produced a nonpositive Gaussian standard deviation')
        row = {'iteration': iteration + 1, 'seconds': time.time() - started,
               'value_loss': losses[0], 'surrogate_loss': losses[1],
               'touchdown_cost': cost_sum, 'touchdown_count': int(touchdowns),
               'quiet_weight': env.quiet_target_weight * env.quiet_progress,
               'adapter_update_norm': sum((p - initial_adapters[k]).square().sum().item()
                                          for k, p in model.actor.named_parameters() if k in initial_adapters) ** 0.5}
        row.update(alg.last_update_metrics)
        history.append(row)
        print(json.dumps(row), flush=True)
        with open(str(output / 'training.jsonl'), 'a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        completed = iteration + 1
        if completed % args.eval_interval == 0 or completed == start_iteration + args.max_iterations:
            checkpoint_path = output / ('model_%d.pt' % completed)
            save_quiet_checkpoint(model, str(checkpoint_path), alg.optimizer, completed,
                                  metadata, {'rng_state': rng_state()})
            if baseline is not None:
                from legged_gym.utils.quiet_evaluation import select_candidate
                report = evaluate(model, args, True)
                screening = select_candidate(report, baseline)
                json_write(output / ('validation_%d.json' % completed),
                           {'evaluation': report, 'screening': screening})
                if screening['accepted'] and screening['score'] < best_cost:
                    best_cost = screening['score']
                    save_quiet_checkpoint(model, str(output / 'selected.pt'), alg.optimizer,
                                          completed, metadata,
                                          {'screening': screening, 'rng_state': rng_state()})
    selected = output / 'selected.pt'
    if selected.exists():
        load_quiet_checkpoint(model, str(selected))
        baseline_test = evaluate(model, args, False, split='test')
        candidate_test = evaluate(model, args, True, split='test')
        from legged_gym.utils.quiet_evaluation import select_candidate
        test_screening = select_candidate(candidate_test, baseline_test)
        json_write(output / 'test.json', {'baseline': baseline_test, 'candidate': candidate_test,
                                         'screening': test_screening})
    json_write(output / 'summary.json', {
        'iterations': len(history), 'frozen_parameters_unchanged': True,
        'adapter_parameters': sum(p.numel() for p in model.adapter_parameters()),
        'calibration': calibration, 'selected_checkpoint': str(selected) if selected.exists() else None,
        'diagnostic_only': args.skip_evaluation,
        'note': 'No qualified policy selected' if not selected.exists() else 'See held-out test.json',
    })
    env.gym.destroy_sim(env.sim)
    print('Results: ' + str(output), flush=True)


if __name__ == '__main__':
    train(get_quiet_args())
