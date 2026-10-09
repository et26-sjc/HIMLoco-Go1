"""Frozen study workers. Run through run_study.py for the preregistered matrix."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def tensor_digest(value):
    """Stable recursive digest for weights, optimizer dictionaries and RNG."""
    import torch
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def main(args):
    import isaacgym
    import torch
    from legged_gym.envs.go1.quietpaw_go1_config import QuietPawStage1Go1Cfg, QuietPawStage1Go1CfgPPO
    from legged_gym.envs.go1.quietpaw_legged_robot import QuietPawLeggedRobot
    from legged_gym.utils import get_args, task_registry
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners.quietpaw_stage1_runner import QuietPawStage1Runner
    from probes import gradient_probe

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    class AuditRobot(QuietPawLeggedRobot):
        def reset(self):
            self.reset_idx(torch.arange(self.num_envs, device=self.device))
            self.initial_hash = tensor_digest({name: getattr(self, name) for name in
                ('root_states', 'dof_pos', 'dof_vel', 'terrain_types', 'terrain_levels',
                 'commands', 'Kp_factors', 'Kd_factors', 'motor_strength_factors',
                 'friction_coeffs', 'payload')})
            result = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device))
            return result[:2]

        def _post_physics_step_callback(self):
            super()._post_physics_step_callback()
            if args.mode == 'deploy':
                self.commands[:, :3] = torch.tensor([1., 0., 0.], device=self.device)

        def _resample_commands(self, ids):
            super()._resample_commands(ids)
            if args.mode == 'deploy':
                self.commands[:, :3] = torch.tensor([1., 0., 0.], device=self.device)

        def _update_terrain_curriculum(self, ids):
            if args.mode != 'deploy':
                super()._update_terrain_curriculum(ids)

        def update_command_curriculum(self, ids):
            if args.mode != 'deploy':
                super().update_command_curriculum(ids)

    cfg, train = QuietPawStage1Go1Cfg(), QuietPawStage1Go1CfgPPO()
    cfg.noise_cost.force_weight, cfg.noise_cost.impact_weight = 0., 1.
    train.algorithm.epsilon_fixed = args.epsilon
    train.algorithm.epsilon_min = train.algorithm.epsilon_max = args.epsilon
    train.algorithm.diagnostics_interval = 50
    if args.arm == 'zero':
        train.algorithm.pid.kp = train.algorithm.pid.ki = train.algorithm.pid.kd = 0.
    train.runner.save_interval = 100
    train.runner.max_iterations = args.iterations
    sys.argv = [sys.argv[0], '--headless', '--num_envs', str(args.num_envs), '--seed', str(args.seed)]
    gym_args = get_args()
    task_registry.task_classes['quietpaw_fixed_study'] = AuditRobot
    env, _ = task_registry.make_env('quietpaw_fixed_study', args=gym_args, env_cfg=cfg)
    runner = QuietPawStage1Runner(env, class_to_dict(train),
                                 str(out) if args.mode == 'train' else None,
                                 device=env.device, use_wandb=False)
    alg, model = runner.alg, runner.alg.actor_critic
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=env.device)
        contract = saved['quietpaw']['contract']
        assert contract['noise_parameters'] == runner.contract['noise_parameters']
        assert contract['algorithm']['gamma'] == alg.gamma
        model.load_state_dict(saved['model_state_dict'])
        source_path = args.checkpoint
    else:
        runner.initialize_from_him(args.him_checkpoint)
        source_path = args.him_checkpoint
    initial = dict(model=tensor_digest(model.state_dict()),
                   optimizer=tensor_digest(alg.optimizer.state_dict()),
                   him_optimizer=tensor_digest(model.estimator.optimizer.state_dict()),
                   simulator_pre_reset_step=env.initial_hash,
                   torch_rng=tensor_digest(torch.get_rng_state()),
                   cuda_rng=tensor_digest(torch.cuda.get_rng_state_all()),
                   source_sha256=hashlib.sha256(Path(source_path).read_bytes()).hexdigest())
    (out / 'initial.json').write_text(json.dumps(initial, indent=2))
    if args.expected_initial:
        expected = json.loads(Path(args.expected_initial).read_text())
        if initial != expected:
            raise RuntimeError('Paired initialization mismatch before training: ' +
                               str([k for k in initial if initial[k] != expected[k]]))
    (out / 'environment.json').write_text(json.dumps(class_to_dict(cfg), indent=2))
    if args.mode == 'train':
        runner.save(str(out / 'model_him_init.pt'))
        update = alg.update
        def instrumented_update():
            iteration = alg.iterations
            record = {'iteration': iteration}
            selected = iteration % 50 == 0 or iteration == args.iterations - 1
            if selected:
                record['before'] = gradient_probe(model, alg.storage, alg.lambda_used, alg.clip_param)
            result = update()
            if selected:
                record['after'] = gradient_probe(model, alg.storage, alg.lambda_used, alg.clip_param)
                record['lambda_used_mean'] = alg.lambda_used.mean().item()
                with (out / 'gradients.jsonl').open('a') as file:
                    file.write(json.dumps(record, allow_nan=False) + '\n')
            return result
        alg.update = instrumented_update
        runner.learn(args.iterations, init_at_random_ep_len=True)
    else:
        model.eval()
        # Reset-step observation is retained; full episode clock begins here.
        env.episode_length_buf.zero_()
        obs = env.get_observations()
        critic = env.get_privileged_observations()
        n, device = env.num_envs, env.device
        active = torch.ones(n, dtype=torch.bool, device=device)
        discount = torch.ones(n, device=device)
        cost_sum, raw_sum, reward_sum, tracking_sum, steps, touchdowns = [torch.zeros(n, device=device) for _ in range(6)]
        terminal_tail = torch.zeros(n, device=device)
        tail_bound = torch.zeros(n, device=device)
        failures = torch.zeros(n, dtype=torch.bool, device=device)
        observer = None
        if args.mode == 'deploy':
            from observer import PhysicalObserver, GymProxy
            assert cfg.control.control_type == 'P'
            observer = PhysicalObserver(env, env.gym)
            env.gym = GymProxy(env.gym, observer)
        with torch.no_grad():
            v_initial = model.evaluate_cost(critic, alg.epsilon).flatten().clone()
            horizon = 1000 if args.mode == 'deploy' else int(env.max_episode_length) + 1
            for tick in range(horizon):
                actions = model.act_inference(obs, alg.epsilon) if args.mode == 'deploy' else model.act(obs, alg.epsilon)
                obs, critic, reward, done, info, ids, terminal = env.step(actions)
                metrics = info['quietpaw']
                cost = metrics['noise_cost']
                cost_sum += discount * (1 - alg.gamma) * cost * active
                raw_sum += cost * active
                reward_sum += reward * active
                tracking_sum += metrics['velocity_tracking_error'] * active
                touchdowns += metrics['touchdown_events'] * active
                steps += active
                discount *= torch.where(active, torch.full_like(discount, alg.gamma), torch.ones_like(discount))
                ending = active & (done.bool() | (tick == horizon - 1))
                failure = metrics['failure'].bool()
                failures |= ending & failure
                truncated = ending & ~failure
                next_critic = critic.clone()
                next_critic[ids] = terminal
                values = model.evaluate_cost(next_critic, alg.epsilon).flatten()
                terminal_tail[truncated] = discount[truncated] * values[truncated]
                tail_bound[truncated] = discount[truncated]
                if observer:
                    observer.finish_control(done, last=tick == horizon - 1)
                active[ending] = False
                if tick % 200 == 0:
                    print('EVAL_PROGRESS', tick, int(active.sum()), flush=True)
                if not active.any():
                    break
        denom = steps.clamp_min(1)
        summary = dict(observed_discounted_cost=cost_sum.mean().item(),
                       pid_bootstrapped_cost=(cost_sum + terminal_tail).mean().item(),
                       timeout_tail_mean=terminal_tail.mean().item(), tail_bound_max=tail_bound.max().item(),
                       tail_bound_mean=tail_bound.mean().item(),
                       cost_per_alive_step=(raw_sum / denom).mean().item(),
                       cost_per_alive_second=(raw_sum / denom / env.dt).mean().item(),
                       survival=(~failures).float().mean().item(), alive_seconds=(steps * env.dt).mean().item(),
                       tracking=(tracking_sum / denom).mean().item(), reward=reward_sum.mean().item(),
                       touchdown_rate=(touchdowns / denom / env.dt).mean().item(),
                       signed_violation=(cost_sum.mean() - args.epsilon).item(),
                       positive_violation_of_mean=max(0., cost_sum.mean().item() - args.epsilon),
                       mean_episode_positive_violation=(cost_sum - args.epsilon).clamp_min(0).mean().item(),
                       initial_cost_prediction=v_initial.mean().item(),
                       initial_cost_bias=(v_initial-cost_sum).mean().item(),
                       initial_cost_rmse=(v_initial-cost_sum).square().mean().sqrt().item(),
                       cost_quantiles=torch.quantile(cost_sum, torch.tensor([0., .1, .5, .9, 1.], device=device)).tolist())
        episodes = {name: value.cpu().tolist() for name, value in dict(
            discounted_cost=cost_sum, timeout_tail=terminal_tail, tail_bound=tail_bound,
            steps=steps, failures=failures, raw_cost=raw_sum, tracking_sum=tracking_sum,
            reward=reward_sum, touchdowns=touchdowns, initial_cost_prediction=v_initial).items()}
        if observer:
            physical, per_env = observer.result()
            summary.update(physical)
            episodes.update(per_env)
            summary['observer_pd_input_checks'] = observer.tick
        result = dict(summary=summary, episodes=episodes, initial=initial, seed=args.seed,
                      mode=args.mode, checkpoint=source_path, gamma=alg.gamma, epsilon=args.epsilon,
                      dt=env.dt, horizon=horizon, stochastic=args.mode != 'deploy')
        (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        print('RESULT', json.dumps(summary), flush=True)
    gym = getattr(env.gym, 'original', env.gym)
    gym.destroy_sim(env.sim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['calibrate', 'train', 'evaluate', 'deploy'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--him-checkpoint', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--expected-initial')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--epsilon', type=float, default=0.)
    parser.add_argument('--arm', choices=['pid', 'zero'], default='pid')
    main(parser.parse_args())
