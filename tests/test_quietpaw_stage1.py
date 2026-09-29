"""Research-critical contracts for the Stage 1 constrained learner."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import isaacgym  # Isaac Gym must be imported before torch.
import torch

from legged_gym.envs.go1.go1_config import Go1RoughCfg
from legged_gym.envs.go1.quietpaw_go1_config import QuietPawStage1Go1Cfg, QuietPawStage1Go1CfgPPO
from legged_gym.envs.go1.quietpaw_noise import noise_cost
from legged_gym.envs.go1.quietpaw_legged_robot import QuietPawLeggedRobot
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules import HIMActorCritic
from rsl_rl.modules.quietpaw_conditional_actor_critic import QuietPawConditionalActorCritic
from rsl_rl.storage.quietpaw_cmdp_storage import QuietPawCMDPStorage
from rsl_rl.algorithms.quietpaw_constrained_ppo import QuietPawConstrainedPPO
from rsl_rl.algorithms.quietpaw_pid import QuietPawPID
from rsl_rl.runners.quietpaw_stage1_runner import QuietPawStage1Runner
from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner


class DummyEnv:
    num_envs, num_obs, num_privileged_obs, num_one_step_obs, num_actions = 10, 270, 238, 45, 12
    cfg = QuietPawStage1Go1Cfg()

    def reset(self):
        return torch.zeros(self.num_envs, 270), torch.zeros(self.num_envs, 238)


class QuietPawStage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, cls=QuietPawConditionalActorCritic):
        torch.manual_seed(17)
        with contextlib.redirect_stdout(io.StringIO()):
            return cls(270, 238, 45, 12)

    def algorithm(self, num_envs=4, steps=4):
        alg = QuietPawConstrainedPPO(self.model(), num_learning_epochs=2,
                                    num_mini_batches=2, gamma=0.5, lam=0.95)
        alg.init_storage(num_envs, steps, [270], [238], [12])
        return alg

    def collect(self, alg):
        torch.manual_seed(3)
        n = alg.storage.num_envs
        histories = torch.randn(alg.storage.num_transitions_per_env + 1, n, 270)
        states = torch.randn(alg.storage.num_transitions_per_env + 1, n, 238)
        with torch.no_grad():
            for step in range(alg.storage.num_transitions_per_env):
                alg.act(histories[step], states[step])
                done = torch.ones(n, dtype=torch.bool) if step == 1 else torch.zeros(n, dtype=torch.bool)
                alg.process_env_step(torch.randn(n), done,
                                     {"quietpaw": {"noise_cost": torch.full((n,), 0.5)}}, states[step + 1])
            alg.compute_returns(states[-1])

    def test_environment_config_preserves_backbone(self):
        quiet = class_to_dict(QuietPawStage1Go1Cfg())
        quiet.pop("noise_cost")
        self.assertEqual(quiet, class_to_dict(Go1RoughCfg()))
        self.assertIs(QuietPawLeggedRobot.step, LeggedRobot.step)
        self.assertIs(QuietPawStage1Runner.learn, HIMOnPolicyRunner.learn)

    def test_conditional_dimensions_and_him_gradient_boundary(self):
        model, baseline = self.model(), self.model(HIMActorCritic)
        self.assertEqual(model.actor[0].in_features, 65)
        self.assertEqual(model.critic[0].in_features, 239)
        self.assertEqual(model.cost_critic[0].in_features, 239)
        for key, value in baseline.estimator.state_dict().items():
            self.assertTrue(torch.equal(value, model.estimator.state_dict()[key]), key)
        history, critic, eps = torch.randn(4, 270), torch.randn(4, 238), torch.ones(4, 1)
        self.assertTrue(torch.equal(model.act_inference(history, eps), baseline.act_inference(history)))
        with self.assertRaises(ValueError):
            model.act_inference(history)
        with torch.no_grad():
            model.actor[0].weight[:, -1].fill_(1.)
        self.assertFalse(torch.equal(model.act_inference(history, eps), model.act_inference(history, eps * 0)))
        loss = model.act_inference(history, eps).square().mean() + model.evaluate(critic, eps).square().mean()
        loss += (model.evaluate_cost(critic, eps) - 1).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.estimator.parameters()))
        self.assertIsNotNone(model.actor[0].weight.grad)
        self.assertIsNotNone(model.cost_critic[-1].weight.grad)

    def test_cost_monotonic_bounded_contact_mask_and_touchdown(self):
        force = torch.zeros(4, 4, 3)
        force[..., 2] = torch.tensor([2., 10., 100., 1000.])[:, None]
        velocity = torch.zeros_like(force)
        velocity[..., 2] = -torch.tensor([0., 0.1, 1., 10.])[:, None]
        previous = torch.zeros(4, 4, dtype=torch.bool)
        valid = torch.ones(4, dtype=torch.bool)
        cost, contact, touchdown = noise_cost(force, velocity, previous, valid)
        self.assertTrue((torch.diff(cost) > 0).all())
        self.assertTrue(((cost >= 0) & (cost <= 1)).all())
        self.assertTrue(touchdown.all())
        no_contact = noise_cost(force * 0, velocity, previous, valid)[0]
        self.assertTrue(torch.equal(no_contact, torch.zeros(4)))
        sustained = noise_cost(force, velocity, contact, valid)[0]
        reset = noise_cost(force, velocity, previous, ~valid)[0]
        self.assertTrue(torch.equal(sustained, reset))
        self.assertTrue((sustained <= cost).all())
        # Negative world-z is downward; upward motion is not an impact speed.
        upward = noise_cost(force, -velocity, previous, valid)[0]
        self.assertTrue(torch.equal(upward, sustained))

    def test_measurement_precedes_reward_and_reset_preserves_returned_cost(self):
        env = object.__new__(QuietPawLeggedRobot)
        env.cfg, env.dt = QuietPawStage1Go1Cfg(), 0.02
        env.num_envs, env.feet_indices = 2, torch.arange(4)
        env.termination_contact_indices = torch.tensor([0])
        env.contact_forces = torch.ones(2, 4, 3) * 20
        env.feet_vel = torch.ones(2, 4, 3)
        env.root_states = torch.zeros(2, 13)
        env.base_lin_vel = torch.zeros(2, 3)
        env.commands, env.projected_gravity = torch.zeros(2, 4), torch.zeros(2, 3)
        env.reset_buf = torch.tensor([1, 0], dtype=torch.bool)
        env.time_out_buf = torch.zeros(2, dtype=torch.bool)
        env.quietpaw_contact = torch.zeros(2, 4, dtype=torch.bool)
        env.quietpaw_velocity, env.quietpaw_force = torch.ones(2, 4, 3) * -1, torch.zeros(2, 4, 3)
        env.quietpaw_base_velocity = torch.zeros(2, 3)
        env.quietpaw_valid = torch.ones(2, dtype=torch.bool)
        env.extras, env.rew_buf = {}, torch.tensor([3., 4.])
        env.extras["time_outs"] = torch.ones(2, dtype=torch.bool)  # stale previous step
        with patch.object(LeggedRobot, "compute_reward") as reward:
            env.compute_reward()
        reward.assert_called_once_with()
        self.assertTrue(torch.equal(env.rew_buf, torch.tensor([3., 4.])))
        self.assertFalse(env.extras["time_outs"].any())
        cost = env.extras["quietpaw"]["noise_cost"].clone()
        with patch.object(LeggedRobot, "reset_idx"):
            env.reset_idx(torch.tensor([0]))
        self.assertTrue(torch.equal(cost, env.extras["quietpaw"]["noise_cost"]))
        self.assertFalse(env.quietpaw_valid[0])
        self.assertTrue(env.quietpaw_valid[1])

    def test_cost_gae_matches_hand_calculation_and_lambda_axis(self):
        storage = QuietPawCMDPStorage(2, 3, [270], [238], [12])
        storage.step = 3
        storage.cost_rewards.copy_(torch.tensor([[.1, .2], [.3, .4], [.5, .6]]).unsqueeze(-1))
        storage.cost_value.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]).unsqueeze(-1))
        storage.dones.copy_(torch.tensor([[0, 1], [1, 0], [0, 0]]).unsqueeze(-1))
        storage.returns.fill_(2.)
        multipliers = torch.tensor([[0.], [3.]])
        storage.compute_cost_returns(torch.tensor([[7.], [8.]]), .5, 1., multipliers)
        expected = torch.tensor([[.25, .2], [.3, 2.7], [4., 4.6]]).unsqueeze(-1)
        torch.testing.assert_close(storage.cost_returns, expected)
        raw = expected - storage.cost_value
        torch.testing.assert_close(storage.cost_advantages, raw)
        torch.testing.assert_close(storage.combined_advantages, (2 - multipliers * raw) / (1 + multipliers))

    def test_timeout_uses_terminal_cost_value_and_failure_does_not(self):
        alg = self.algorithm(num_envs=2)
        next_state = torch.zeros(2, 238)
        next_state[:, 0] = torch.tensor([.8, .9])
        with torch.no_grad():
            alg.act(torch.zeros(2, 270), torch.zeros(2, 238))
            original_values = alg.transition.values.clone()
            transition_epsilon = alg.transition.epsilon.clone()
            with patch.object(alg.actor_critic, "evaluate_cost", side_effect=lambda obs, eps: obs[:, :1]) as cost_value, \
                 patch.object(alg.actor_critic, "evaluate", side_effect=lambda obs, eps: 10 * obs[:, :1] + eps) as reward_value:
                alg.process_env_step(torch.ones(2), torch.ones(2, dtype=torch.bool),
                                     {"quietpaw": {"noise_cost": torch.tensor([.2, .4])},
                                      "time_outs": torch.tensor([True, False])}, next_state)
        torch.testing.assert_close(alg.storage.cost_rewards[0], torch.tensor([[.5], [.2]]))
        torch.testing.assert_close(alg.completed_cost, torch.tensor([[.5], [.2]]))
        torch.testing.assert_close(alg.storage.next_privileged_observations[0], next_state)
        expected_reward = torch.ones(2, 1) + .5 * (10 * next_state[:, :1] + transition_epsilon) * torch.tensor([[1.], [0.]])
        torch.testing.assert_close(alg.storage.rewards[0], expected_reward)
        torch.testing.assert_close(alg.storage.values[0], original_values)
        torch.testing.assert_close(reward_value.call_args.args[0], next_state)
        torch.testing.assert_close(reward_value.call_args.args[1], transition_epsilon)
        torch.testing.assert_close(cost_value.call_args.args[0], next_state)
        torch.testing.assert_close(cost_value.call_args.args[1], transition_epsilon)
        self.assertTrue(torch.equal(alg.episode_cost, torch.zeros(2, 1)))

    def test_pid_direction_independence_missing_data_and_checkpoint(self):
        pid = QuietPawPID(torch.tensor([[.2], [.8], [.1]]), kp=0., ki=1., kd=0., ema=0., initial=.5)
        pid.update(torch.tensor([[.7], [.3], [1.]]), torch.tensor([[True], [True], [False]]))
        torch.testing.assert_close(pid.multiplier, torch.tensor([[1.], [0.], [.5]]))
        self.assertTrue(torch.equal(pid.updates, torch.tensor([[1], [1], [0]])))
        other = QuietPawPID(pid.epsilon)
        other.load_state_dict(pid.state_dict())
        self.assertTrue(torch.equal(other.multiplier, pid.multiplier))
        with self.assertRaises(ValueError):
            QuietPawPID(torch.zeros(2, 1)).load_state_dict(pid.state_dict())

    def test_physical_failure_does_not_bootstrap_even_at_time_limit(self):
        alg = self.algorithm(num_envs=2)
        with torch.no_grad():
            alg.act(torch.zeros(2, 270), torch.zeros(2, 238))
            with patch.object(alg.actor_critic, "evaluate_cost", return_value=torch.full((2, 1), 99.)), \
                 patch.object(alg.actor_critic, "evaluate", return_value=torch.full((2, 1), 99.)):
                alg.process_env_step(torch.ones(2), torch.ones(2, dtype=torch.bool),
                                     {"quietpaw": {"noise_cost": torch.full((2,), .4),
                                                   "failure": torch.ones(2, dtype=torch.bool)},
                                      "time_outs": torch.ones(2, dtype=torch.bool)}, torch.zeros(2, 238))
        torch.testing.assert_close(alg.storage.rewards[0], torch.ones(2, 1))
        torch.testing.assert_close(alg.storage.cost_rewards[0], torch.full((2, 1), .2))
        self.assertEqual(alg.success_count.sum().item(), 0.)

    def test_shuffled_batches_keep_conditions_and_targets_aligned(self):
        alg = self.algorithm()
        self.collect(alg)
        storage = alg.storage
        ids = torch.arange(16).reshape(4, 4)
        storage.observations[:, :, 0] = ids
        storage.cost_returns[:, :, 0] = ids + 10
        seen = []
        for batch in storage.cmdp_mini_batch_generator(2, 1):
            index = batch["observations"][:, 0].long()
            torch.testing.assert_close(batch["epsilon"], alg.epsilon[index % 4])
            torch.testing.assert_close(batch["cost_returns"][:, 0], index.float() + 10)
            seen.extend(index.tolist())
        self.assertEqual(sorted(seen), list(range(16)))

    def test_cost_policy_gradient_direction_and_ppo_clipping(self):
        # Equal reward advantage, lambda=1; high cost should lose probability.
        log_prob = torch.zeros(2, requires_grad=True)
        advantage = -torch.tensor([1., -1.]) / 2
        loss = QuietPawConstrainedPPO.policy_loss(log_prob, torch.zeros(2), advantage, .2)
        loss.backward()
        torch.testing.assert_close(log_prob.grad, torch.tensor([.25, -.25]))
        # Descent lowers the first log probability and raises the second.
        clipped = torch.tensor([1.5, .5]).log().requires_grad_()
        loss = QuietPawConstrainedPPO.policy_loss(clipped, torch.zeros(2), torch.tensor([1., -1.]), .2)
        torch.testing.assert_close(loss, torch.tensor(-.2))
        loss.backward()
        torch.testing.assert_close(clipped.grad, torch.zeros(2))

    def test_update_keeps_auxiliary_frequency_gradients_and_dual_critics(self):
        alg = self.algorithm()
        self.collect(alg)
        ac = alg.actor_critic
        reference_estimator = copy.deepcopy(ac.estimator)
        before_him = {k: v.clone() for k, v in ac.estimator.state_dict().items()}
        before_cost = ac.cost_critic[-1].weight.clone()
        with patch.object(ac.estimator, "update", wraps=ac.estimator.update) as update:
            losses = alg.update()
        self.assertEqual(update.call_count, 4)
        for call in update.call_args_list:
            self.assertEqual(call.args[0].shape[1], 270)
            self.assertEqual(call.args[1].shape[1], 238)
            self.assertEqual(call.kwargs["lr"], alg.learning_rate)
            reference_estimator.update(*call.args, **call.kwargs)
        # Replay exactly those minibatches through an independent, original
        # HIMEstimator. PPO must not have made any additional estimator update.
        for key, value in ac.estimator.state_dict().items():
            self.assertTrue(torch.equal(value, reference_estimator.state_dict()[key]), key)
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertTrue(all(p.grad is None for p in ac.estimator.parameters()))
        for prefix in ("encoder", "target", "proto"):
            self.assertTrue(any(not torch.equal(v, before_him[k]) for k, v in ac.estimator.state_dict().items()
                                if k.startswith(prefix)), prefix)
        self.assertFalse(torch.equal(before_cost, ac.cost_critic[-1].weight))
        self.assertEqual(alg.diagnostics["him_updates"], 4)
        self.assertTrue(torch.equal(alg.pid.updates, torch.ones(4, 1, dtype=torch.long)))
        self.assertEqual(alg.pid.multiplier[alg.epsilon.argmax()].item(), 0.)
        self.assertGreater(alg.pid.multiplier[alg.epsilon.argmin()].item(), 0.)
        self.assertEqual(alg.lambda_used.sum().item(), 0.)
        self.assertAlmostEqual(alg.diagnostics["epsilon_bin_0/noise_cost_mean"], .5)
        self.assertIn("epsilon_bin_0/reward_mean", alg.diagnostics)
        self.assertIn("epsilon_bin_0/completed_episodes", alg.diagnostics)

    def test_resume_restores_pid_optimizers_iteration_and_rejects_contract_change(self):
        cfg = class_to_dict(QuietPawStage1Go1CfgPPO())
        cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=2, epsilon_bins=2)
        cfg["runner"]["num_steps_per_env"] = 4
        with contextlib.redirect_stdout(io.StringIO()):
            runner = QuietPawStage1Runner(DummyEnv(), cfg, device="cpu", use_wandb=False)
            other = QuietPawStage1Runner(DummyEnv(), copy.deepcopy(cfg), device="cpu", use_wandb=False)
        self.collect(runner.alg)
        runner.alg.update()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            runner.save(path)
            other.load(path)
            self.assertEqual(other.current_learning_iteration, 1)
            self.assertEqual(other.alg.iterations, 1)
            for key, value in runner.alg.pid.state_dict().items():
                self.assertTrue(torch.equal(value, other.alg.pid.state_dict()[key]))
            self.assertTrue(other.alg.optimizer.state_dict()["state"])
            self.assertTrue(other.alg.actor_critic.estimator.optimizer.state_dict()["state"])
            self.assertEqual(other.alg.episode_cost.sum().item(), 0.)
            original_mapping = copy.deepcopy(other.alg.assignment)
            other.alg.assignment["terrain"][0] = 999
            with self.assertRaisesRegex(ValueError, "mapping"):
                other.load(path)
            other.alg.assignment = original_mapping
            other.contract["budget"] = "incorrect"
            with self.assertRaises(ValueError):
                other.load(path)


if __name__ == "__main__":
    unittest.main()
