"""Stage 0 behavioral contracts; run in the existing himloco environment."""

import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import isaacgym  # Must precede torch in this environment.
import torch

from legged_gym.envs.go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
from legged_gym.envs.go1.quietpaw_go1_config import QuietPawGo1Cfg, QuietPawGo1CfgPPO
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.task_registry import task_registry
from rsl_rl.algorithms import HIMPPO, QuietPawHIMPPO
from rsl_rl.modules import HIMActorCritic, QuietPawHIMActorCritic
from rsl_rl.runners import HIMOnPolicyRunner, QuietPawRunner
from rsl_rl.storage import HIMRolloutStorage, QuietPawRolloutStorage


class DummyEnv:
    """Only constructor/reset interfaces; real physics is tested by train.py."""

    num_envs = 8
    num_obs = 270
    num_privileged_obs = 238
    num_one_step_obs = 45
    num_actions = 12

    def reset(self):
        return torch.zeros(8, 270), torch.zeros(8, 238)


class QuietPawStage0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self, cls):
        torch.manual_seed(123)
        with contextlib.redirect_stdout(io.StringIO()):
            return cls(270, 238, 45, 12)

    def assert_same_state(self, first, second):
        self.assertEqual(set(first.state_dict()), set(second.state_dict()))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

    def test_frozen_core_files(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "rsl_rl/rsl_rl/modules/him_estimator.py": "a0cb97016fce9714168f357358dd50d0fb2eaba0017fe37ed5bbc469727024d2",
            "rsl_rl/rsl_rl/modules/him_actor_critic.py": "c516cb3b7bc3b6734182547bacaa6eb6fb0604fac9a0708ff87d869a79c17cc6",
            "rsl_rl/rsl_rl/algorithms/him_ppo.py": "1eac94e9fe99766022c7036fcd28a669962b7dfaecff10c8a7972486002bb0f8",
            "rsl_rl/rsl_rl/storage/him_rollout_storage.py": "971369bb58bdc43e737f317f5ae57443df8f46de463597ee732150c5389bfd9a",
            "legged_gym/legged_gym/envs/base/legged_robot.py": "a4d17f05c52d41c1b60583b0155f3452bb48351864e3b5ea0654f517c684190b",
            "legged_gym/legged_gym/envs/go1/go1_config.py": "08af584e7b42a3652c1752cca9bd6fde53cc96650ece74cc9b86faaf29825a59",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest, name)

    def test_config_preserves_environment_and_optimization(self):
        baseline_env, quiet_env = Go1RoughCfg(), QuietPawGo1Cfg()
        self.assertEqual(class_to_dict(baseline_env), class_to_dict(quiet_env))
        baseline, quiet = Go1RoughCfgPPO(), QuietPawGo1CfgPPO()
        self.assertEqual(class_to_dict(baseline.policy), class_to_dict(quiet.policy))
        self.assertEqual(class_to_dict(baseline.algorithm), class_to_dict(quiet.algorithm))
        expected_runner = class_to_dict(baseline.runner)
        expected_runner.update(policy_class_name="QuietPawHIMActorCritic",
                               algorithm_class_name="QuietPawHIMPPO",
                               experiment_name="quietpaw_go1_stage0")
        self.assertEqual(expected_runner, class_to_dict(quiet.runner))
        quiet_env.rewards.scales.tracking_lin_vel = 7.0
        self.assertEqual(baseline_env.rewards.scales.tracking_lin_vel, 1.0)
        self.assertEqual(Go1RoughCfg().rewards.scales.tracking_lin_vel, 1.0)

    def test_networks_outputs_and_gradients_unchanged(self):
        baseline = self.make_model(HIMActorCritic)
        quiet = self.make_model(QuietPawHIMActorCritic)
        self.assert_same_state(baseline, quiet)
        self.assertEqual(quiet.actor[0].in_features, 64)
        self.assertEqual(quiet.critic[0].in_features, 238)
        self.assertEqual(quiet.actor[-1].out_features, 12)
        self.assertFalse(hasattr(quiet, "cost_critic"))
        history = torch.randn(8, 270)
        self.assertTrue(torch.equal(baseline.act_inference(history), quiet.act_inference(history)))
        quiet.act(history)
        quiet.action_mean.square().sum().backward()
        self.assertTrue(any(p.grad is not None for p in quiet.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in quiet.estimator.parameters()))

    def test_unavailable_placeholders_cannot_enter_training(self):
        storage = QuietPawRolloutStorage(8, 4, [270], [238], [12])
        for name in ("epsilon", "noise_cost", "cost_value"):
            value = getattr(storage, name)
            self.assertEqual(tuple(value.shape), (4, 8, 1))
            self.assertTrue(torch.isnan(value).all())
            transition = QuietPawRolloutStorage.Transition()
            setattr(transition, name, torch.zeros(8, 1))
            with self.assertRaises(NotImplementedError):
                storage.add_transitions(transition)
            self.assertEqual(storage.step, 0)
            transition.clear()
            self.assertIsNone(getattr(transition, name))

    def test_same_rollout_produces_identical_ppo_and_him_updates(self):
        baseline = self.make_model(HIMActorCritic)
        quiet = self.make_model(QuietPawHIMActorCritic)
        initial_encoder = {k: v.clone() for k, v in quiet.estimator.state_dict().items()}
        algorithms = [HIMPPO(baseline, num_learning_epochs=2, num_mini_batches=2),
                      QuietPawHIMPPO(quiet, num_learning_epochs=2, num_mini_batches=2)]
        torch.manual_seed(42)
        histories = torch.randn(5, 8, 270)
        states = torch.randn(5, 8, 238)
        rewards = torch.randn(4, 8)
        dones = torch.zeros(4, 8, dtype=torch.bool)
        dones[1, 0] = True
        dones[2, 1] = True
        results = []
        for alg in algorithms:
            alg.init_storage(8, 4, [270], [238], [12])
            torch.manual_seed(9)
            with torch.no_grad():
                for step in range(4):
                    alg.act(histories[step], states[step])
                    timeout = torch.zeros(8, dtype=torch.bool)
                    if step == 2:
                        timeout[1] = True
                    alg.process_env_step(rewards[step], dones[step],
                                         {"time_outs": timeout}, states[step + 1])
                alg.compute_returns(states[-1])
            results.append((alg.storage.returns.clone(), alg.storage.advantages.clone()))
            torch.manual_seed(10)
            results[-1] += (alg.update(),)
        for first, second in zip(results[0][:2], results[1][:2]):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(results[0][2], results[1][2])
        self.assertTrue(all(torch.isfinite(torch.tensor(results[0][2]))))
        self.assert_same_state(baseline, quiet)
        self.assertTrue(any(not torch.equal(v, initial_encoder[k])
                            for k, v in quiet.estimator.state_dict().items()))
        self.assertIs(QuietPawHIMPPO.update, HIMPPO.update)
        self.assertIs(QuietPawRolloutStorage.compute_returns, HIMRolloutStorage.compute_returns)
        for name in ("epsilon", "noise_cost", "cost_value"):
            self.assertTrue(torch.isnan(getattr(algorithms[1].storage, name)).all())

    def test_runner_dispatch_and_checkpoint_compatibility(self):
        args = SimpleNamespace(seed=None, max_iterations=None, resume=False,
                               experiment_name=None, run_name=None, load_run=None,
                               checkpoint=None, rl_device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            base_runner, _ = task_registry.make_alg_runner(
                DummyEnv(), args=args, train_cfg=Go1RoughCfgPPO(), log_root=None)
            quiet_runner, _ = task_registry.make_alg_runner(
                DummyEnv(), args=args, train_cfg=QuietPawGo1CfgPPO(), log_root=None)
        self.assertIs(type(base_runner), HIMOnPolicyRunner)
        self.assertIs(type(quiet_runner), QuietPawRunner)
        self.assertIs(type(quiet_runner.alg), QuietPawHIMPPO)
        self.assertIs(type(quiet_runner.alg.actor_critic), QuietPawHIMActorCritic)
        self.assertIs(type(quiet_runner.alg.storage), QuietPawRolloutStorage)
        self.assertIs(QuietPawRunner.learn, HIMOnPolicyRunner.learn)
        self.assertIs(task_registry.get_task_class("go1"), task_registry.get_task_class("go1_quietpaw"))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "baseline.pt")
            base_runner.save(path)
            quiet_runner.load(path)
            self.assert_same_state(base_runner.alg.actor_critic, quiet_runner.alg.actor_critic)

    def test_active_quietpaw_settings_fail_explicitly(self):
        for name, value in (("stage", 1), ("epsilon", 0.5), ("noise_cost", "impact")):
            cfg = class_to_dict(QuietPawGo1CfgPPO())
            cfg["quietpaw"][name] = value
            with self.assertRaises(NotImplementedError):
                QuietPawRunner(DummyEnv(), cfg, device="cpu", use_wandb=False)


if __name__ == "__main__":
    unittest.main()
