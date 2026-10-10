"""CPU gates for policy adaptation, Gaussian consistency, resume and export.

Run without Isaac Gym: PYTHONPATH=rsl_rl python -m unittest discover -s rsl_rl/tests
"""

import contextlib
import io
import os
import tempfile
import unittest

import torch
from torch.distributions import Normal

from rsl_rl.modules import HIMActorCritic, QuietHIMActorCritic
from rsl_rl.modules.quiet_actor_critic import LoRALinear, DeltaLinear
from rsl_rl.algorithms import HIMPPO
from rsl_rl.utils.quiet_checkpoint import (
    file_sha256, load_base_checkpoint, save_quiet_checkpoint,
    load_quiet_checkpoint, load_quiet_policy,
)


class QuietPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(13)
        self.arguments = dict(num_actor_obs=270, num_critic_obs=238,
                              num_one_step_obs=45, num_actions=12,
                              init_noise_std=0.6)
        with contextlib.redirect_stdout(io.StringIO()):
            self.baseline = HIMActorCritic(**self.arguments)
        self.history = torch.randn(16, 270)
        self.privileged = torch.randn(16, 238)

    def model(self, mode='lora'):
        with contextlib.redirect_stdout(io.StringIO()):
            result = QuietHIMActorCritic(adaptation=mode, **self.arguments)
        result.load_base_model(self.baseline.state_dict())
        return result

    def optimizer(self, model):
        return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=3e-4)

    def update(self, model, optimizer):
        model.update_distribution(self.history)
        # A valid Gaussian sample is kept unchanged through log_prob evaluation.
        action = model.distribution.sample()
        advantage = torch.linspace(-1.0, 1.0, self.history.size(0))
        loss = (-(model.get_actions_log_prob(action) * advantage).mean()
                + model.evaluate(self.privileged).square().mean()
                - 0.01 * model.entropy.mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def test_zero_initialization_and_gaussian_equivalence(self):
        model = self.model()
        self.baseline.update_distribution(self.history)
        model.update_distribution(self.history)
        torch.testing.assert_close(model.action_mean, self.baseline.action_mean, rtol=0, atol=0)
        torch.testing.assert_close(model.action_std, self.baseline.action_std, rtol=0, atol=0)
        actions = model.distribution.sample()
        old_log_probability = self.baseline.get_actions_log_prob(actions)
        torch.testing.assert_close(model.get_actions_log_prob(actions), old_log_probability,
                                   rtol=0, atol=0)
        ratio = torch.exp(model.get_actions_log_prob(actions) - old_log_probability)
        torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0)
        manual = Normal(model.action_mean, model.action_std)
        torch.testing.assert_close(model.get_actions_log_prob(actions),
                                   manual.log_prob(actions).sum(-1), rtol=0, atol=0)
        torch.testing.assert_close(model.entropy, manual.entropy().sum(-1), rtol=0, atol=0)

    def test_modes_and_parameter_counts(self):
        for mode, expected in (('frozen', 0), ('head', 1548), ('lora', 21516), ('full', 199052)):
            model = self.model(mode)
            self.assertEqual(sum(p.numel() for p in model.adapter_parameters()), expected)
            self.assertTrue(all(not p.requires_grad for p in model.estimator.parameters()))
            self.assertTrue(all(not p.requires_grad for p in model.base_actor.parameters()))
            self.assertTrue(model.std.requires_grad)
            self.assertTrue(all(p.requires_grad for p in model.critic.parameters()))
            torch.testing.assert_close(model.act_inference(self.history),
                                       self.baseline.act_inference(self.history), rtol=0, atol=0)
        model = self.model()
        self.assertIsInstance(model.actor[0], LoRALinear)
        self.assertIsInstance(model.actor[2], LoRALinear)
        self.assertIsInstance(model.actor[4], LoRALinear)
        self.assertIsInstance(model.actor[6], DeltaLinear)

    def test_frozen_weights_and_estimator_have_no_gradient(self):
        model = self.model()
        before = {name: p.detach().clone() for name, p in model.named_parameters()
                  if not p.requires_grad}
        optimizer = self.optimizer(model)
        self.assertTrue(torch.isfinite(self.update(model, optimizer)).item())
        # At zero B, the first derivative to A is zero. B starts learning first.
        for layer in (model.actor[0], model.actor[2], model.actor[4]):
            self.assertEqual(layer.lora_A.grad.abs().sum().item(), 0)
            self.assertGreater(layer.lora_B.grad.abs().sum().item(), 0)
        self.update(model, optimizer)
        for layer in (model.actor[0], model.actor[2], model.actor[4]):
            self.assertGreater(layer.lora_A.grad.abs().sum().item(), 0)
        self.assertGreater(model.actor[6].delta_weight.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            if name in before:
                self.assertIsNone(parameter.grad, name)
                self.assertTrue(torch.equal(parameter.detach(), before[name]), name)

    def test_disabled_restores_original_means_and_std_in_all_modes(self):
        for mode in ('head', 'lora', 'full'):
            model = self.model(mode)
            self.update(model, self.optimizer(model))
            with torch.no_grad():
                model.std.add_(0.15)
            model.set_adapter_enabled(False)
            model.update_distribution(self.history)
            self.baseline.update_distribution(self.history)
            torch.testing.assert_close(model.action_mean, self.baseline.action_mean,
                                       rtol=0, atol=0)
            torch.testing.assert_close(model.action_std, self.baseline.action_std,
                                       rtol=0, atol=0)

    def test_audited_load_rejects_missing_unknown_and_shape_without_mutation(self):
        for problem in ('missing', 'unknown', 'shape'):
            model = self.model()
            before = {name: value.clone() for name, value in model.state_dict().items()}
            state = dict(self.baseline.state_dict())
            if problem == 'missing':
                del state['actor.0.bias']
            elif problem == 'unknown':
                state['unexpected'] = torch.zeros(1)
            else:
                state['actor.0.weight'] = torch.zeros(1)
            with self.assertRaises(RuntimeError):
                model.load_base_model(state)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)

    def test_baseline_file_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline_path = os.path.join(temporary, 'base.pt')
            resume_path = os.path.join(temporary, 'quiet.pt')
            torch.save({'model_state_dict': self.baseline.state_dict(), 'iter': 1000}, baseline_path)
            original_hash = file_sha256(baseline_path)
            model = self.model()
            metadata = load_base_checkpoint(model, baseline_path)
            self.assertEqual(metadata['base_checkpoint_sha256'], original_hash)
            self.assertEqual(metadata['base_iteration'], 1000)
            metadata['calibration'] = {'touch_cost_rate': 0.2, 'tracking_reward_rate': 1.0}
            optimizer = self.optimizer(model)
            self.update(model, optimizer)
            save_quiet_checkpoint(model, resume_path, optimizer, iteration=7, metadata=metadata)
            replacement = self.model()
            replacement_optimizer = self.optimizer(replacement)
            loaded = load_quiet_checkpoint(replacement, resume_path, replacement_optimizer)
            self.assertEqual(loaded['iter'], 7)
            self.assertEqual(loaded['quiet_metadata']['calibration'], metadata['calibration'])
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, replacement.state_dict()[name]), name)
            self.assertEqual(len(optimizer.state), len(replacement_optimizer.state))
            for state in replacement_optimizer.state.values():
                self.assertEqual(state['step'], 1)
            self.assertEqual(file_sha256(baseline_path), original_hash)
            with contextlib.redirect_stdout(io.StringIO()):
                restored, _ = load_quiet_policy(resume_path, adapter_enabled=False)
            torch.testing.assert_close(restored.act_inference(self.history),
                                       self.baseline.act_inference(self.history), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load_quiet_checkpoint(self.model('head'), resume_path)

    def test_jit_merging_enabled_and_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self.model()
            self.update(model, self.optimizer(model))
            for enabled in (True, False):
                path = os.path.join(temporary, 'policy_{}.pt'.format(enabled))
                model.set_adapter_enabled(enabled)
                expected = model.act_inference(self.history).detach()
                model.export_policy(path, adapter_enabled=enabled)
                actual = torch.jit.load(path)(self.history)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(tuple(actual.shape), (16, 12))
                self.assertTrue(all(not isinstance(layer, (LoRALinear, DeltaLinear))
                                    for layer in model.merged_actor(enabled)))
                self.assertEqual(model.adapter_enabled, enabled)

    def test_real_ppo_storage_and_update_keep_raw_action_distribution(self):
        model = self.model()

        def forbidden_estimator_update(*args, **kwargs):
            raise AssertionError("Frozen HIM estimator optimizer must not run")

        model.estimator.update = forbidden_estimator_update
        # Exercise environmental clipping with actual Gaussian tail samples.
        # This diagnostic std is intentionally large; it is not a train default.
        with torch.no_grad():
            model.std.fill_(120.0)
        algorithm = HIMPPO(model, num_learning_epochs=2, num_mini_batches=2,
                           gamma=0.99, lam=0.95, learning_rate=3e-4,
                           entropy_coef=0.01, update_estimator=False)
        optimized = {id(parameter) for group in algorithm.optimizer.param_groups
                     for parameter in group['params']}
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        self.assertEqual(optimized, expected)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters()
                  if not p.requires_grad}
        critic_before = {name: p.detach().clone() for name, p in model.critic.named_parameters()}
        std_before = model.std.detach().clone()
        environments, steps = 8, 6
        algorithm.init_storage(environments, steps, [270], [238], [12])
        saw_clipped_sample = False
        with torch.inference_mode():
            observations = torch.randn(environments, 270)
            privileged = torch.randn(environments, 238)
            for step in range(steps):
                raw = algorithm.act(observations, privileged)
                stored_log_probability = model.get_actions_log_prob(raw).clone()
                executed = raw.clamp(-100.0, 100.0)
                saw_clipped_sample = saw_clipped_sample or not torch.equal(raw, executed)
                rewards = 1.0 - executed.square().mean(-1) / 10000.0
                dones = torch.zeros(environments, dtype=torch.bool)
                timeouts = torch.zeros(environments, dtype=torch.bool)
                if step == 3:
                    dones[0] = True
                    timeouts[0] = True
                next_observations = torch.randn(environments, 270)
                next_privileged = torch.randn(environments, 238)
                algorithm.process_env_step(rewards, dones, {'time_outs': timeouts},
                                           next_privileged)
                torch.testing.assert_close(algorithm.storage.actions[step], raw,
                                           rtol=0, atol=0)
                torch.testing.assert_close(algorithm.storage.actions_log_prob[step, :, 0],
                                           stored_log_probability, rtol=0, atol=0)
                observations, privileged = next_observations, next_privileged
            algorithm.compute_returns(privileged)
        self.assertTrue(saw_clipped_sample)
        self.assertTrue(torch.isfinite(algorithm.storage.returns).all().item())
        self.assertTrue(torch.isfinite(algorithm.storage.advantages).all().item())
        with torch.no_grad():
            flat_observations = algorithm.storage.observations.flatten(0, 1)
            flat_actions = algorithm.storage.actions.flatten(0, 1)
            model.update_distribution(flat_observations)
            old_log_probability = algorithm.storage.actions_log_prob.flatten(0, 1)[:, 0]
            actual = model.get_actions_log_prob(flat_actions)
            torch.testing.assert_close(actual, old_log_probability, rtol=1e-6, atol=1e-5)
            ratio = (actual - old_log_probability).exp()
            torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-5, atol=1e-5)
            expected = Normal(model.action_mean, model.action_std)
            torch.testing.assert_close(actual, expected.log_prob(flat_actions).sum(-1),
                                       rtol=0, atol=0)
        losses = algorithm.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(loss)).item() for loss in losses))
        self.assertEqual(losses[2:], (0.0, 0.0))
        self.assertEqual(algorithm.storage.step, 0)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all().item(), name)
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                self.assertTrue(torch.equal(parameter, frozen[name]), name)
        for layer in (model.actor[0], model.actor[2], model.actor[4]):
            self.assertGreater(layer.lora_B.grad.abs().sum().item(), 0)
            self.assertGreater(layer.lora_A.grad.abs().sum().item(), 0)
        self.assertTrue(any(not torch.equal(p, critic_before[name])
                            for name, p in model.critic.named_parameters()))
        self.assertFalse(torch.equal(model.std, std_before))


if __name__ == '__main__':
    unittest.main()
