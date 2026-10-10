"""Validation controls and instrumentation must preserve the Stage 1 learner."""

import contextlib
import io
from types import SimpleNamespace
import unittest

import isaacgym
import torch

from legged_gym.envs.go1.quietpaw_go1_config import QuietPawStage1Go1Cfg, QuietPawStage1Go1CfgPPO
from legged_gym.scripts.train_quietpaw import configure_experiment
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.algorithms.quietpaw_constrained_ppo import QuietPawConstrainedPPO
from rsl_rl.modules.quietpaw_conditional_actor_critic import QuietPawConditionalActorCritic
from rsl_rl.utils.quietpaw_diagnostics import actor_gradient_diagnostics


class ConstraintValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def algorithm(self, **kwargs):
        torch.manual_seed(91)
        with contextlib.redirect_stdout(io.StringIO()):
            model = QuietPawConditionalActorCritic(270, 238, 45, 12)
        alg = QuietPawConstrainedPPO(model, num_learning_epochs=2, num_mini_batches=2,
                                    gamma=.5, **kwargs)
        alg.init_storage(4, 4, [270], [238], [12])
        return alg

    def collect(self, alg):
        torch.manual_seed(92)
        history = torch.randn(5, 4, 270)
        state = torch.randn(5, 4, 238)
        with torch.no_grad():
            for t in range(4):
                alg.act(history[t], state[t])
                done = torch.full((4,), t == 2, dtype=torch.bool)
                alg.process_env_step(torch.randn(4), done,
                                     {'quietpaw': {'noise_cost': torch.full((4,), .6)}}, state[t+1])
            alg.compute_returns(state[-1])

    def test_omitted_options_preserve_configuration(self):
        env, cfg = QuietPawStage1Go1Cfg(), QuietPawStage1Go1CfgPPO()
        before_env, before_cfg = class_to_dict(env), class_to_dict(cfg)
        configure_experiment(env, cfg, SimpleNamespace())
        self.assertEqual(before_env, class_to_dict(env))
        self.assertEqual(before_cfg, class_to_dict(cfg))

    def test_gain_scale_and_fixed_budget_are_explicit(self):
        env, cfg = QuietPawStage1Go1Cfg(), QuietPawStage1Go1CfgPPO()
        original_env = class_to_dict(env)
        configure_experiment(env, cfg, SimpleNamespace(pid_gain_scale=100., epsilon_fixed=.085,
                                                        constraint_diagnostics_interval=50))
        self.assertEqual((cfg.algorithm.pid.kp, cfg.algorithm.pid.ki, cfg.algorithm.pid.kd),
                         (100., 10., 10.))
        self.assertEqual((cfg.algorithm.epsilon_min, cfg.algorithm.epsilon_max), (.085, .085))
        self.assertEqual(cfg.algorithm.diagnostics_interval, 50)
        self.assertEqual(original_env, class_to_dict(env))
        for args in (SimpleNamespace(pid_gain_scale=-1), SimpleNamespace(epsilon_fixed=float('nan')),
                     SimpleNamespace(epsilon_fixed=.1, epsilon_min=.08)):
            with self.assertRaises(ValueError):
                configure_experiment(QuietPawStage1Go1Cfg(), QuietPawStage1Go1CfgPPO(), args)

    def test_fixed_assignment_and_zero_multiplier_control(self):
        alg = self.algorithm(epsilon_fixed=.085, pid=dict(kp=0., ki=0., kd=0., initial=0.))
        alg.configure_assignment(torch.arange(4), torch.zeros(4))
        torch.testing.assert_close(alg.epsilon, torch.full((4,1), .085))
        self.assertEqual(alg.assignment['mode'], 'fixed_budget_v1')
        self.assertEqual(alg.assignment['bins'], 1)
        self.assertEqual(alg.assignment['epsilon_min'], .085)
        self.assertEqual(alg.epsilon_labels.unique().tolist(), [0])
        self.collect(alg)
        alg.update()
        self.assertEqual(alg.pid.multiplier.sum().item(), 0.)

    def test_actor_probe_does_not_change_rng_parameters_or_gradients(self):
        alg = self.algorithm()
        alg.pid.multiplier.fill_(.75)
        self.collect(alg)
        model=alg.actor_critic
        model.actor[0].weight.grad = torch.ones_like(model.actor[0].weight)
        state = {k:v.clone() for k,v in model.state_dict().items()}
        gradients = {k:None if p.grad is None else p.grad.clone() for k,p in model.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        results = actor_gradient_diagnostics(model, alg.storage, alg.lambda_used, alg.epsilon_labels)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v,state[k]),k)
        for k,p in model.named_parameters():
            if gradients[k] is None:
                self.assertIsNone(p.grad,k)
            else:
                self.assertTrue(torch.equal(p.grad,gradients[k]),k)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in results.values()))
        self.assertGreater(results['epsilon_bin_0/actor_reward_gradient_norm'],0.)
        self.assertGreater(results['epsilon_bin_0/actor_cost_gradient_norm'],0.)

    def test_instrumented_update_is_identical_to_uninstrumented_update(self):
        first = self.algorithm()
        second = self.algorithm(diagnostics_interval=1)
        for alg in (first,second):
            alg.pid.multiplier.fill_(.75)
            self.collect(alg)
        losses=[]
        rng=[]
        for alg in (first,second):
            torch.manual_seed(93)
            losses.append(alg.update())
            rng.append(torch.random.get_rng_state().clone())
        self.assertEqual(losses[0],losses[1])
        self.assertTrue(torch.equal(rng[0],rng[1]))
        for name,value in first.actor_critic.state_dict().items():
            self.assertTrue(torch.equal(value,second.actor_critic.state_dict()[name]),name)
        for name,value in first.pid.state_dict().items():
            self.assertTrue(torch.equal(value,second.pid.state_dict()[name]),name)
        self.assertIn('epsilon_bin_0/ppo_clip_fraction',second.diagnostics)
        self.assertIn('epsilon_bin_0/weighted_cost_reward_advantage_ratio',second.diagnostics)
        self.assertIn('epsilon_bin_0/actor_cost_reward_gradient_ratio',second.diagnostics)


if __name__=='__main__':
    unittest.main()
