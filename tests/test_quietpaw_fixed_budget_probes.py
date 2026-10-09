"""Study instrumentation must not become a second learning intervention."""

import importlib.util
from pathlib import Path
import unittest

from test_quietpaw_constraint_validation import ConstraintValidationTests
import torch

spec = importlib.util.spec_from_file_location('fixed_probes', Path(__file__).resolve().parents[1] /
                                            'experiments/quietpaw_fixed_budget_20261009/probes.py')
probes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probes)


class FixedBudgetProbeTests(unittest.TestCase):
    def test_probe_preserves_learning_and_rng_with_nontrivial_clipping(self):
        helpers = ConstraintValidationTests()
        torch.set_num_threads(1)
        alg = helpers.algorithm()
        alg.pid.multiplier.fill_(2.)
        helpers.collect(alg)
        with torch.no_grad():
            alg.actor_critic.actor[-1].bias.add_(.5)
        alg.actor_critic.actor[0].weight.grad = torch.ones_like(alg.actor_critic.actor[0].weight)
        model = alg.actor_critic
        state = {key: value.clone() for key, value in model.state_dict().items()}
        old_grad = model.actor[0].weight.grad.clone()
        rng = torch.get_rng_state().clone()
        result = probes.gradient_probe(model, alg.storage, alg.lambda_used)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(old_grad, model.actor[0].weight.grad))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()))
        self.assertGreater(result['unclipped_cost_norm'], 0.)
        self.assertGreater(result['clipping_active_fraction'], 0.)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
        # Reconstruct the norm of the actual clipped joint PPO gradient from
        # the two reported contributions; independent clipping would fail.
        storage = alg.storage
        obs = storage.observations[:2].flatten(0, 1)
        eps = storage.epsilon[:2].flatten(0, 1)
        actions = storage.actions[:2].flatten(0, 1)
        mean = model.act_inference(obs, eps)
        logp = torch.distributions.Normal(mean, mean * 0 + model.std).log_prob(actions).sum(-1)
        loss = alg.policy_loss(logp, storage.actions_log_prob[:2].flatten(),
                               storage.advantages[:2].flatten(), alg.clip_param)
        grad = torch.autograd.grad(loss, tuple(model.actor.parameters()) + (model.std,))
        actual = sum(g.square().sum() for g in grad).sqrt().item()
        nr, nc = result['clipped_reward_norm'], result['clipped_cost_norm']
        predicted = max(0., nr * nr + nc * nc + 2 * nr * nc * result['clipped_loss_gradient_cosine']) ** .5
        self.assertAlmostEqual(actual, predicted, places=5)

    def test_zero_lambda_retains_unweighted_cost_signal(self):
        helpers = ConstraintValidationTests()
        alg = helpers.algorithm()
        helpers.collect(alg)
        result = probes.gradient_probe(alg.actor_critic, alg.storage, alg.lambda_used)
        self.assertEqual(result['unclipped_cost_norm'], 0.)
        self.assertGreater(result['unweighted_cost_norm'], 0.)
        self.assertFalse(result['clipped_cosine_defined'])


if __name__ == '__main__':
    unittest.main()
