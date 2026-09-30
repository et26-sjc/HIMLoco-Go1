"""Behavioral contract for the HIM-to-CMDP weight-only transition."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from rsl_rl.modules.him_actor_critic import HIMActorCritic
from rsl_rl.modules.quietpaw_conditional_actor_critic import QuietPawConditionalActorCritic
from rsl_rl.utils.quietpaw_him_warmstart import initialize_from_him_checkpoint


class HIMWarmStartTest(unittest.TestCase):
    def test_initial_policy_and_reward_value_match_him_for_every_epsilon(self):
        torch.manual_seed(77)
        him = HIMActorCritic(270, 238, 45, 12)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_7500.pt"
            torch.save({"model_state_dict": him.state_dict(), "iter": 0}, path)
            conditioned = QuietPawConditionalActorCritic(270, 238, 45, 12)
            metadata = initialize_from_him_checkpoint(conditioned, path)

            history = torch.randn(13, 270)
            privileged = torch.randn(13, 238)
            with torch.no_grad():
                expected_action = him.act_inference(history)
                expected_reward_value = him.evaluate(privileged)
                for level in (0., .1, .5, 1.):
                    epsilon = torch.full((13, 1), level)
                    torch.testing.assert_close(conditioned.act_inference(history, epsilon),
                                               expected_action, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(conditioned.evaluate(privileged, epsilon),
                                               expected_reward_value, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(conditioned.evaluate_cost(privileged, epsilon),
                                               torch.zeros(13, 1), rtol=0, atol=0)

            for name, value in him.estimator.state_dict().items():
                self.assertTrue(torch.equal(value, conditioned.estimator.state_dict()[name]), name)
            self.assertTrue(torch.equal(him.std, conditioned.std))
            self.assertTrue(torch.equal(conditioned.actor[0].weight[:, -1],
                                        torch.zeros_like(conditioned.actor[0].weight[:, -1])))
            self.assertEqual(metadata["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(metadata["source_checkpoint_iter_field"], 0)
            self.assertEqual(metadata["optimizer_state"], "fresh_PPO_HIM_PID")

    def test_rejects_non_him_source_before_loading(self):
        torch.manual_seed(78)
        conditioned = QuietPawConditionalActorCritic(270, 238, 45, 12)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "conditioned.pt"
            torch.save({"model_state_dict": conditioned.state_dict(), "quietpaw": {}}, path)
            with self.assertRaises(ValueError):
                initialize_from_him_checkpoint(conditioned, path)


if __name__ == "__main__":
    unittest.main()
