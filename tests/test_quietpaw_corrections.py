"""Independent hand checks for review corrections C1--C3."""

import unittest
import isaacgym
import torch
from rsl_rl.algorithms.quietpaw_assignment import balanced_assignment
from rsl_rl.utils.quietpaw_evaluation import EpsilonEvaluation


class CorrectionTests(unittest.TestCase):
    def test_complete_grid_seeded_balanced_in_every_joint_cohort(self):
        for count in (160, 4096):
            terrain = torch.div(torch.arange(count), count / 20, rounding_mode="floor").long()
            command = (torch.arange(count) < count * .2).long()
            rng = torch.random.get_rng_state().clone()
            epsilon, labels = balanced_assignment(terrain, command, 71)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            torch.testing.assert_close(epsilon.flatten().sort().values, torch.linspace(0., 1., count))
            self.assertTrue(torch.equal(epsilon, balanced_assignment(terrain, command, 71)[0]))
            self.assertFalse(torch.equal(epsilon, balanced_assignment(terrain, command, 72)[0]))
            for group in torch.unique(torch.stack((terrain, command), dim=1), dim=0):
                mask = (terrain == group[0]) & (command == group[1])
                counts = torch.bincount(labels[mask], minlength=8)
                self.assertGreaterEqual(counts.min().item(), 1)
                self.assertLessEqual((counts.max() - counts.min()).item(), 1)

    def test_undersized_cohorts_fail_instead_of_claiming_balance(self):
        with self.assertRaisesRegex(ValueError, "cohort"):
            balanced_assignment(torch.arange(16), torch.zeros(16), 1)

    def test_first_episode_statistics_ignore_reset_and_distinguish_budget_units(self):
        stats = EpsilonEvaluation(2, .2, .5, "cpu")
        metrics = dict(noise_cost=torch.tensor([.4, .8]), failure=torch.tensor([True, False]),
                       touchdown_events=torch.tensor([1., 2.]), fz_peak=torch.tensor([10., 20.]))
        stats.add(torch.tensor([1., 2.]), torch.tensor([True, False]), metrics,
                  torch.tensor([False, False]), torch.tensor([[9.], [.3]]))
        metrics.update(noise_cost=torch.tensor([1., .4]), failure=torch.tensor([False, False]),
                       fz_peak=torch.tensor([9999., 30.]))
        stats.add(torch.tensor([999., 4.]), torch.tensor([False, False]), metrics,
                  torch.tensor([False, False]), torch.tensor([[9.], [.6]]), last_step=True)
        result = stats.summary()
        self.assertAlmostEqual(result["locomotion_reward"], 3.5)
        self.assertAlmostEqual(result["observed_discounted_reward"], 2.5)
        self.assertEqual(stats.episodes()["observed_discounted_reward"], [1., 4.])
        self.assertAlmostEqual(result["noise_cost"], .5)
        self.assertAlmostEqual(result["survival_success_rate"], .5)
        self.assertAlmostEqual(result["fz_peak_max"], 30.)
        # env0 failure: .5*.4=.2; env1 horizon: .5*.8+.25*.4+.25*.6=.65
        self.assertAlmostEqual(result["estimated_budget_cost"], .425, places=6)
        self.assertAlmostEqual(result["estimated_budget_violation"], .225, places=6)
        self.assertEqual(stats.steps.tolist(), [1., 2.])


if __name__ == "__main__":
    unittest.main()
