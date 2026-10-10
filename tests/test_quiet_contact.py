"""CPU hard gates for the quiet contact/reward measurements."""

import importlib.util
from pathlib import Path
import unittest

import torch


_PATH = Path(__file__).resolve().parents[1] / "legged_gym/legged_gym/utils/quiet_contact.py"
_SPEC = importlib.util.spec_from_file_location("quiet_contact_test_target", str(_PATH))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
QuietContactTracker = _MODULE.QuietContactTracker
touchdown_cost_rate = _MODULE.touchdown_cost_rate


class ContactTests(unittest.TestCase):
    def make_tracker(self, num_envs=1, num_feet=1, decimation=4, startup_time=0.0):
        return QuietContactTracker(num_envs, num_feet, "cpu", 0.005,
                                   decimation, startup_time=startup_time)

    def sample(self, tracker, force, velocity=-1.0, age=1.0):
        force = torch.full(tracker.contact.shape, float(force))
        velocity = torch.full(tracker.contact.shape, float(velocity))
        age = torch.full((tracker.contact.shape[0],), float(age))
        return tracker.observe(force, velocity, age)

    def test_first_sample_and_sustained_contact_do_not_count(self):
        tracker = self.make_tracker()
        tracker.begin_step()
        for _ in range(4):
            self.sample(tracker, 20.0)
        self.assertEqual(tracker.snapshot()["touchdown_count"].sum().item(), 0)

    def test_touchdown_uses_previous_sample_copy(self):
        tracker = self.make_tracker()
        tracker.begin_step()
        self.sample(tracker, 0.0, -0.2)
        self.sample(tracker, 0.0, -0.4)
        source = torch.tensor([[-0.8]])
        tracker.observe(torch.zeros_like(source), source, torch.ones(1))
        source.fill_(100.0)
        self.sample(tracker, 8.0, 0.0)
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["touchdown_count"].sum().item(), 1)
        self.assertAlmostEqual(snapshot["touchdown_cost"].item(), 0.64, places=6)

    def test_hysteresis_and_air_time_suppress_chatter(self):
        tracker = self.make_tracker()
        tracker.begin_step()
        self.sample(tracker, 8.0)
        self.sample(tracker, 1.0)  # remains in contact
        self.sample(tracker, 0.2)  # one 5 ms air sample
        self.sample(tracker, 8.0)  # insufficient air time
        self.assertEqual(tracker.snapshot()["touchdown_count"].sum().item(), 0)
        tracker.begin_step()
        self.sample(tracker, 0.0)
        self.sample(tracker, 0.0)
        self.sample(tracker, 9.0)
        self.sample(tracker, 9.0)
        self.assertEqual(tracker.snapshot()["touchdown_count"].sum().item(), 1)

    def test_window_crosses_control_boundary(self):
        tracker = self.make_tracker()
        tracker.begin_step()
        for _ in range(3):
            self.sample(tracker, 0.0)
        self.sample(tracker, 10.0)
        first = tracker.snapshot()
        self.assertEqual(first["touchdown_count"].sum().item(), 1)
        self.assertEqual(first["impact_count"].sum().item(), 0)
        tracker.begin_step()
        for force in (20.0, 15.0, 5.0, 3.0):
            self.sample(tracker, force)
        second = tracker.snapshot()
        self.assertEqual(second["impact_count"].sum().item(), 1)
        self.assertEqual(second["impact_peak"].item(), 20.0)
        self.assertAlmostEqual(second["impact_load"].item(), 0.25, places=6)
        self.assertEqual(first["touchdown_count"].sum().item(), 1)  # cloned

    def test_startup_exclusion_and_partial_reset(self):
        tracker = self.make_tracker(num_envs=2, startup_time=0.5)
        tracker.begin_step()
        for _ in range(3):
            self.sample(tracker, 0.0, age=0.1)
        self.sample(tracker, 10.0, age=0.1)
        self.assertEqual(tracker.snapshot()["touchdown_count"].sum().item(), 0)
        tracker.begin_step()
        self.sample(tracker, 0.0)
        self.sample(tracker, 0.0)
        self.sample(tracker, 10.0)
        terminal = tracker.snapshot(torch.tensor([True, False]))
        self.assertEqual(terminal["impact_invalid"][0].item(), 1)
        self.assertEqual(terminal["impact_invalid"][1].item(), 0)
        tracker.reset(torch.tensor([0]))
        self.assertFalse(tracker.initialized[0].item())
        self.assertTrue(tracker.initialized[1].item())
        self.assertEqual(terminal["touchdown_count"].sum().item(), 2)

    def test_squared_cost_and_dt_scaling(self):
        costs = torch.tensor([1.0, 0.25])
        penalty_50hz = 0.02 * touchdown_cost_rate(costs, 0.02, 4.0)
        penalty_100hz = 0.01 * touchdown_cost_rate(costs, 0.01, 4.0)
        self.assertTrue(torch.allclose(penalty_50hz, penalty_100hz))
        self.assertAlmostEqual(penalty_50hz[1].item() / penalty_50hz[0].item(), 0.25)
        with self.assertRaises(ValueError):
            touchdown_cost_rate(costs, 0.02, 0.0)

    def test_positive_vertical_velocity_has_zero_touchdown_cost(self):
        tracker = self.make_tracker()
        tracker.begin_step()
        for _ in range(3):
            self.sample(tracker, 0.0, 0.2)
        self.sample(tracker, 10.0, 0.0)
        self.assertEqual(tracker.snapshot()["touchdown_cost"].item(), 0.0)
        self.assertEqual(tracker.snapshot()["touchdown_count"].sum().item(), 1)


if __name__ == "__main__":
    unittest.main()
