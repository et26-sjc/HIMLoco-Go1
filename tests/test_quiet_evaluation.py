"""CPU tests for episode masking, quietness evidence, and paired screening."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np


_PATH = Path(__file__).resolve().parents[1] / "legged_gym/legged_gym/utils/quiet_evaluation.py"
_SPEC = importlib.util.spec_from_file_location("quiet_evaluation_test_target", str(_PATH))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def snapshot(failed=(False, False), speeds=(.4, .4), time=1.0, positions=(.3, .3)):
    event = np.zeros((2, 4, 1), dtype=bool)
    event[:, 0, 0] = True
    sample = np.zeros((2, 4, 1))
    sample[:, 0, 0] = speeds
    peaks = np.zeros_like(sample)
    peaks[:, 0, 0] = 30.
    loads = np.zeros_like(sample)
    loads[:, 0, 0] = .3
    return {
        "track_error": np.array([.01, .01]), "yaw_error": np.zeros(2),
        "base_velocity": np.array([[.6, 0., 0.], [.6, 0., 0.]]),
        "projected_gravity": np.array([[0., 0., -1.], [0., 0., -1.]]),
        "torques": np.ones((2, 12)), "dof_velocity": np.ones((2, 12)),
        "contact": np.ones((2, 1), dtype=bool), "foot_velocity": np.zeros((2, 1, 3)),
        "root_position": np.array([[positions[0], 0., 0.], [positions[1], 0., 0.]]),
        "touchdown_count": np.ones((2, 1), dtype=np.int64),
        "impact_count": np.ones((2, 1), dtype=np.int64),
        "impact_invalid": np.zeros((2, 1), dtype=np.int64),
        "touchdown_event": event, "touchdown_speed": sample,
        "impact_event": event.copy(), "impact_peak_samples": peaks,
        "impact_load_samples": loads, "episode_time": np.full(2, time),
        "failed": np.asarray(failed), "done": np.asarray(failed),
    }


class EvaluationTests(unittest.TestCase):
    def test_invalid_push_suite_stops_before_simulator_or_model_use(self):
        script_path = _PATH.parents[1] / 'scripts/evaluate_quiet.py'
        spec = importlib.util.spec_from_file_location('quiet_preflight_script_test', str(script_path))
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        cfg = SimpleNamespace(sim=SimpleNamespace(dt=.005), control=SimpleNamespace(decimation=4))
        registry = SimpleNamespace(get_cfgs=mock.Mock(return_value=(cfg, None)),
                                   make_env=mock.Mock())
        # No simulator is supplied. Correct preflight raises before make_env
        # or any access to the model sentinel.
        modules = {'isaacgym': SimpleNamespace(gymtorch=object()),
                   'legged_gym.envs': SimpleNamespace(task_registry=registry),
                   'legged_gym.utils.quiet_evaluation': _MODULE}
        with mock.patch.dict('sys.modules', modules):
            with self.assertRaisesRegex(ValueError, 'extend past the scheduled push'):
                script.evaluate_policy(object(), SimpleNamespace(), suite='all', duration=5.)
        registry.make_env.assert_not_called()

    def test_suite_preflight_push_and_horizon_boundaries(self):
        conditions = _MODULE.evaluation_conditions("all")
        for duration in (4.98, 5.0):
            with self.assertRaisesRegex(ValueError, "extend past the scheduled push"):
                _MODULE.validate_evaluation_schedule(conditions, duration, .02)
        steps, pushes = _MODULE.validate_evaluation_schedule(conditions, 6., .02)
        self.assertEqual(steps, 300)
        self.assertEqual(pushes["flat_push"], 250)
        # A flat-only short suite remains useful for simulator smoke checks.
        self.assertEqual(_MODULE.validate_evaluation_schedule(
            _MODULE.evaluation_conditions("flat"), 4.98, .02)[0], 249)

    def test_schedule_accepts_float_rounding_and_final_step_push(self):
        push = {"name": "rounded", "push_time": .02 * 249}
        steps, pushes = _MODULE.validate_evaluation_schedule([push], .02 * 250, .02)
        self.assertEqual(steps, 250)
        self.assertEqual(pushes["rounded"], 249)
        self.assertEqual(_MODULE.validate_evaluation_schedule([], .1 + .2, .01)[0], 30)

    def test_schedule_rejects_nonfinite_negative_or_unaligned_times(self):
        for duration in (float("nan"), float("inf"), -1., 0., 6.001):
            with self.assertRaises(ValueError):
                _MODULE.validate_evaluation_schedule([], duration, .02)
        for dt in (float("nan"), float("inf"), -.02, 0.):
            with self.assertRaises(ValueError):
                _MODULE.validate_evaluation_schedule([], 6., dt)
        for push_time in (float("nan"), float("inf"), -.02, 5.001):
            with self.assertRaises(ValueError):
                _MODULE.validate_evaluation_schedule(
                    [{"name": "invalid", "push_time": push_time}], 6., .02)

    def test_config_copy_keeps_data_dictionaries_and_isolates_classes(self):
        script_path = _PATH.parents[1] / "scripts/evaluate_quiet.py"
        spec = importlib.util.spec_from_file_location("quiet_eval_script_test", str(script_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Config:
            class control:
                stiffness = {"joint": 40.}
                decimation = 4
        clone = module._clone_config(Config())
        clone.control.decimation = 2
        clone.control.stiffness["joint"] = 20.
        self.assertEqual(Config.control.decimation, 4)
        self.assertEqual(Config.control.stiffness["joint"], 40.)
        self.assertEqual(list(clone.control.stiffness.keys()), ["joint"])

    def metrics(self):
        condition = _MODULE.evaluation_conditions("flat")[1]
        return _MODULE.QuietEpisodeMetrics(condition, 2, .5, 1., ["foot"], np.full(12, 20.))

    def report(self):
        metrics = self.metrics()
        metrics.set_initial_position(np.zeros((2, 3)))
        metrics.add(snapshot(time=.5))
        metrics.add(snapshot(time=1., positions=(.6, .6)))
        return {"split": "validation", "conditions": [metrics.finish()]}

    def test_terminal_step_counted_but_next_reset_episode_masked(self):
        metrics = self.metrics()
        metrics.add(snapshot(failed=(True, False)))
        metrics.add(snapshot(speeds=(100., .2)))
        result = metrics.finish()
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["failure_count"], 1)
        self.assertEqual(result["touchdown_count"], 3)
        self.assertLess(result["touchdown_velocity_p95"], .5)
        self.assertAlmostEqual(result["tracking_xy_rmse"], .1)

    def test_event_squared_cost_and_units(self):
        metrics = self.metrics()
        metrics.set_initial_position(np.zeros((2, 3)))
        metrics.add(snapshot(speeds=(.4, .2), time=.5))
        metrics.add(snapshot(speeds=(.4, .2), time=1., positions=(.6, .6)))
        result = metrics.finish()
        self.assertAlmostEqual(result["touchdown_velocity_rms"], np.sqrt(.1))
        self.assertAlmostEqual(result["touchdown_cost_per_second"], .4)
        self.assertEqual(result["impact_complete_count"], 4)
        self.assertEqual(result["completion_rate"], 1.)

    def test_drift_is_converted_to_joint_radians(self):
        metrics = self.metrics()
        metrics.add(snapshot(), action_drift=np.full((2, 12), .1))
        result = metrics.finish()
        self.assertAlmostEqual(result["action_drift_rad_rms"], .025)
        self.assertAlmostEqual(result["action_drift_rad_p95"], .025)

    def test_actual_clipping_and_deployment_range_are_distinct(self):
        metrics = self.metrics()
        raw = np.tile([1., -1.1, 100., -101.], (2, 3))
        metrics.add(snapshot(failed=(True, False)), raw_actions=raw)
        # Discard the reset robot's large actions; count the live robot only.
        metrics.add(snapshot(), raw_actions=np.array([[1000.] * 12, [.5] * 12]))
        result = metrics.finish()
        self.assertEqual(result["action_metric_sample_count"], 3)
        self.assertAlmostEqual(result["action_clip_fraction"], 1. / 6.)
        self.assertAlmostEqual(result["deployment_action_limit_exceeded_fraction"], .5)
        self.assertEqual(result["action_clip_limit"], 100.)
        self.assertEqual(result["deployment_action_limit"], 1.)
        self.assertNotIn("action_saturation_fraction", result)

    def test_missing_action_samples_are_not_reported_as_zero_clipping(self):
        result = self.report()["conditions"][0]
        self.assertIsNone(result["action_clip_fraction"])
        self.assertIsNone(result["deployment_action_limit_exceeded_fraction"])
        self.assertEqual(result["action_metric_sample_count"], 0)

    def test_action_limit_settings_are_positive_and_paired(self):
        for limit in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                _MODULE.QuietEpisodeMetrics({}, 2, .02, 6., [], np.ones(12),
                                           deployment_action_limit=limit)
        base = self.report()
        candidate = copy.deepcopy(base)
        candidate["conditions"][0]["deployment_action_limit"] = 2.
        self.assertFalse(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])

    def test_report_format_versions_must_match(self):
        base = self.report()
        candidate = copy.deepcopy(base)
        base['format_version'] = 1
        candidate['format_version'] = _MODULE.QUIET_EVALUATION_FORMAT_VERSION
        self.assertFalse(_MODULE.select_candidate(candidate, base, require_quiet=False)['accepted'])

    def test_seed_does_not_depend_on_suite_order_and_test_is_separate(self):
        seed = _MODULE.condition_seed("flat_forward_0.6", "validation")
        self.assertEqual(seed, _MODULE.condition_seed("flat_forward_0.6", "validation"))
        self.assertNotEqual(seed, _MODULE.condition_seed("flat_forward_0.6", "test"))

    def test_quieter_policy_passes_only_with_motion_retained(self):
        base = self.report()
        candidate = copy.deepcopy(base)
        result = candidate["conditions"][0]
        result["touchdown_velocity_rms"] *= .8
        result["touchdown_velocity_p95"] *= .8
        result["touchdown_cost_per_second"] *= .64
        self.assertTrue(_MODULE.select_candidate(candidate, base)["accepted"])
        result["actual_command_speed"] = .1
        self.assertFalse(_MODULE.select_candidate(candidate, base)["accepted"])

    def test_no_touchdown_is_not_a_quiet_success(self):
        base = self.report()
        candidate = copy.deepcopy(base)
        candidate["conditions"][0]["touchdown_valid"] = False
        candidate["conditions"][0]["touchdown_velocity_rms"] = None
        self.assertFalse(_MODULE.select_candidate(candidate, base)["accepted"])

    def test_rejects_unpaired_tests(self):
        base = self.report()
        candidate = copy.deepcopy(base)
        candidate["split"] = "test"
        self.assertFalse(_MODULE.select_candidate(candidate, base)["accepted"])
        candidate["split"] = "validation"
        candidate["conditions"][0]["episodes"] += 1
        self.assertFalse(_MODULE.select_candidate(candidate, base)["accepted"])

    def test_optional_sampling_metadata_must_match_when_present(self):
        base = self.report()
        candidate = copy.deepcopy(base)
        self.assertTrue(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])
        candidate["conditions"][0]["physical_dt_s"] = .005
        self.assertFalse(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])
        base["conditions"][0]["physical_dt_s"] = .005
        self.assertTrue(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])
        candidate["conditions"][0]["control_dt_s"] = .02
        base["conditions"][0]["control_dt_s"] = .01
        self.assertFalse(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])

    def test_command_terrain_difficulty_and_seed_are_paired(self):
        base = self.report()
        for field, changed in (("command", [.5, 0., 0.]), ("terrain", "trimesh"),
                               ("difficulty", .8)):
            candidate = copy.deepcopy(base)
            candidate["conditions"][0]["condition"][field] = changed
            self.assertFalse(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])
        candidate = copy.deepcopy(base)
        candidate["conditions"][0]["seed"] = 100
        self.assertFalse(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])
        candidate = copy.deepcopy(base)
        candidate["duration_s"] = 20.
        base["duration_s"] = 2.
        self.assertFalse(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])

    def test_short_horizon_is_not_push_validation(self):
        base = self.report()
        base["conditions"][0]["condition"]["push_time"] = 5.
        candidate = copy.deepcopy(base)
        result = _MODULE.select_candidate(candidate, base, require_quiet=False)
        self.assertFalse(result["accepted"])
        self.assertIn("push validation is incomplete", " ".join(result["reasons"]))
        candidate["conditions"][0]["push_applied"] = True
        base["conditions"][0]["push_applied"] = True
        self.assertTrue(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])

    def test_rough_screen_needs_actual_platform_exit_evidence(self):
        base = self.report()
        condition = base["conditions"][0]
        condition["condition"]["terrain"] = "trimesh"
        condition["condition"]["quiet_eligible"] = False
        condition["terrain_platform_exit_count"] = 0
        candidate = copy.deepcopy(base)
        result = _MODULE.select_candidate(candidate, base, require_quiet=False)
        self.assertFalse(result["accepted"])
        self.assertIn("rough validation is incomplete", " ".join(result["reasons"]))
        candidate["conditions"][0]["terrain_platform_exit_count"] = 2
        self.assertFalse(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])
        base["conditions"][0]["terrain_platform_exit_count"] = 2
        self.assertTrue(_MODULE.select_candidate(candidate, base, require_quiet=False)["accepted"])

    def test_more_failures_rejected_without_discarding_quiet_data(self):
        base = self.report()
        candidate = copy.deepcopy(base)
        candidate["conditions"][0]["failure_count"] = 1
        result = _MODULE.select_candidate(candidate, base, require_quiet=False)
        self.assertFalse(result["accepted"])
        self.assertIn("failure count", " ".join(result["reasons"]))


if __name__ == "__main__":
    unittest.main()
