import copy
import json
from pathlib import Path
import tempfile
import unittest

from go1_gym_deploy.himloco.config import load_config, readiness_errors, validate_config
from go1_gym_deploy.himloco.contract import SDK_JOINT_NAMES

EXAMPLE = Path(__file__).resolve().parents[2] / "configs" / "himloco_go1.example.json"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_template_inspectable_but_cannot_command(self):
        validate_config(self.config)
        self.assertIsNone(self.config["model"]["training_dof_names"])
        self.assertEqual(len(readiness_errors(self.config)), 6)
        with self.assertRaisesRegex(ValueError, "Live deployment is not ready"):
            validate_config(self.config, require_ready=True)

    def test_explicit_evidence_gates_all_required(self):
        model = self.config["model"]
        model.update(sha256="0" * 64, training_dof_names=list(SDK_JOINT_NAMES),
                     training_config_reference="SYNTHETIC TEST metadata")
        self.config["alignment"].update(verified=True, verification_reference="SYNTHETIC TEST",
                                        joint_mapping_verified=True, joint_mapping_reference="SYNTHETIC TEST")
        self.config["runtime"].update(safety_reviewed=True, safety_review_reference="SYNTHETIC TEST")
        validate_config(self.config, require_ready=True)
        self.assertEqual(readiness_errors(self.config), [])
        for group, key, invalid in (("model", "sha256", None),
                                    ("model", "training_dof_names", None),
                                    ("alignment", "verified", False),
                                    ("alignment", "joint_mapping_reference", None),
                                    ("runtime", "safety_reviewed", False)):
            bad = copy.deepcopy(self.config)
            bad[group][key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(bad, require_ready=True)

    def test_unknown_keys_and_contract_drift_rejected(self):
        cases = [(None, "foo", 1), ("model", "hip_reduction", 0.5),
                 ("runtime", "policy_dt", 0.002), ("runtime", "command_timeout_s", 0.2),
                 ("runtime", "state_timeout_s", 0.051), ("runtime", "rc_timeout_s", 0.2),
                 ("runtime", "max_cycle_lateness_s", 0.1), ("runtime", "history_warmup_frames", 5),
                 ("runtime", "startup_joint_margin_rad", .201),
                 ("runtime", "calibration_gain_ramp_s", 4),
                 ("runtime", "calibration_target_inset_rad", .02),
                 ("runtime", "safety_reviewed", "true"), ("runtime", "command_axis_signs", [1, 1, True]),
                 ("runtime", "command_deadzone", 1), ("runtime", "action_ramp_s", float("nan")),
                 ("alignment", "attitude_source", "xyzw"), (None, "source_commit", "main")]
        for group, key, value in cases:
            bad = copy.deepcopy(self.config)
            (bad[group] if group else bad)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(bad)

    def test_relative_model_path_resolves_from_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder) / "config.json"
            self.config["model"]["path"] = "models/policy.pt"
            config_path.write_text(json.dumps(self.config), encoding="utf-8")
            loaded = load_config(config_path)
            self.assertEqual(Path(loaded["model"]["path"]), Path(folder) / "models" / "policy.pt")
            validate_config(loaded)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.json"
            path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
