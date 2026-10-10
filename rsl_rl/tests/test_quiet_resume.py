"""Exercise real config/restore definitions without importing simulator code.

Isaac Gym must precede torch during simulator import. Load only the pure config
classes and restoration functions from their ASTs to keep this CPU gate usable
without Isaac Gym or a GPU. The tested functions are the production definitions.
"""

import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[2]


def source_definitions(path, names, namespace):
    module = ast.parse(path.read_text())
    module.body = [node for node in module.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(module, str(path), 'exec'), namespace)


class QuietResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {'copy': copy, 'inspect': inspect, 'SimpleNamespace': SimpleNamespace}
        base = REPO / 'legged_gym' / 'legged_gym'
        source_definitions(base / 'envs/base/base_config.py', ['BaseConfig'], namespace)
        source_definitions(base / 'envs/base/legged_robot_config.py',
                           ['LeggedRobotCfg', 'LeggedRobotCfgPPO'], namespace)
        source_definitions(base / 'envs/go1/go1_config.py',
                           ['Go1RoughCfg', 'Go1RoughCfgPPO'], namespace)
        source_definitions(base / 'envs/go1/quiet_config.py',
                           ['Go1QuietCfg', 'Go1QuietCfgPPO'], namespace)
        source_definitions(base / 'utils/helpers.py', ['class_to_dict'], namespace)
        source_definitions(base / 'scripts/train_quiet.py',
                           ['apply_saved_config', 'restore_resume_configuration'], namespace)
        cls.source = namespace

    def setUp(self):
        self.cfg = self.source['Go1QuietCfg']()
        self.train_cfg = self.source['Go1QuietCfgPPO']()
        original_cfg = self.source['Go1QuietCfg']()
        original_cfg.seed = 9
        original_cfg.env.num_envs = 80
        original_cfg.sim.dt = 0.0025
        original_cfg.control.decimation = 8
        original_cfg.domain_rand.friction_range = [0.3, 1.5]
        original_cfg.init_state.default_joint_angles['FL_hip_joint'] = 0.12
        original_train = self.source['Go1QuietCfgPPO']()
        original_train.algorithm.learning_rate = 1e-4
        original_train.algorithm.gamma = 0.97
        original_train.runner.num_steps_per_env = 64
        serialize = self.source['class_to_dict']
        self.metadata = {
            'environment_config': copy.deepcopy(serialize(original_cfg)),
            'algorithm_config': copy.deepcopy(serialize(original_train.algorithm)),
            'runner_config': copy.deepcopy(serialize(original_train.runner)),
            'calibration': {'steps': 800, 'touchdown_reference': 0.2,
                            'positive_tracking_reference': 1.0},
            'seed': 9, 'quiet_fraction': 0.15, 'ramp_iterations': 20,
            'policy': {'adaptation': 'head'},
            'evaluation_protocol': {'eval_interval': 10, 'eval_episodes': 12,
                                    'eval_duration': 5.0, 'skip_evaluation': False},
        }
        self.args = SimpleNamespace(num_envs=256, seed=1, adaptation='lora',
                                    quiet_fraction=0.1, ramp_iterations=50,
                                    calibration_steps=1000, eval_interval=50,
                                    eval_episodes=32, eval_duration=20.0,
                                    skip_evaluation=False, explicit_options=set())

    def restore(self):
        return self.source['restore_resume_configuration'](
            self.args, self.cfg, self.train_cfg, self.metadata)

    def test_resume_restores_saved_physics_task_and_training_before_construction(self):
        prior_metadata = copy.deepcopy(self.metadata)
        original_control = self.cfg.control
        self.restore()
        self.assertIs(self.cfg.control, original_control)
        self.assertEqual(self.cfg.sim.dt, 0.0025)
        self.assertEqual(self.cfg.control.decimation, 8)
        self.assertEqual(self.cfg.domain_rand.friction_range, [0.3, 1.5])
        self.assertEqual(self.cfg.init_state.default_joint_angles['FL_hip_joint'], 0.12)
        self.assertEqual(self.train_cfg.algorithm.learning_rate, 1e-4)
        self.assertEqual(self.train_cfg.algorithm.gamma, 0.97)
        self.assertEqual(self.train_cfg.runner.num_steps_per_env, 64)
        self.assertFalse(self.train_cfg.algorithm.update_estimator)
        self.assertEqual(self.args.num_envs, 80)
        self.assertEqual(self.args.seed, 9)
        self.assertEqual(self.args.adaptation, 'head')
        self.assertEqual(self.args.eval_episodes, 12)
        self.assertEqual(self.args.calibration_steps, 800)
        self.assertEqual(self.metadata, prior_metadata)
        self.assertTrue(callable(self.cfg.init_member_classes))
        self.assertEqual(self.source['class_to_dict'](self.cfg), self.metadata['environment_config'])

    def test_explicit_environment_count_is_allowed_without_overwriting_provenance(self):
        self.args.num_envs = 16
        self.args.explicit_options = {'--num_envs'}
        self.restore()
        self.assertEqual(self.cfg.env.num_envs, 16)
        self.assertEqual(self.metadata['environment_config']['env']['num_envs'], 80)

    def test_explicit_conflicting_learning_changes_are_rejected(self):
        alternatives = {'seed': 11, 'quiet_fraction': 0.99,
                        'adaptation': 'full', 'eval_duration': 3.0}
        for flag, value in alternatives.items():
            setattr(self.args, flag, value)
            self.args.explicit_options = {'--' + flag}
            with self.assertRaises(ValueError):
                self.restore()

    def test_legacy_v1_checkpoint_uses_documented_protocol_and_rollout(self):
        del self.metadata['runner_config']
        del self.metadata['evaluation_protocol']
        self.metadata['diagnostic_only'] = True
        self.restore()
        self.assertEqual(self.train_cfg.runner.num_steps_per_env, 100)
        self.assertEqual(self.args.eval_episodes, 32)
        self.assertEqual(self.args.eval_duration, 20.0)
        self.assertEqual(self.args.eval_interval, 50)
        self.assertTrue(self.args.skip_evaluation)

    def test_resume_quiet_values_are_local_to_the_restored_config(self):
        self.metadata['environment_config']['quiet']['touchdown_reference'] = 3.0
        self.metadata['environment_config']['quiet']['target_weight'] = .25
        self.restore()
        self.assertEqual(self.cfg.quiet.touchdown_reference, 3.0)
        self.assertEqual(self.cfg.quiet.target_weight, .25)
        fresh = self.source['Go1QuietCfg']()
        self.assertIsNone(fresh.quiet.touchdown_reference)
        self.assertEqual(fresh.quiet.target_weight, 0.0)
        self.assertIsNone(self.source['Go1QuietCfg'].quiet.touchdown_reference)


if __name__ == '__main__':
    unittest.main()
