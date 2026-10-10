"""CPU regressions for instance calibration and raw-action measurements.

Execute the real config/robot definitions with only the simulator base mocked.
This keeps Isaac Gym and a GPU out of these state-isolation checks.
"""
import ast
import importlib.util
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


BASE = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym'


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(tree, str(path), 'exec'), namespace)


class SimulatorBase:
    def _init_buffers(self):
        pass

    def step(self, actions):
        self.received_actions = actions
        limit = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -limit, limit)
        return self.actions


class QuietConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('quiet_contact_config_test',
                                                      str(BASE / 'utils/quiet_contact.py'))
        contact = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contact)
        namespace = dict(inspect=inspect, torch=torch, math=math, LeggedRobot=SimulatorBase,
                         QuietContactTracker=contact.QuietContactTracker,
                         touchdown_cost_rate=contact.touchdown_cost_rate)
        definitions(BASE / 'envs/base/base_config.py', ['BaseConfig'], namespace)
        definitions(BASE / 'envs/base/legged_robot_config.py',
                    ['LeggedRobotCfg', 'LeggedRobotCfgPPO'], namespace)
        definitions(BASE / 'envs/go1/go1_config.py', ['Go1RoughCfg', 'Go1RoughCfgPPO'], namespace)
        definitions(BASE / 'envs/go1/quiet_config.py', ['Go1QuietCfg'], namespace)
        definitions(BASE / 'envs/go1/quiet_robot.py', ['Go1QuietRobot'], namespace)
        cls.Config = namespace['Go1QuietCfg']
        cls.Robot = namespace['Go1QuietRobot']

    def robot(self, cfg):
        env = self.Robot()
        env.cfg, env.num_envs, env.device = cfg, 1, 'cpu'
        env.feet_indices = torch.tensor([0])
        env.root_states = torch.zeros(1, 13)
        env.sim_params = SimpleNamespace(dt=cfg.sim.dt)
        env._init_buffers()
        return env

    def test_nested_config_is_instantiated_and_calibration_does_not_mutate_shared_config(self):
        shared_cfg = self.Config()
        self.assertFalse(isinstance(shared_cfg.quiet, type))
        first = self.robot(shared_cfg)
        first.set_quiet_reward(4.0, .2, .5)
        self.assertEqual(first.quiet_reference, 4.0)
        self.assertEqual(first.quiet_target_weight, .2)
        self.assertEqual(first.quiet_progress, .5)
        self.assertIsNone(shared_cfg.quiet.touchdown_reference)
        self.assertEqual(shared_cfg.quiet.target_weight, 0.0)
        # The registry can hand the exact same cfg object to a second robot.
        second = self.robot(shared_cfg)
        self.assertIsNone(second.quiet_reference)
        self.assertEqual(second.quiet_target_weight, 0.0)
        self.assertEqual(second._reward_quiet_touchdown().item(), 0.0)
        self.assertIsNone(self.Config().quiet.touchdown_reference)
        self.assertIsNone(self.Config.quiet.touchdown_reference)

    def test_measurements_use_raw_actions_without_changing_them(self):
        env = self.robot(self.Config())
        raw = torch.tensor([[1.0, -1.1, 100.0, -101.0]])
        original = raw.clone()
        result = env.step(raw)
        self.assertTrue(torch.equal(raw, original))
        self.assertIs(env.received_actions, raw)
        self.assertTrue(torch.equal(result, torch.tensor([[1.0, -1.1, 100.0, -100.0]])))
        self.assertEqual(env.quiet_action_clip_fraction.item(), .25)
        self.assertEqual(env.quiet_deployment_limit_exceeded_fraction.item(), .75)


if __name__ == '__main__':
    unittest.main()
