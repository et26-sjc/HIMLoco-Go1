"""Experiment diagnostics must preserve optimizer results and random state."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import test_quietpaw_constraint_validation as helpers
import torch

spec=importlib.util.spec_from_file_location('gain_instrumentation',Path(__file__).resolve().parents[1]/
    'experiments/quietpaw_gain_20261009/instrumentation.py')
instrumentation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(instrumentation)


class GainInstrumentationTests(unittest.TestCase):
    def test_identical_learning_rng_and_pid(self):
        helper=helpers.ConstraintValidationTests()
        torch.set_num_threads(1)
        first=helper.algorithm(learning_rate=1e-5)
        second=helper.algorithm(learning_rate=1e-5)
        for alg in (first,second):
            alg.pid.multiplier.fill_(3.)
            helper.collect(alg)
        states=[]
        losses=[]
        with tempfile.TemporaryDirectory() as folder:
            instrumentation.install(second,folder,endpoint=1)
            for alg in (first,second):
                torch.manual_seed(97)
                losses.append(alg.update())
                states.append(torch.get_rng_state())
            self.assertEqual(losses[0],losses[1])
            self.assertTrue(torch.equal(*states))
            for k,v in first.actor_critic.state_dict().items():
                self.assertTrue(torch.equal(v,second.actor_critic.state_dict()[k]),k)
            for k,v in first.pid.state_dict().items():
                self.assertTrue(torch.equal(v,second.pid.state_dict()[k]),k)
            for left,right in [(first.optimizer,second.optimizer),
                               (first.actor_critic.estimator.optimizer,second.actor_critic.estimator.optimizer)]:
                l,r=left.state_dict(),right.state_dict()
                self.assertEqual(l['param_groups'],r['param_groups'])
                for k,state in l['state'].items():
                    for field,value in state.items():
                        self.assertTrue(torch.equal(value,r['state'][k][field]))
            self.assertTrue((Path(folder)/'pid_inputs.pt').exists())


if __name__=='__main__':
    unittest.main()
