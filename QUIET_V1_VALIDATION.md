# Quiet V1 validation record

Date: 2026-10-09. These tests check the implementation. They do not establish
acoustic improvement or a Pareto front.

## Runtime and source

- Python: 3.8, from the `HIM` conda environment.
- PyTorch: 2.0.1+cu118.
- Simulator: Isaac Gym, GPU PhysX.
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU.
- Baseline: `legged_gym/legged_gym/model_7500.pt`.
- Baseline SHA256: `2de4c315944b5db672c3af79c09a0d25867e37ba4c4e3c03ba503b322a3143a9`.

The original checkpoint was not changed. Its stored iteration is zero. Do not
infer a training iteration from its filename.

## Completed checks

| Check | Result |
|---|---|
| Policy, PPO and resume tests | 12 passed |
| Contact and evaluation tests | 20 passed |
| GPU training: 64 robots, 5 PPO updates | Passed |
| GPU resume: 64 robots, 2 further updates | Passed |
| Integrated train/evaluate: 8 robots, 1 update | Passed |
| Evaluation: 14 conditions, 2 episodes each, 6 s | Passed for baseline and adapted policy |
| Original actor and estimator tensors after training | Exact match to baseline |
| Disabled adapter means and standard deviations | Exact match to baseline |
| Merged JIT action means | Match within `1e-5` absolute float32 error |
| Original checkpoint hash after all runs | Unchanged |

The 64-robot run collected 1,769 baseline touchdown events during calibration.
All rewards, returns, advantages and model parameters stayed finite. Adapter
parameters changed. Frozen parameters did not change. Gaussian standard
deviations stayed positive.

The integrated test applied the scheduled push to both robots at 5 s. Both
robots left the central platform in each of the six rough-terrain conditions.
Neither policy had a failed episode in this short test. This sample is too small
to establish a failure rate or retained terrain performance.

The integrated policy did not pass selection. It did not reduce touchdown RMS
and p95 by 10% at every forward speed. Some slip and force metrics exceeded their
budgets. No `selected.pt` was produced. This is the expected decision for an
unqualified early checkpoint.

## Result files

- `legged_gym/logs/quiet_go1/smoke64/`: calibration, five-update log, checkpoint
  and merged JIT policy.
- `legged_gym/logs/quiet_go1/smoke64_resume/`: resume log and `model_7.pt`.
- `legged_gym/logs/quiet_go1/integrated_smoke8/`: paired baseline/candidate
  reports, selection reasons, source fingerprint and `model_1.pt`.

These local logs are ignored by Git. Checkpoint metadata records the experiment
configuration. New checkpoints also record a hash of the local Python source,
including source files that have not been committed.

The first integrated attempt failed in the native simulator. The next attempt
reported GPU allocation failures with two processes. Evaluation now runs in a
separate process, and V1 reserves fewer GPU contact pairs. The full integrated
test then completed. Physics and control periods remain 5 ms and 20 ms.

## Next experiment

Run the 256-robot, 200-update experiment in [QUIET_V1.md](QUIET_V1.md). Keep the
baseline and candidate evaluation protocol identical. Use at least three
training seeds after the first useful result. Compare output-head adaptation,
LoRA adaptation and full actor tuning under the same training budget.

Confirm a useful touchdown reduction before adding motion-cost value heads or
terrain experts. Measure the acoustic result on the robot at matched actual
speed. Use later constrained experiments to estimate the empirical Pareto front.

## Review corrections

The three review points were checked against the real config construction.
`BaseConfig` instantiates nested classes. A fresh config therefore has its own
`quiet` object. The registry still returns a shared config instance. Runtime
reward calibration now changes robot-instance attributes only. Resume tests
also check that restored quiet values do not change a fresh config or its class.

Action reports now give the raw-action clip fraction at the Isaac Gym limit and
the fraction beyond the current MuJoCo deployment boundary of 1. They include
both limits. Missing raw samples produce null values. They do not report these
two values as physical saturation. The report format is version 2; resume
recomputes a baseline report from the old format.

The complete evaluation schedule is checked before simulation starts. A push at
5 s requires a duration longer than 5 s, including the exact-equality case.
Invalid values and partial control periods fail early. Short flat-only tests
remain valid. The selector still rejects a valid schedule if all robots fail
before the push occurs.

All 43 CPU tests passed: 13 policy/resume tests and 30 config/contact/evaluation
tests. A fresh 8-robot, one-update GPU run also passed. At the same seed, its
losses, touchdown cost and adapter-update norm exactly matched the previous
one-update run.

Seven flat conditions were checked with two episodes of 1 s each. Actual Isaac
clipping was zero. The new deployment-range fraction was nonzero: 34.58% at the
0.6 m/s command. This establishes that the two measurements report different
facts. It does not establish locomotion or quietness performance.

Results: `legged_gym/logs/quiet_go1/review_fix_smoke8/`.
