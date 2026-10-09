# Stage 1 fixed-budget audit — 2026-10-09

## Scope and current evidence

This audit does not change the training cost, PID formula, PPO objective, or HIM
auxiliary update. It adds a reproducible fixed-budget experiment around the
existing Stage 1 implementation. Stage 2 is not enabled.

The Oct08 force/impact and Oct09 speed-only CMDP8000 comparisons were trained
from scratch. They cannot establish what continued constrained training does
from the same HIM initialization. The source for this experiment is exclusively
`rough_go1/Sep29_01-01-22_baseline_seed1/model_8000.pt`; no later weights are used.

Confirmed in the current source:

- The speed-only optimization cost is the four-foot mean of
  `1-exp(-downward_speed**2/0.5)`, masked by a 50-Hz newly detected contact and
  valid previous state. It is a touchdown-speed proxy, not a contact-force or
  acoustic measurement. The 200-Hz observer is evaluation-only.
- Cost reward uses `(1-gamma)*c`, with terminal-observation timeout bootstrap
  in the same normalized value units. Failure has no bootstrap.
- Reward and cost advantages are combined before one common normalization.
  Thus lambda magnitude alone is insufficient evidence of constraint pressure;
  reward/cost gradient magnitudes and their directions are needed.
- PID integral can decrease under negative error. A clipped positive derivative
  does not by itself establish a bug. The formula is frozen for this experiment.
- A 20-second episode is approximately 1000 control steps at 0.02 seconds;
  the current strict `>` timeout check produces 1001. A surviving environment
  therefore completes about one episode per ten 100-step rollouts, not forty.
- Every PPO minibatch retains one HIM auxiliary optimizer update. The PPO
  backward pass checks that no gradient reaches the detached HIM estimator.

## Budget calibration completed

4096 environments per seed, stochastic HIM8000 actions, training commands,
terrain and randomization; one complete first episode per environment.

| Calibration seed | Mean observed normalized discounted cost |
| --- | ---: |
| 101 | 0.041772947 |
| 102 | 0.041885458 |

The preregistered rule gives **epsilon = 0.037646**. No cost critic prediction
enters this choice. With gamma=0.99 and 1001 steps, the maximum omitted timeout
tail is about 0.000042313 for a bounded [0,1] cost.

This is a diagnostic budget, not a claim about the paper's original epsilon
semantics or verified feasibility. Gamma=0.99 weights roughly the first two
seconds most heavily; full-episode alive-time cost and physical metrics must
also be reported. Training curriculum and changing policy occupancy may shift
the cost distribution beyond this initial first-episode calibration.

The initial deterministic 1 m/s evaluation passed 4000 observer PD-input
checks. It measured 10.416 touchdown events/s with the training 50-Hz detector,
versus 14.995 events/s with the 200-Hz observer. These counts are not event-matched:
the difference can include short contacts or contact chatter, and does not by
itself measure a missed-impact rate. It does establish that training and
evaluation touchdown counts are sampling-dependent. Keep this distinction
when interpreting a falling speed proxy against physical force metrics.

## Experiment and instrumentation

The frozen protocol is [quietpaw_fixed_budget_protocol_20261009.md](quietpaw_fixed_budget_protocol_20261009.md).
Training seeds 4 and 5 each receive a PID and zero-lambda arm, 300 updates each.
Both arms still train reward/cost critics and HIM. Model/optimizer/RNG hashes
are checked before the PID arm starts learning. All final evaluations use the
predeclared endpoint, not the most favorable intermediate checkpoint.

Added files under `experiments/quietpaw_fixed_budget_20261009/`:

- `worker.py`: calibration, matched training, stochastic and deployment evaluation.
- `probes.py`: reward/cost gradient decomposition before/after the actual joint
  PPO clipping gate. Norms include the actor and action-standard-deviation
  parameters. Cosine is between **loss contributions**: negative means conflict.
- `observer.py`: reused 200-Hz physical observer with runtime checks that refresh
  does not change the joint-state tensors read by the P controller.
- `run_study.py`: sequential frozen study, source/checkpoint hashes and fail-fast
  job records. Existing source modifications are included in the manifest.
- `analyze.py`: refuses incomplete studies; verifies optimizer step counts and
  reports seed-paired primary contrasts with survival and alive-time cost.

The observer's state-input check does not assert bitwise simulator trajectory
identity. Its 40-ms event peaks are contact-window measurements, not continuous
force maxima over an entire episode; short censored windows are reported.

Tests in `himloco`: 36 test executions passed (the probe test module also
imports the existing five-test helper class). New tests check that probes do
not change parameters, RNG or existing gradients, and that lambda=0 removes
the weighted cost gradient while retaining an unweighted cost signal.

## Artifacts and use

All run data is under `legged_gym/logs/quietpaw_fixed_budget_20261009/`.
`manifest.json` is the authoritative live status; an incomplete manifest is
not evidence of successful constrained training. The orchestration log is
`/tmp/quietpaw_fixed_study.log`.

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# Launch once only; do not start a second copy while a study is running.
WANDB_MODE=disabled python -u experiments/quietpaw_fixed_budget_20261009/run_study.py
# After the manifest records complete:
python experiments/quietpaw_fixed_budget_20261009/analyze.py
```

No conclusion about PID effectiveness or Stage 2 benefit is justified until
the four final checkpoints and their matched evaluations are complete.
