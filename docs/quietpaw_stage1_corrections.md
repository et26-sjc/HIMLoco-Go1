# Stage 1 review corrections C1–C3

These corrections change target construction, experimental assignment and
measurement, not the optimization algorithm. Noise proxy, reward definition,
HIM auxiliary losses, observation/history, PD, robot, training terrain and
domain randomization settings remain unchanged. Original baseline source
files retain the Stage 0 fingerprints.

## C1: terminal bootstrap for both critics

Previously Stage 1 delegated reward processing to `HIMPPO.process_env_step`,
which added gamma times the **pre-action** reward value on timeout. The old
test explicitly asserted that behavior and was incorrect.

Stage 1 now stores transitions directly after evaluating both critics on the
runner's restored, pre-reset terminal critic observation and the transition's
epsilon. A timeout contributes `gamma * Vr(terminal, epsilon)` to the reward
target and `gamma * Vc(terminal, epsilon)` to the cost target. A non-timeout
failure gets no bootstrap. Neither correction mutates the environment reward.
If physical failure and timeout coincide, failure wins for both target paths.
The physical contact-failure signal is reported separately without changing
the simulator's reset behavior. The old `transition.values = Vr(s_t, epsilon)` is preserved for GAE subtraction
and PPO value clipping. Done still cuts GAE recursion into the reset episode.

HIMPPO itself is unchanged. This intentionally corrects a baseline timeout
convention in the Stage 1 extension; it cannot be described as bitwise
equivalent to the old Stage 1 reward targets. The revised test uses distinct
current/terminal values and an epsilon-dependent critic, checks the old value
is retained, and verifies that a true terminal transition has no correction.

## C2: seeded, balanced assignment over actual joint cohorts

The full set of N distinct `linspace(0,1,N)` values is retained. Cohorts are
formed from the actual `env.terrain_types` and the exact command rule
`env_id < 0.2*N`. With a dedicated CPU RNG (`epsilon_seed=1` by default), each
joint cohort's environment IDs are shuffled, then assigned cyclic epsilon-bin
labels; the actual grid values within each bin are independently shuffled.
There are eight bins by default. This does not consume the simulator/training
global RNG or change environment indexing, terrain generation or commands.

Every nonempty **terrain × command** cohort contains all eight epsilon bins,
with per-bin counts differing by at most one; consequently terrain and command
groups both cover the full constraint range. The grid as a whole is an exact
permutation of the original full grid, with no dropped/duplicated values.
The mapping stays fixed through resets and PPO updates. Every environment
continues to own a separate PID multiplier.

An N-element unique grid cannot have every exact value in each cohort smaller
than N. “All epsilon levels” here therefore means coverage of every constraint
interval, not impossible replication of all N unique scalars. Exact pointwise
comparisons are supplied by C3: each selected evaluation epsilon runs on the
same complete environment set.

Each joint cohort must have at least eight environments. In the default Go1
20-column layout, use 160 environments for the balanced smoke test (4096
default environments also pass). Small 16-environment Stage 1 tests are now
rejected explicitly rather than pretending to be balanced. Other counts can
produce small boundary cohorts and are checked from the actual mapping.

The runner writes `epsilon_assignment.json` with seed, bins, environment-wise
epsilon, bin, terrain and command-cohort IDs. The checkpoint also contains this
mapping. Contract v2 records the corrected target/assignment semantics. Resume
requires identical mapping/topology, epsilon grid and algorithm contract;
pre-correction v1 runs are rejected rather than silently continuing biased
targets and an incompatible multiplier assignment.

## C3: training groups and fixed-condition epsilon sweep

Training TensorBoard/W&B diagnostics now include, for every epsilon bin:
raw locomotion reward, raw noise cost, tracking error, force/force-rate peaks,
base acceleration, touchdown event counts and downward touchdown speed, tilt,
failure frequency, completed-episode count, timeout completion fraction,
lambda and observed completed-episode budget violation. Bins with no completed
episode do not emit a fabricated success/violation value; the episode count
is always logged. Per-step reward/physics statistics are unbootstrapped.

`legged_gym/legged_gym/scripts/evaluate_quietpaw.py` launches a fresh process
per epsilon with the same seed and commands. An evaluation-only subclass fixes
the commanded `(vx,vy,yaw_rate)` even after heading conversion/resampling and
freezes terrain/command curriculum updates. Training configs are not edited.
Terrain generation, initial difficulty distribution, PD, physics and domain
randomization remain the Go1 settings.

Before the reset's zero-action physics step, each worker records the assigned
terrain types/levels, root/joint states and available randomized parameter
tensors. The sweep requires their hashes to match before producing a paired
summary. The seed is also applied to terrain generation. GPU PhysX need not
be bitwise deterministic after stepping, and policy-dependent resets can
change subsequent random draws; matching seed/initial states is not a claim
of identical full disturbance trajectories. Repeat across seeds for research
conclusions. No evaluation result is used as an optimization cost.

For each environment, only its first episode, up to the chosen horizon, counts.
Failed environments stop contributing immediately even though the simulator
auto-resets them. Outputs include per-episode records and epsilon summaries:

- Locomotion return and mean reward per valid step.
- Mean raw noise cost and observed normalized discounted cost.
- Estimated budget cost/violation including cost-critic tail bootstrap on a
  time limit or evaluation horizon. Failures have no tail. This is explicitly
  an **estimate**, sensitive to critic accuracy, not a direct noise measurement.
- `max(mean(Jc)-epsilon,0)` and mean per-episode violation are distinct fields.
- Tracking error, survival-to-horizon success rate, full timeout completion
  rate and survival duration. Short-horizon survival is not 20-second success.
- Sampled fz/f3d/df3d peaks, base acceleration, touchdown counts and touchdown
  speed. Both per-episode step means and temporal maxima are reported; these
  remain policy-rate samples, not true physics-substep acoustic peaks.

Each sweep produces `epsilon_XX.json`/`.log`, `summary.csv`, `summary.json`
and `reward_noise.png`. JSON contains initial states, checkpoint SHA-256,
configuration, command, seed, horizon and cost contract. The plotted frontier
is empirical non-dominance for return versus mean proxy cost in that sweep;
always examine success/tracking alongside it. A failed short episode can
appear cheap, and a smoke-test frontier is not a scientific Pareto result.

## Commands

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

WANDB_MODE=disabled python legged_gym/legged_gym/scripts/train_quietpaw.py \
  --task go1_quietpaw_stage1 --headless --num_envs 160 --max_iterations 2 \
  --seed 1 --run_name stage1_correction

python legged_gym/legged_gym/scripts/evaluate_quietpaw.py \
  --checkpoint legged_gym/logs/quietpaw_go1_stage1/YOUR_RUN/model_2.pt \
  --output legged_gym/logs/epsilon_sweep --epsilons 0 0.1 0.2 0.4 0.6 0.8 1 \
  --num-envs 160 --steps 1000 --seed 7 --command 1 0 0

python -m unittest discover -s tests -p 'test_quietpaw*.py' -v
python -m compileall -q rsl_rl/rsl_rl legged_gym/legged_gym tests
git diff --check
```

No value decomposition, alternate optimizer, noise reward or new locomotion
objective was introduced. Noise definition/budget/PID assumptions from the
Stage 1 document still apply. Correcting C1 and C2 does not establish that
QuietPaw's unspecified implementation choices have been faithfully recovered.

## Files changed

Modified:

- `rsl_rl/rsl_rl/algorithms/quietpaw_constrained_ppo.py`: terminal reward
  bootstrap, assignment hookup, epsilon-bin training aggregation.
- `rsl_rl/rsl_rl/runners/quietpaw_stage1_runner.py`: actual cohort topology,
  saved assignment JSON, v2 contract and resume checks.
- `legged_gym/legged_gym/envs/go1/quietpaw_go1_config.py`: assignment seed/bins.
- `legged_gym/legged_gym/envs/go1/quietpaw_legged_robot.py`: touchdown speed
  diagnostic and physical failure signal even when timeout coincides.
- `tests/test_quietpaw_stage1.py`: corrected target expectation, failure
  precedence, topology resume rejection, conditional diagnostic checks.
- `docs/quietpaw_stage1.md`: remove obsolete timeout/assignment/evaluation claims.

Added:

- `rsl_rl/rsl_rl/algorithms/quietpaw_assignment.py`: stratified full-grid mapping.
- `rsl_rl/rsl_rl/utils/quietpaw_evaluation.py`: first-episode metric accumulator.
- `legged_gym/legged_gym/scripts/evaluate_quietpaw.py`: paired-condition sweep.
- `tests/test_quietpaw_corrections.py`: assignment and evaluation arithmetic tests.
- This correction record.

## Validation (2026-09-29, conda himloco)

- **22 tests pass**, including the original seven Stage 0 tests and frozen
  source hashes. No new environment or dependencies were installed.
- Explicit assignment coverage tests pass for **160 and 4096** environments:
  every joint cohort covers all bins, within-cohort counts differ by at most
  one, the entire grid is retained, the mapping is seed-reproducible, and
  the global RNG state is unchanged. Undersized cohorts are rejected.
- Corrected Stage 1 trained for two iterations with 160 environments
  (**32,000 transitions**), then completed two separate resumed iterations.
  Checkpoint PPO/HIM Adam steps matched **40 → 60 → 80**, with finite parameters.
- The saved JSON assignment equals the checkpoint mapping. Training resume
  succeeds with this mapping and tests reject a modified topology.
- TensorBoard event inspection found **136 epsilon-conditioned scalar tags**,
  including reward, cost, tracking and all physical diagnostics in every bin.
- Original `go1` completed a fresh 16-environment, one-iteration smoke test
  (**1,600 transitions**) with finite losses and checkpoint output.
- A paired evaluation sweep at epsilon **0, 0.5, 1**, 16 environments, fixed
  `(1,0,0)` command, seed 7 and 100-step horizon produced CSV, JSON and PNG.
  Initial-state and checkpoint hashes match across all three runs. The
  accumulator's hand test verifies that post-failure auto-reset samples cannot
  contaminate return, noise, force peaks or survival, and checks budget-tail
  arithmetic independently.
- Syntax compilation and `git diff --check` pass.
- The final code also passed an epsilon **0/1** sweep of the iteration-4
  checkpoint, with **20 environments** (all 20 terrain columns), seed 7,
  fixed command and **200-step** horizon. Initial fingerprints matched and
  summary/plot artifacts were generated in `quietpaw_correction_eval_final`.

Local ignored artifacts:

- `/tmp/quietpaw_correction_tests.log`
- `/tmp/quietpaw_correction_train.log`
- `/tmp/quietpaw_correction_resume.log`
- `/tmp/quietpaw_correction_final.log`
- `/tmp/quietpaw_correction_baseline.log`
- `legged_gym/logs/quietpaw_go1_stage1/Sep29_22-26-37_stage1_correction/`
- `legged_gym/logs/quietpaw_go1_stage1/Sep29_22-29-08_correction_resume/`
- `legged_gym/logs/quietpaw_go1_stage1/Sep29_22-30-52_correction_final/`
- `legged_gym/logs/quietpaw_correction_eval/`
- `legged_gym/logs/quietpaw_correction_eval_final/`

The old passing tests were insufficient: they confirmed the implementation's
old behavior rather than the correct terminal target. The replacement target
test, independent cost-GAE calculation, exact HIM auxiliary replay and explicit
cohort checks now support the corrected implementation. They do not establish
convergence, noise controllability, constraint satisfaction or a robust Pareto
frontier. Training-bin correlations are descriptive, not causal evidence;
use paired evaluation across multiple seeds and commands for that claim.
