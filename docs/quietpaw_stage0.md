# QuietPaw reproduction — Stage 0

Historical Stage 0 record. The original tasks and checks remain available;
the separate Stage 1 task and its explicit reproduction assumptions are
documented in [quietpaw_stage1.md](quietpaw_stage1.md).

Stage 0 only establishes extension points. It runs the original HIM PPO
objective, one reward critic, and the original HIM auxiliary update. There is
no noise optimization, conditional actor, cost critic, multiplier, PID update,
or value decomposition yet.

## Baseline captured before implementation

- Original branch: `main`.
- Original commit: `c59f76349c4718da19aa9b055a830ac1a383f038`.
- Existing user edits, retained as the reproduction starting point:
  `max_iterations=20000`, `save_interval=100` in the base training config.
- Implementation branch: `quietpaw-reproduction`; `main` is not advanced.
- Environment: existing conda `himloco`, Python 3.8.19,
  PyTorch `2.3.0a0+git63d5e92`, Isaac Gym Preview bindings `gym_38`,
  CUDA PhysX on NVIDIA RTX 5070 Ti.

The environment needs its library directory on `LD_LIBRARY_PATH`; direct
Python invocation without it failed to resolve `libpython3.8.so.1.0`. No
environment was created and no dependencies were installed or changed.

From the repository root:

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
WANDB_MODE=disabled python legged_gym/legged_gym/scripts/train.py \
  --task go1 --headless --num_envs 16 --max_iterations 1 \
  --seed 1 --run_name stage0_before
```

This completed successfully before code changes: 100 steps × 16 environments,
five PPO epochs × four minibatches, finite PPO/HIM losses, and checkpoint save.
The reduced environment count and one iteration are smoke-test overrides only;
terrain, rewards, dynamics randomization, control and simulator settings were
not overridden.

| Interface | Dimension |
| --- | ---: |
| One-frame proprioceptive observation | 45 |
| Six-frame history / exported policy input | 270 |
| Actor MLP input (45 + velocity 3 + latent 16) | 64 |
| Privileged reward critic input | 238 |
| Action / actor output | 12 |

Before-change output: `/tmp/quietpaw_stage0_baseline_before.log`.
Local checkpoints: `legged_gym/logs/rough_go1/Sep29_17-50-00_stage0_before/`.
Logs/checkpoints are local, ignored artifacts, not committed fixtures.

## Frozen source fingerprints

SHA-256 captured before implementation; these files must remain unchanged:

```text
a0cb97016fce9714168f357358dd50d0fb2eaba0017fe37ed5bbc469727024d2  rsl_rl/rsl_rl/modules/him_estimator.py
c516cb3b7bc3b6734182547bacaa6eb6fb0604fac9a0708ff87d869a79c17cc6  rsl_rl/rsl_rl/modules/him_actor_critic.py
1eac94e9fe99766022c7036fcd28a669962b7dfaecff10c8a7972486002bb0f8  rsl_rl/rsl_rl/algorithms/him_ppo.py
971369bb58bdc43e737f317f5ae57443df8f46de463597ee732150c5389bfd9a  rsl_rl/rsl_rl/storage/him_rollout_storage.py
a4d17f05c52d41c1b60583b0155f3452bb48351864e3b5ea0654f517c684190b  legged_gym/legged_gym/envs/base/legged_robot.py
08af584e7b42a3652c1752cca9bd6fde53cc96650ece74cc9b86faaf29825a59  legged_gym/legged_gym/envs/go1/go1_config.py
```

## Structure and Stage 0 data contract

```text
task go1                              task go1_quietpaw
Go1RoughCfg / Go1RoughCfgPPO           QuietPawGo1Cfg / QuietPawGo1CfgPPO
                \                     /
                 original LeggedRobot
                          |
      observations/history 270 + critic observations 238
                          |
HIMOnPolicyRunner                     QuietPawRunner (inherits HIM runner)
HIMPPO                                QuietPawHIMPPO (inherits HIM PPO)
HIMActorCritic                        QuietPawHIMActorCritic (inherits HIM AC)
HIMRolloutStorage                     QuietPawRolloutStorage (inherits HIM storage)

Actor path, identical in both tasks:
history 270 -> original HIMEstimator -> estimated velocity 3 + latent 16
current observation 45 + estimated velocity 3 + latent 16 -> actor 64 -> action 12
action -> original joint target scaling -> original PD -> original Go1 physics

Training paths, identical in both tasks:
privileged observations 238 -> original reward critic -> scalar value
reward + done + stored rollout -> inherited returns/advantages -> inherited PPO
history + next critic observations -> original HIM auxiliary update
```

The QuietPaw actor subclass introduces **no additional parameters**. Estimator,
source/target encoders, prototypes, HIM losses, observation normalization and
history remain inherited. The policy does not receive epsilon in Stage 0; the
actor input is still 64, not 65. There is one reward critic and no cost critic.

| Placeholder | Runner / transition | Rollout buffer | Stage 0 behavior |
| --- | --- | --- | --- |
| `epsilon` | `None` | `[T, N, 1]`, all `NaN` | Unavailable; no conditioning or budget sampling |
| `noise_cost` | `None` | `[T, N, 1]`, all `NaN` | Unavailable; no force collection or cost computation |
| `cost_value` | `None` | `[T, N, 1]`, all `NaN` | Unavailable; no network or value target |

`T` is rollout length and `N` is number of environments. The extra buffers are
excluded from the inherited minibatch generator, returns, advantages and
losses. NaN distinguishes an unavailable measurement from a measured zero;
none of these buffers enters a computation. A non-`None` transition field,
an active epsilon/noise-cost config, or a stage other than zero raises
`NotImplementedError`, rather than silently training with an ignored setting.

The only general integration changes are task registration, class exports,
runner selection in the task registry, and two class-resolution methods in
`HIMOnPolicyRunner`. The original learning, logging and checkpoint loops are
inherited. The environment is the original `LeggedRobot` for both tasks.

## File inventory

Paths below are relative to the repository root.

New files:

| File | Purpose |
| --- | --- |
| `legged_gym/legged_gym/envs/go1/quietpaw_go1_config.py` | Inherit all Go1 settings; add unset placeholders and select QuietPaw classes/log directory |
| `rsl_rl/rsl_rl/modules/quietpaw_him_actor_critic.py` | Actor/critic extension point, currently fully inherited |
| `rsl_rl/rsl_rl/algorithms/quietpaw_him_ppo.py` | Select extended storage/transition; inherit all optimization |
| `rsl_rl/rsl_rl/storage/quietpaw_him_rollout_storage.py` | Reserve scalar side-channel buffers and reject active use |
| `rsl_rl/rsl_rl/runners/quietpaw_runner.py` | Validate Stage 0 settings and resolve subclass selection |
| `tests/test_quietpaw_stage0.py` | Behavioral, configuration, checkpoint and frozen-source regression checks |
| `docs/quietpaw_stage0.md` | Baseline record, interface specification and validation instructions |

Modified existing files:

| File | Change |
| --- | --- |
| `legged_gym/legged_gym/envs/__init__.py` | Register `go1_quietpaw` |
| `legged_gym/legged_gym/utils/task_registry.py` | Select original or QuietPaw runner from an explicit registry |
| `rsl_rl/rsl_rl/runners/him_on_policy_runner.py` | Add policy/algorithm class-resolution extension methods |
| `rsl_rl/rsl_rl/modules/__init__.py` | Export new policy class |
| `rsl_rl/rsl_rl/algorithms/__init__.py` | Export new algorithm class |
| `rsl_rl/rsl_rl/storage/__init__.py` | Export new storage class |
| `rsl_rl/rsl_rl/runners/__init__.py` | Export new runner class |
| `legged_gym/legged_gym/envs/base/legged_robot_config.py` | Preserve the two user edits already present before Stage 0; no additional edits |

## Running and checking

Use the existing `himloco` activation and `LD_LIBRARY_PATH` setup shown above.
The following commands run one real training iteration, including PPO/HIM
updates and checkpoint writes:

```bash
# Original HIM task on this branch
WANDB_MODE=disabled python legged_gym/legged_gym/scripts/train.py \
  --task go1 --headless --num_envs 16 --max_iterations 1 \
  --seed 1 --run_name stage0_after

# QuietPaw scaffold; still the original HIM optimization
WANDB_MODE=disabled python legged_gym/legged_gym/scripts/train.py \
  --task go1_quietpaw --headless --num_envs 16 --max_iterations 1 \
  --seed 1 --run_name stage0_scaffold

# Interface and learning regression checks
python -m unittest discover -s tests -p test_quietpaw_stage0.py -v

# Syntax checks
python -m compileall -q rsl_rl/rsl_rl legged_gym/legged_gym tests
git diff --check
```

For a full configured training run, omit `--num_envs 16 --max_iterations 1`.
The tasks write under `legged_gym/logs/rough_go1` and
`legged_gym/logs/quietpaw_go1_stage0`, respectively. `WANDB_MODE=disabled`
keeps these checks local; it does not change the optimization.

## Validation scope

Seven regression tests pass in `himloco`:

1. The six frozen source files have exactly the pre-change SHA-256 hashes.
2. Environment configs, PPO hyperparameters and policy configs match; derived
   config instances do not mutate the original task's settings.
3. Identical model state dictionaries and deterministic actions; actor 64,
   reward critic 238, action 12; actor gradients do not enter HIMEstimator.
4. Placeholder shapes and unavailable values are correct; populated fields
   fail before a transition can be stored.
5. On the same seeded synthetic rollout (including termination and timeout),
   returns, advantages, losses, and **all updated model parameters** match
   exactly between HIM and QuietPaw. The HIM estimator is updated in both.
6. Task registry selects the intended subclasses and the baseline checkpoint
   loads into the scaffold. The original learning method remains inherited.
7. Active Stage 1 settings fail explicitly.

CPU synthetic-rollout equality isolates the learning implementation from
simulator variability. The real CUDA PhysX smoke runs establish pipeline
compatibility; they do not establish bitwise trajectory reproducibility,
convergence, or quiet-locomotion performance.

All three real training runs completed successfully on 2026-09-29, each with
16 environments, 100 rollout steps, 1 iteration, 20 PPO minibatch updates and
`model_1.pt` saved. All reported losses were finite:

| Run | Value loss | Surrogate loss | HIM estimation loss | HIM swap loss |
| --- | ---: | ---: | ---: | ---: |
| Original, before changes | 0.4764 | 0.0060 | 0.5708 | 0.1082 |
| Original, after changes | 0.4801 | -0.0038 | 0.5680 | 0.1082 |
| QuietPaw Stage 0 | 0.4794 | -0.0046 | 0.5338 | 0.1082 |

The real runs are not numerically identical despite using the same seed;
they are smoke checks, not a controlled training-performance comparison.
Learning equivalence is established by the identical-input regression test.

After-change logs are `/tmp/quietpaw_stage0_baseline_after.log` and
`/tmp/quietpaw_stage0_scaffold.log`; regression test output is
`/tmp/quietpaw_stage0_contracts.log`. These temporary logs may be removed by
the operating system. Saved training checkpoints are in:

- `legged_gym/logs/rough_go1/Sep29_17-57-11_stage0_after/`
- `legged_gym/logs/quietpaw_go1_stage0/Sep29_17-57-42_stage0_scaffold/`

Import checks resolve all four new exported classes and the `go1_quietpaw`
task. Compilation and `git diff --check` pass. All three final checkpoints
were loaded on CPU and checked: every model tensor is finite, the iteration
is 1, and actor/critic/action weight dimensions are 64/238/12 as expected.

## Stage 1 handoff — not implemented

Keep the frozen HIM source files and inherited Go1 locomotion settings intact.
The extension points are:

| Location | Stage 1 work |
| --- | --- |
| QuietPaw Go1 config | Define paper-grounded epsilon sampling, noise-cost settings and CMDP hyperparameters |
| `QuietPawHIMActorCritic` | Explicit epsilon side input (actor 64 to 65), conditioned reward/cost critics; no epsilon in history or estimator |
| `QuietPawRolloutStorage` | Store actual epsilon/cost/cost-value data, cost returns/advantages, aligned minibatches and correct terminal/timeout data |
| `QuietPawHIMPPO` | Dual-value losses, reward/cost advantages and paper-matched constrained PPO/multiplier updates; retain HIM auxiliary training |
| `QuietPawRunner` | Maintain epsilon across collection/reset/update, pass cost data, log constraint diagnostics and persist added training state |
| Future QuietPaw environment subclass | Measure paper-defined noise cost at the correct simulation timing; preserve the original reward and robot/terrain/control configuration |
| Regression and integration tests | Verify cost timing, transition alignment, timeout handling, privileged-information boundaries and estimator gradient isolation |

Before enabling a cost, resolve its formula, sign, units, normalization,
contact mask, sampling timing, episode/discount aggregation and budget meaning
from QuietPaw's paper/implementation, and record any reproduction assumptions.
Stage 0 deliberately supplies no defaults for those research choices.
Multiplier update frequency, PID use and sharing across epsilon values also
remain undecided here. Stage 2's value decomposition must live in the QuietPaw
critic extension; the HIM latent is not a successor-feature representation.

The inherited checkpoint format works unchanged at Stage 0. Stage 1 will need
explicit persistence/migration for any new parameters, cost critic,
optimizers, multipliers and condition-sampling state; Stage 0 checkpoint
compatibility is not a claim that a future conditioned model is identical.
