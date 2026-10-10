# Go1 HIM Baseline Notes

This branch is a cleanup of `prime` for new Go1 locomotion experiments. The goal is to keep the released HIMLoco training algorithm intact while making experiment tracking and sim2sim behavior more reproducible.

## Baseline invariants

The following network, storage and task definitions remain unchanged:

- `rsl_rl/rsl_rl/modules/him_estimator.py`
- `rsl_rl/rsl_rl/modules/him_actor_critic.py`
- `rsl_rl/rsl_rl/storage/him_rollout_storage.py`
- `legged_gym/legged_gym/envs/go1/go1_config.py`

The default PPO/HIM optimization math is retained. Estimation loss and swap loss
are reported as averages over minibatch updates. Quiet V1 adds an optional
estimator-update flag (default: enabled), filters the optimizer to trainable
parameters, and records PPO diagnostics. The base robot adds inactive substep
measurement hooks and a plane-terrain debug fix. Terrain generation can accept
fixed evaluation settings; normal generation follows the original path.

The separate `go1_quiet` task and entrypoints implement the quiet experiment.
They freeze the original actor and estimator. See [QUIET_V1.md](QUIET_V1.md) for
the architecture, validation gates and experiment commands.

## Intentional cleanup changes

- Go1 is the default task.
- W&B is pinned to `wandb==0.15.12`, compatible with the baseline Python 3.7 environment.
- W&B project/entity/group/tags can be configured with environment variables.
- Parameter/gradient histogram duplication and automatic model uploads are disabled by default to reduce training overhead.
- Local checkpoints remain enabled and are the source of truth.
- MuJoCo sim2sim recomputes PD torque at every physics substep, matching the LeggedGym control loop more closely.

## W&B setup

Authenticate once:

```bash
wandb login
```

Choose a W&B project for the new research line:

```bash
export WANDB_PROJECT=himloco-go1-baseline
# Optional:
# export WANDB_ENTITY=<your-team-or-user>
# export WANDB_RUN_GROUP=baseline
# export WANDB_TAGS=go1,him,baseline
```

Then train the Go1 baseline:

```bash
cd legged_gym/legged_gym/scripts
python train.py --task go1 --run_name baseline_seed1 --seed 1
```

Because Go1 is now the default task, `python train.py` is also valid, but explicit `--task go1` is recommended in recorded experiments.

## Baseline dimensions and rates

- actor observation: 45 values per frame
- history: 6 frames, 270 values total
- action dimension: 12
- policy rate: 50 Hz
- simulator step: 0.005 s (200 Hz)
- Go1 control decimation: 4
- action target: default joint angle + 0.25 * policy action
- Go1 PD gains: Kp 40, Kd 1

## Experiment discipline

For new research experiments, keep a frozen baseline commit/tag and branch all method changes from it. Record at least:

- git commit
- seed
- task
- reward/config changes
- observation changes
- policy/HIM architecture changes
- domain randomization changes
- training iterations
- W&B run URL/name

Do not change the baseline and an experimental method in the same commit.
