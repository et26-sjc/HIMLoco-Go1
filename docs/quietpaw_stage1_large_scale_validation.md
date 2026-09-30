# Stage 1 large-scale validation

This experiment asks whether the current HIMLoco + QuietPaw Stage 1 code
actually produces behavior that depends on the deployment value of
`epsilon`. It is a validation of the existing implementation, not a change to
the algorithm. No reward, robot, PD, terrain, observation, HIM estimator or
domain-randomization code was changed.

The tests ran on 2026-09-30 in the existing `himloco` environment on branch
`quietpaw-reproduction` at commit `c2517327`. Isaac Gym used CUDA PhysX on the
RTX 5070 Ti. The evaluation horizon is 1000 simulator steps (`20 s` at the
recorded `dt=0.02`). The reported noise quantity is the Stage 1 sampled noise
proxy; it is not an acoustic measurement.

## What counts as evidence

The validation separates four claims:

1. **Identifiability:** changing epsilon must not change the initial terrain,
   robot state or randomized parameters. Every epsilon worker therefore runs
   in a fresh process and the launcher rejects a sweep when initial-state
   hashes differ.
2. **Conditional sensitivity:** with the same initial state and command,
   locomotion and physical diagnostics must change when epsilon changes.
3. **Repeatability:** the change should be visible in an independent training
   seed, rather than only in one checkpoint.
4. **CMDP semantics:** a change is not automatically a successful quiet-
   locomotion trade-off. The expected budget direction, tracking, survival and
   failure behavior must be checked separately.

This definition avoids treating a confounded training-bin difference or a
short failed rollout as evidence of a Pareto policy.

## Training runs

All runs used the same command and the existing Stage 1 configuration:

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
WANDB_MODE=disabled python legged_gym/legged_gym/scripts/train_quietpaw.py \
  --task go1_quietpaw_stage1 --headless --num_envs 4096 \
  --max_iterations 100 --seed 4 --run_name large_validation_4096_seed4
```

The seed-5 run differs only in `--seed 5` and its run name. Each run produced
100 PPO iterations, 40,960,000 environment transitions, and a `model_100.pt`
checkpoint. Peak device allocation was approximately 10.1 GiB for seed 4 and
9.9 GiB for seed 5; neither run ran out of memory. The training log reports 20
HIM auxiliary updates per PPO iteration and finite reward/cost losses.

For context, three 160-environment runs (seeds 1, 2 and 3) were also trained
for 100 iterations, 1,600,000 transitions each. Their fixed-command
evaluations failed the first episode early (zero survival-to-horizon success),
so they are reported as an instability diagnostic and are not used as positive
evidence of learned quiet locomotion.

## Assignment and implementation checks at 4096 environments

The saved assignment contains all 4096 unique values of the original global
epsilon grid, exactly 512 labels in each of eight bins. The 20 actual
terrain-by-command cohorts contain 204 or 205 environments each; every cohort
contains all eight epsilon bins and its bin-count spread is at most one. The
mapping is the seeded mapping stored in
`legged_gym/logs/quietpaw_go1_stage1/Sep30_17-44-28_large_validation_4096_seed4/epsilon_assignment.json`.

The actor/critic interfaces remain the Stage 1 interfaces (actor input 65,
reward and cost critic input 239, action output 12). The HIM estimator remains
outside the PPO actor gradient path. The full 22-test suite passes, as do
`compileall` and `git diff --check`.

## Paired evaluation design

The evaluation launcher was run with 4096 environments, seed 11, fixed terrain
and command curriculum, and a fixed command. It starts one fresh Isaac Gym
process per epsilon, records the pre-step terrain/robot/randomization state,
and requires the initial SHA-256 fingerprint to be identical across all
epsilon workers. Only each environment's first episode contributes to the
statistics; simulator auto-resets after failure are ignored.

The full nine-level sweep for seed 4 used:

```text
epsilon = 0, .125, .25, .375, .5, .625, .75, .875, 1
```

The seed-5 confirmation used epsilon `0, .5, 1` under the same
`(vx, vy, yaw_rate)=(1,0,0)` command. Additional seed-4 endpoint sweeps used
commands `(0,0,0)` and `(0.5,0,0)`. The output directories contain one JSON
and log per epsilon, `summary.csv`, `summary.json`, and the reward/noise plot.

## Results: command `(1,0,0)`

Seed 4 produced a strictly monotone response across all nine epsilon levels in
the principal quantities (Spearman `rho=-1` for return, sampled noise cost,
tracking error, survival, and survival duration; `rho=+1` for force and
base-acceleration means in the full sweep). The endpoints are:

| metric | epsilon 0 | epsilon 1 | change (1 - 0) |
| --- | ---: | ---: | ---: |
| locomotion return | 6.8846 | 5.7074 | -1.1772 |
| mean sampled noise proxy | 0.18851 | 0.18485 | -0.00366 |
| velocity tracking error | 0.9355 | 0.9066 | -0.0289 |
| survival-to-horizon rate | 0.6296 | 0.4932 | -0.1365 |
| mean survival steps | 713.5 | 591.1 | -122.4 |
| fz peak mean (N) | 56.66 | 57.70 | +1.05 |
| df3d peak mean | 544.35 | 555.63 | +11.28 |
| base acceleration mean | 1.401 | 1.682 | +0.281 |
| touchdown events/episode | 104.69 | 93.72 | -10.97 |

The paired initial-state hash was identical for all nine workers. This is
strong evidence that the actor uses epsilon to select different closed-loop
behavior under otherwise matched conditions.

Seed 5 reproduces the main direction at the endpoints, although its policy is
less stable under this command:

| metric | epsilon 0 | epsilon 1 | change (1 - 0) |
| --- | ---: | ---: | ---: |
| locomotion return | 1.9930 | 0.9715 | -1.0215 |
| mean sampled noise proxy | 0.17322 | 0.16961 | -0.00361 |
| velocity tracking error | 0.7897 | 0.7142 | -0.0755 |
| survival-to-horizon rate | 0.1057 | 0.0256 | -0.0801 |
| mean survival steps | 210.3 | 103.9 | -106.4 |
| touchdown events/episode | 44.51 | 22.23 | -22.28 |

Both seed-4 and seed-5 workers used the same within-sweep initial-state hash.
The repeated decrease in sampled cost and survival as epsilon grows is a
reproducible conditional response, but it is the opposite of the usual
interpretation that a larger cost budget should permit a noisier, higher-
performance policy. The current implementation therefore demonstrates
conditioning, not yet correct QuietPaw budget semantics.

## Command dependence

The seed-4 checkpoint was also evaluated at epsilon `{0, .5, 1}`:

| command | noise cost at 0 -> 1 | return at 0 -> 1 | survival at 0 -> 1 | fz mean at 0 -> 1 |
| --- | ---: | ---: | ---: | ---: |
| `(0,0,0)` | 0.20559 -> 0.20606 | 28.2106 -> 28.2999 | 0.8721 -> 0.8835 | 46.04 -> 45.91 N |
| `(0.5,0,0)` | 0.20642 -> 0.20589 | 15.8212 -> 15.8047 | 0.7471 -> 0.7393 | 46.99 -> 47.43 N |
| `(1,0,0)` | 0.18851 -> 0.18485 | 6.8846 -> 5.7074 | 0.6296 -> 0.4932 | 56.66 -> 57.70 N |

The response is therefore state/command dependent. It is not a constant
epsilon offset in the logger, and it should not be summarized as one universal
monotone Pareto curve from this checkpoint.

## Artifacts

Training checkpoints:

- `legged_gym/logs/quietpaw_go1_stage1/Sep30_17-44-28_large_validation_4096_seed4/model_100.pt`
- `legged_gym/logs/quietpaw_go1_stage1/Sep30_18-12-25_large_validation_4096_seed5/model_100.pt`

Evaluation directories:

- `legged_gym/logs/quietpaw_large_eval_4096_seed4/` — nine-level `(1,0,0)` sweep.
- `legged_gym/logs/quietpaw_large_eval_4096_seed5_cmd1/` — seed-5 endpoint sweep.
- `legged_gym/logs/quietpaw_large_eval_4096_cmd0/` — seed-4 zero-command sweep.
- `legged_gym/logs/quietpaw_large_eval_4096_cmd05/` — seed-4 medium-command sweep.
- `legged_gym/logs/quietpaw_large_eval_seed1/`, `seed2/`, `seed3/` —
  160-environment cross-seed diagnostics.

The log and checkpoint directories are ignored local artifacts. The exact
training/evaluation stdout is in `/tmp/quietpaw_large_*.log`.

## Conclusion and limits

The implementation passes the causal measurement checks: epsilon assignment
is balanced, each paired sweep has identical initial conditions, and a
4096-environment checkpoint changes its closed-loop locomotion and physical
diagnostics substantially and repeatably when epsilon changes. This supports
the narrow conclusion that the current HIMLoco + QuietPaw actor is
epsilon-conditioned.

It does **not** establish a successful quiet-locomotion Pareto frontier. The
large-command response often trades away survival and return as epsilon grows,
and the sampled noise proxy decreases rather than increasing with the larger
budget. The 160-environment policies also show training instability. Before
claiming a faithful QuietPaw reproduction, train multiple long runs, add a
fixed-epsilon/unconditioned control, inspect lambda and budget calibration,
and test whether the desired cost-performance direction survives commands,
terrain cohorts and seeds. Those are follow-up experiments; no algorithm code
was changed in this validation.
