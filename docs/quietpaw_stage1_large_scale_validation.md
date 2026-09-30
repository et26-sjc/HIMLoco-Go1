# Stage 1 large-scale validation

> **Validity correction (2026-09-30):** The 100-iteration policies have not
> demonstrated basic commanded locomotion. Under a fixed 0.5 m/s command,
> seed 4's tracking error is approximately 0.501 m/s across epsilon values;
> under a 1 m/s command it is 0.935--0.907 m/s. At a zero-speed command it is
> only about 0.036 m/s. A follow-up paired evaluation measured only **0.074
> m/s actual forward velocity** at a 1 m/s command and epsilon 0.5. These
> results are consistent with a mostly stationary or weakly moving policy.
> The epsilon sweeps below establish sensitivity of
> an **immature** policy, not learned epsilon-conditioned locomotion, effective
> noise control, or a Pareto frontier. Gait competence must be established
> before interpreting any noise--performance trade-off.

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

This was only 10,000 policy steps per environment, or 200 simulated seconds.
The repository's HIMLoco paper (Section 3.4) reports that its 4096-environment,
100-step-rollout baseline needs about **1000 rollouts**, with performance still
improving until 2000. Total parallel transitions are not a substitute for
training maturity. In the local original-HIM long run, the tracking reward
component rose from 0.412 at iteration 100 to 0.644 at iteration 300, while
mean terrain level rose from 0.13 to 4.33. The Stage 1 seed-4/5 runs were
still at mean terrain level 0 at iteration 99.

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

### Locomotion competence check (follow-up audit)

The original HIM baseline was evaluated using the user-selected
`model_7500.pt`, not the late checkpoint that suffered training degradation.
Both policies ran the same fixed 1 m/s command, 160 environments, evaluation
seed 11 and 1000-step horizon. Their pre-step terrain, robot and available
randomization state had the same SHA-256 fingerprint
`8d06315cc2c3969876e9fe9ed51403f09d0d9765d9db7021dd956c5453d54153`.
The actual forward velocity was recovered from the privileged observation's
base-linear-velocity component at each **pre-reset** state; it was used only
for evaluation, not passed to either actor.

| Policy | Mean forward velocity (m/s) | Tracking error (m/s) | Reward/live step | 20 s survival |
| --- | ---: | ---: | ---: | ---: |
| Original HIM, iteration 7500 | 0.908 | 0.160 | 0.02355 | 0.831 |
| Stage 1, iteration 100, epsilon 0.5 | 0.074 | 0.932 | 0.00957 | 0.563 |
| Stage 1, iteration 500, epsilon 0.5 | 0.550 | 0.510 | 0.01625 | 0.850 |
| Stage 1, iteration 1000, epsilon 0.5 | 0.576 | 0.473 | 0.01747 | 0.881 |

These checkpoints used different training seeds and update counts. The
500- and 1000-iteration Stage 1 checkpoints resumed from earlier iterations
without restoring the simulator's full state. This is a locomotion-competence
reference, **not** an attribution of the difference to CMDP. It shows that
the 100-iteration policy cannot support a test of quiet commanded locomotion;
by iterations 500 and 1000, it moves forward but still tracks well below the
mature HIM reference. Read-only evaluation logs are
`/tmp/quietpaw_him_baseline7500_fixed1_eval.log`,
`/tmp/quietpaw_stage1_100_fixed1_eval.log` and
`/tmp/quietpaw_stage1_500_fixed1_eval.log` and
`/tmp/quietpaw_stage1_1000_fixed1_eval.log`.

The iteration-1000 checkpoint was additionally evaluated with the existing
epsilon-sweep launcher at epsilon `{0, .05, .1, .15, .2, .25, .5, 1}` on the
same 160-environment, seed-11, fixed-command, 20-second protocol. All workers
had the same initial fingerprint. Tracking error remained around 0.46--0.48
m/s and observed normalized discounted cost around 0.173--0.175; none of
these quantities varied monotonically with epsilon. Epsilon at or below 0.15 was
still over budget, whereas epsilon 0.2 and above was satisfied under this
sampled cost metric. At iteration 1000, 703 of 4096 PID multipliers were
positive, with the highest active epsilon about 0.187. The sweep is an
exploratory training-maturity diagnostic, **not** a claim of successful
constraint control. It still lacks the tracking quality of the selected HIM
baseline and tests only one Stage 1 training seed. Artifacts are
`legged_gym/logs/quietpaw_stage1_maturity_eval_1000_cmd1/` and
`/tmp/quietpaw_stage1_maturity_eval_1000_cmd1.log`.

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

The paired initial-state hash was identical for all nine workers. This shows
that the immature actor's closed-loop behavior is epsilon-sensitive under
otherwise matched initial conditions. It does not establish that the actor
has learned to walk at the commanded speed.

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

The implementation passes the assignment and paired-initialization checks:
epsilon assignment is balanced, and each paired sweep starts from identical
recorded initial conditions. The 4096-environment checkpoint changes its
behavior when epsilon changes, but the fixed-command tracking errors show
that 100 iterations are insufficient to validate **epsilon-conditioned
locomotion**. The supported conclusion is only epsilon sensitivity in an
immature policy.

It does **not** establish a successful quiet-locomotion Pareto frontier. The
large-command response often trades away survival and return as epsilon grows,
and the sampled noise proxy decreases rather than increasing with the larger
budget. Shorter failed episodes also lower total return and discounted cost.
The 160-environment policies show training instability. Before any CMDP or
Pareto claim, first demonstrate commanded forward speed, acceptable tracking,
survival and a locomotion gait against a mature HIM baseline under matched
conditions. Then train and evaluate multiple sufficiently mature Stage 1 seeds,
inspect lambda and budget calibration, and compare commands and terrain
cohorts. No algorithm code was changed in this validation.
