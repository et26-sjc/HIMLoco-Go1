# Direct epsilon budgets and velocity-weighted cost

This experiment implements two requested changes to QuietPaw Stage 1 on the
HIMLoco backbone. It does not implement Stage 2 value decomposition.

## Definition

The actor and both critics receive the actual `epsilon` cost budget. The
training grid is sampled directly over `[0.2,1.0]`, retaining the seeded
terrain/command cohort stratification. There is no preference-to-budget map.
The normalized discounted cost constraint is

```text
epsilon_grid = linspace(epsilon_min, epsilon_max, num_envs)
J_cost <= epsilon
defaults: epsilon_min = 0.2, epsilon_max = 1.0
```

Actor, critics, storage and PID all use the same epsilon value. PID compares
measured discounted cost with that value. History, network dimensions,
reward/cost GAE, combined advantage, timeout handling, PID equations and HIM
auxiliary training are retained. The assignment changes only its grid
endpoints. Logs report `cost_budget` as an explicit alias for epsilon.

The cost function never receives epsilon and is not multiplied or divided
by epsilon. A different condition may change actions and therefore the
physical trajectory and measured cost; it does not change the cost formula.
The cost critic is conditioned on epsilon to predict the return of the
corresponding conditional policy, not to define the physical cost.

The existing complementary physical cost keeps its masks and 50 Hz timing:

```text
c_t = mean_feet[0.2 * contact * (1 - exp(-force_norm / 50))
             + 0.8 * touchdown * (1 - exp(-downward_speed_squared / 0.5))]
```

The weights sum to one, so the cost remains bounded by `[0,1]`. Both weighted
contributions are logged separately as `noise_force_cost` and
`noise_impact_cost`; evaluation also records their observed discounted sums.
The force term still includes support force. Reducing its coefficient does
not remove this effect, and 0.8/0.2 coefficients do not imply 80%/20% of the
time-averaged cost. Contact events and velocity remain approximations at the
policy sampling rate. This is a physical proxy, not measured acoustic noise.

The epsilon interval excludes zero and very small absolute budgets as
requested. Changing that interval and changing the physical cost weights
are independent experiment settings. The weights are experimental starting
values, not recovered author settings. The nominal Go1 four-foot static
support estimate drops from about `0.213` to `0.0852`. The force term has NOT
been changed to an excess-over-reference-force penalty; that is a separate
cost definition requiring its own reference-force choice.

## Checkpoints

Contract v3 saves the weights and the direct epsilon range. Old v2 checkpoints cannot
resume under the changed cost definition/grid. Evaluation supports both versions
and reconstructs each checkpoint's saved cost. Do not compare their raw costs
as if they used the same definition. For this experiment start from original
HIM weights, or resume a v3 checkpoint with identical configuration.

## Small training experiment

Use the existing original HIM `model_7500.pt` as a weight-only initialization.
The saved initialization provides a mature locomotion control under the NEW
cost definition. All PPO, HIM auxiliary optimizer and PID states start fresh.
Do not interpret a short run from random initialization as a locomotion test.

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
WANDB_MODE=disabled python legged_gym/legged_gym/scripts/train_quietpaw.py \
  --task go1_quietpaw_stage1 --headless --num_envs 1024 \
  --max_iterations 200 --seed 4 --run_name budget_velocity_seed4 \
  --him_checkpoint legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt \
  --epsilon_min 0.2 --epsilon_max 1.0 --noise_force_weight 0.2
```

This is 20,480,000 additional transitions. Check initialization, iteration
100, and iteration 200 under the same evaluator configuration. Replace
`RUN_DIRECTORY` with the actual printed training directory:

```bash
python legged_gym/legged_gym/scripts/evaluate_quietpaw.py \
  --checkpoint RUN_DIRECTORY/model_200.pt \
  --output legged_gym/logs/budget_velocity_eval200_seed11 \
  --epsilons 0.2 0.4 0.6 0.8 1 --num-envs 512 --steps 1000 --seed 11 --command 1 0 0
```

Repeat this command for `model_him_init.pt`, evaluation seed 12, and command
`0.5 0 0`. GPU PhysX workers need not reproduce identical trajectories even
with identical initial hashes; initialization has zero epsilon actor weights
and serves as a null control for this variation.

## Interpretation

Start with budget calibration: inspect completed discounted costs and lambda
per epsilon bin. If all multipliers stay zero, the interval is too loose for
this policy/distribution; if tight-bin multipliers grow while survival or
tracking collapses, investigate the strict endpoint and cost contributions.
In particular, increasing the velocity weight can lower overall cost below
0.2. The requested range avoids impossible zero budgets but does not ensure
active constraints or an epsilon-conditioned trade-off. Report that outcome
if it occurs. Do not automatically remap epsilon to force a response.
Use `--epsilon_min` and `--epsilon_max` to configure a NEW experiment if the
range is later revised; do not silently change it mid-run or reuse
incompatible optimizer/PID state.

An encouraging pilot result has lower observed discounted cost at the strict
end, maintained commanded motion, and a repeatable response larger than the
initialization null-control variation. Report both cost contributions,
tracking, per-live-step reward, survival and contact frequency. Fewer total
contacts after earlier failure do not establish quieter gait. A pilot is
allowed to show no detectable trade-off; 200 updates do not guarantee one.
Use subsequent independent training seeds before calling the result robust.

No training run or behavioral verification was performed as part of this
implementation. Readiness here means the new configuration and data flow
are implemented; empirical feasibility remains the pilot's first question.
