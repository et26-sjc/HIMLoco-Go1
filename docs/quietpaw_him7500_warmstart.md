# HIM 7500 -> QuietPaw Stage 1 exploratory training

This is a **weight-only warm-start experiment**, not QuietPaw's reported
two-stage reward curriculum. The user selected original HIM `model_7500.pt`
because later baseline training degraded. The original Go1 locomotion reward,
observation, HIM estimator architecture, robot, PD, terrain and domain
randomization remain unchanged. QuietPaw's paper instead describes a sigmoid
decay of gait-related reward around its stated "2k step" in a 10k-iteration
run; it does not specify a 1000-iteration cost-off switch. This experiment
does not claim to reproduce that schedule.

The original HIM weights are transferred into Stage 1 as follows:

| Stage 1 parameter | Initialization |
| --- | --- |
| HIM estimator, actor layers after the first, actor bias and action std | Exact original HIM tensors |
| Actor first layer | Original 64 columns copied; new epsilon column zero |
| Reward critic first layer | Original 238 columns copied; new epsilon column zero |
| Other reward critic layers | Exact original HIM tensors |
| Cost critic | Copy transferred reward-critic trunk; zero scalar output layer |
| PPO optimizer, HIM auxiliary optimizer, PID state | Fresh Stage 1 state |

Thus at the initialization boundary, the deterministic actor action and
reward value equal the original HIM model for any epsilon; the cost value is
zero. The source file's SHA-256 is recorded in `him_warmstart.json` and each
Stage 1 checkpoint. The source `model_7500.pt` has an internal `iter=0` field
because the inherited HIM runner saved intermediate checkpoints before it
advanced `current_learning_iteration`; use the filename and hash for identity,
not that field as evidence of its training duration.

Run from the repository root with the existing `himloco` environment:

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
WANDB_MODE=disabled python legged_gym/legged_gym/scripts/train_quietpaw.py \
  --task go1_quietpaw_stage1 --headless --num_envs 4096 \
  --max_iterations 1000 --seed 4 --run_name him7500_cmdp_seed4 \
  --him_checkpoint legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt
```

`--him_checkpoint` cannot be combined with `--resume`. The run writes a
`model_him_init.pt` before its first Stage 1 update and then normal
`model_*.pt` checkpoints. Resume a Stage 1 checkpoint with the existing
`--resume --load_run ... --checkpoint ...` options and **without**
`--him_checkpoint`. Simulator/RNG state is not restored by Stage 1 resume.

Evaluation protocol: first verify the transferred initialization against the
HIM model on identical observations. Then compare initialization, iteration
100/500/1000, and the original HIM `model_7500.pt` at fixed commands, terrain
distribution, seeds and 20-second first episodes. Report actual forward
velocity, tracking error, survival, observed normalized discounted cost,
per-live-step reward, force/impact diagnostics and per-epsilon PID state.
Budget decisions use observed discounted cost, not raw per-step noise.

The initialization has an exact zero epsilon column, yet separate GPU PhysX
workers with the same initial-state and checkpoint hashes produced tracking
errors of approximately 0.158, 0.153 and 0.165 m/s at epsilon 0, 0.5 and 1.
Therefore a matched initial hash alone is insufficient evidence that a
closed-loop epsilon difference was caused by conditioning. Treat this as a
null-control variation; also measure action differences on **identical stored
observation histories** and repeat rollouts across evaluation seeds. One
warm-start training seed is an exploratory result, not a Pareto or constraint
validation.
