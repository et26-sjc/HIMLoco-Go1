# QuietPaw reproduction — Stage 0

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
