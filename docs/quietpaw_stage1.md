# QuietPaw on HIMLoco — Stage 1

**Review corrections C1–C3 are implemented.** See
[quietpaw_stage1_corrections.md](quietpaw_stage1_corrections.md) for the current
training/evaluation commands, cohort assignment and validation. Checkpoint
contract is now v2; pre-correction v1 training checkpoints cannot resume as
corrected runs. The validation section below is a historical pre-review record,
not proof that the original timeout handling or epsilon assignment was correct.

Stage 1 implements a conditional actor, asymmetric reward/cost critics,
separate cost GAE, and PPO-Lagrangian with per-environment PID multipliers.
It does **not** implement the Stage 2 value decomposition. This is a working
reproduction under the explicit assumptions below, not a claim of numerical
equivalence to unavailable author code or successful quiet locomotion.

The `go1` baseline and `go1_quietpaw` Stage 0 tasks remain available. Stage 1
has its own `go1_quietpaw_stage1` task. The six frozen source hashes in
[Stage 0](quietpaw_stage0.md) still match, including HIMEstimator, HIMActorCritic,
HIMPPO, HIMRolloutStorage, LeggedRobot and the original Go1 configuration.

## Source audit and reproduction assumptions

Audit date: 2026-09-29. The local `QuietPaw.pdf` is arXiv 2503.05035v1.
Checked the [paper](https://arxiv.org/html/2503.05035v1),
[arXiv source archive](https://arxiv.org/src/2503.05035),
[first author's homepage](https://zhangyuyou-10.github.io/),
[first author's repositories](https://github.com/ZhangYuyou-10?tab=repositories),
[Shiqi Liu](https://shiqiliu-67.github.io/),
[Changyi Lin](https://linchangyi1.github.io/),
[Yaru Niu](https://yaruniu.com/publications/), and
[Safe AI Lab](https://safeai-lab.github.io/).
GitHub repository/API searches and GitLab/project/supplement searches did not
locate a verifiable QuietPaw implementation. Coauthor publication entries link
the paper without a code link. The source archive contains the paper's TeX
and figures, with no separate supplementary implementation. The IEEE landing
page was not sufficiently accessible to verify any associated supplementary
files. This does not prove that no such files exist elsewhere.
The archive's `data.txt` contains numeric experiment summaries, not a cost
normalization formula, PID settings, or a training implementation.

Crucially, `2_method.tex` confirms the same decreasing exponentials as the PDF;
the sign inconsistency is **not** a PDF text-extraction error. Eq. (1) decreases
with increasing force and squared velocity, while the text and Figure 2
describe an increasing noise cost and Eq. (2) minimizes that cost. Neither
the exact normalization nor the missing contact/timing details could be
verified. The user's instruction allows a carefully documented physical
interpretation if the authors made an error.

| Item | Paper evidence | Implemented choice / status |
| --- | --- | --- |
| Force and velocity scales | III-A: sigma_F=50, sigma_v=0.5; weights 0.5/0.5 | Retained; force in N, squared velocity in m²/s² |
| Cost direction | Eq. (1) decreases with impact; prose/Fig. 2 says increasing | Explicit monotone complement, named `normalized_complement_v1`; unverified correction |
| Normalization | IV-A says [0,1], no formula | Componentwise complement and fixed four-foot average; no running normalization |
| F_max | Appears without a numerical setting | Cancels when force score is normalized by its zero-force value; no invented threshold |
| Contact mask | Not specified | Force-norm >1 N; force term on contact, velocity term only at detected touchdown |
| Impact velocity | Not specified beyond impact velocity | Previous policy sample's downward world-z foot speed; an approximation to pre-impact normal speed |
| Timing | Not specified | Existing post-physics/pre-reset boundary, 50 Hz; physics/PD unchanged |
| Epsilon | III-D: uniform interpolation on [0,1], one per environment | Complete `linspace(0,1,N)` grid stratified/shuffled across joint terrain/command cohorts; fixed across rollouts and resets |
| Budget units | Eq. (3) uses discounted sum; experiments discuss normalized cost | Normalized discounted return, `(1-gamma)*sum(gamma^t*c_t)`; explicit unit choice, not verified author behavior |
| Cost discount/GAE | Not reported | Inherit HIM gamma and GAE lambda |
| Multipliers | III-D: independent lambda per environment; PID [53] | Independent PID state per environment, not one global scalar |
| PID gains/smoothing | Not reported | Kp=1, Ki=0.1, Kd=0.1, EMA=0.95; nonnegative integral/output, one-observation derivative delay, initial lambda=0 |
| PID measurements | Episode/rollout aggregation not specified | Mean completed-episode normalized discounted cost per environment; skip environments without fresh completions |
| PID frequency/order | Algorithm 1 computes advantage before multiplier update | Once per rollout/update cycle, old lambda fixes this cycle's advantages; new lambda used next cycle |
| Advantage normalization | Not specified | Raw reward/cost GAE combined first using Eq. (8), then one common batch standardization |
| Value loss | Eq. (7) squared error | Retain HIM value clipping convention for both independent critics; cost coefficient=1 |
| Stage 1 critics | Full method uses decomposition | Direct concatenation MLPs for this intermediate stage; no Stage 2 representation claimed |
| Initialization | Not specified | Existing actor/reward weights retained, new epsilon columns zero; separate cost critic with zero output layer |

The PID structure follows the paper's cited
[Stooke et al. method](https://proceedings.mlr.press/v119/stooke20a.html) and
[author implementation](https://github.com/astooke/rlpyt/blob/master/rlpyt/projects/safe/cppo_pid.py).
The gains above are local assumptions, not recovered QuietPaw hyperparameters.

## Cost definition and sampling

For foot i, define `F_i = ||contact_force_i||`, contact `m_i = (F_i > 1 N)`,
touchdown `d_i = m_i & ~previous_contact_i`, and
`v_i = max(-previous_foot_velocity_z_i, 0)`:

```text
c_t = (1/4) sum_i [0.5 * m_i * (1 - exp(-F_i / 50))
                 +0.5 * d_i * (1 - exp(-v_i² / 0.5))]
```

The force complement can be derived as
`1 - exp((F_max-F_i)/sigma_F) / exp(F_max/sigma_F)`.
This normalizes each component before complementing; it is not asserted to
be the authors' unspecified normalization of the full expression. It retains
the reported exponential scales and makes both components nondecreasing in
their corresponding physical inputs. The proxy is bounded in [0,1], is zero
without contact, and includes no arbitrary dB interpretation or measured audio.

The denominator is always four, rather than the number of contacting feet.
`noise_cost` remains separate from `rew_buf`; locomotion reward is never
replaced by a fixed weighted noise penalty. `QuietPawLeggedRobot.compute_reward`
reads the refreshed force/foot tensors, saves cost and diagnostics, then calls
the original reward computation. Reset clears only the measurement history;
it cannot erase the terminal transition's saved cost. Touchdown, force-rate
and acceleration history are invalidated at reset to suppress cross-episode
differences. Force cost is still measured on the first sample after reset.

This is a policy-rate measurement, not a true 200-Hz impact sampler. A contact
starting and ending between samples can be missed. The preceding 20-ms foot
velocity is an approximation to pre-impact velocity, and world-z is not the
local normal on sloped terrain. PhysX contact collection settings remain the
baseline settings. These limitations must accompany any later noise result.

## Budget, cost GAE and multiplier units

Use `c_bar_t=(1-gamma)*c_t` for the cost Bellman target. Cost critic Vc,
cost advantage Ac, PID measured return and epsilon all use those units:

```text
Vc(s, epsilon) = E[sum_t gamma^t * c_bar_t]
delta_c = c_bar_t + gamma * Vc(s_next, epsilon) - Vc(s, epsilon)
Ac_t = delta_c_t + gamma * gae_lambda * Ac_next
A_combined = (Ar_raw - lambda_i * Ac_raw) / (1 + lambda_i)
```

True terminal states have no bootstrap or GAE continuation. A time limit
adds `gamma*Vc(terminal_state, epsilon)` into the last cost target, then cuts
the GAE recursion at `done`. A rollout boundary bootstraps the last current
critic observation normally. The inherited runner already restores terminal
critic observations before handing them to PPO. Both reward and cost timeout
targets now use that terminal observation and the transition epsilon, never
the reset observation or current-state value. Original HIMPPO is unchanged.

A real simulator check exposed stale `infos['time_outs']` in the original
environment: it is only updated inside nonempty `reset_idx`, while the current
timeout tensor is reassigned every step. The QuietPaw subclass publishes fresh
flags every step. The original environment is untouched. This also prevents
stale reward-side timeout corrections in the Stage 1 task.

For each completed episode of length L, the PID measurement is
`sum_{t<L} gamma^t*c_bar_t`, plus `gamma^L*Vc(terminal)` on a timeout. A physical
failure truncates with no tail value, including a failure coinciding with a
time limit. Stage 1 reads the physical termination-contact signal separately
without changing reset decisions. Episodes spanning rollout boundaries retain
their accumulators.

Each environment gets at most one PID update per PPO iteration using its mean
over fresh completed episodes. Empty environments retain all PID state; old
episode costs are never reused as new samples. PID state is fixed throughout
PPO minibatches. The PID uses positive `measured_return - epsilon` to increase
constraint enforcement. Its state is outside autograd and outside Adam.

Reward advantage is recovered from unnormalized `returns-values`; cost
advantage is not independently centered or rescaled. Joint standardization
is applied only after Eq. (8). The PPO ratio and clipping then act on this
joint advantage. A single aligned permutation carries epsilon, targets,
observations and old action distributions into every minibatch. The equal T
samples per environment make the full batch mean an equal-environment mean.

## Architecture and HIM auxiliary preservation

```text
original o_t (45) + original six-frame history (270)
                         |
                  original HIMEstimator
                    v_hat (3), z (16), detached
                         |
                 [o_t, v_hat, z, epsilon] (65)
                         |
                 conditional actor -> actions (12) -> original PD/Go1

original privileged critic observation (238) + epsilon (1)
                         |                         |
                  reward critic (1)         cost critic (1)

reward / cost / conditions / old values / dones / old log-probabilities
                         -> aligned CMDP rollout storage
                         -> independent GAE -> combined advantage -> PPO

unchanged history (270) + next critic observation (238)
                         -> original HIM auxiliary optimizer, EVERY minibatch
```

Epsilon is explicit actor/critic side input. It never enters history,
normalization, estimator, source/target encoders, prototypes or HIM targets.
No contact force or true velocity enters the actor through the new interface.
Both critics have independent 239→512→256→128→1 ELU MLPs; the actor is
65→512→256→128→12 with the original learned Gaussian action standard deviation.

`QuietPawConstrainedPPO.update` retains the original order: conditional actor
and critic forward passes; adaptive KL learning-rate adjustment; exactly one
`estimator.update(obs_270, next_critic_238, lr=current_lr)`; combined PPO/value/
entropy loss; optimizer zero_grad; backward; gradient clipping; optimizer step.
The HIM estimator still performs its own backward/Adam step with unchanged
velocity and prototype-swap losses. PPO has no gradient path into it. A runtime
check rejects any estimator gradient after PPO backward. Tests also replay the
same auxiliary minibatches through an independent original HIMEstimator and
require exactly matching encoder, target and prototype parameters.

Trainable components are actor MLP, Gaussian std, reward critic, cost critic,
and the original HIM encoder/target/prototype parameters. PID variables are
controller state. No successor features or constraint-weight network exists.

## Files and entry points

New components inherit Stage 0 classes, which remain usable:

| File | Role |
| --- | --- |
| `legged_gym/legged_gym/envs/go1/quietpaw_noise.py` | Pure tensor noise proxy |
| `legged_gym/legged_gym/envs/go1/quietpaw_legged_robot.py` | Pre-reset measurement, reset hygiene, fresh timeout flags |
| `rsl_rl/rsl_rl/modules/quietpaw_conditional_actor_critic.py` | Conditioned actor and dual critics |
| `rsl_rl/rsl_rl/storage/quietpaw_cmdp_storage.py` | Cost GAE, raw/combined advantages, aligned batches |
| `rsl_rl/rsl_rl/algorithms/quietpaw_pid.py` | Per-environment PID state |
| `rsl_rl/rsl_rl/algorithms/quietpaw_constrained_ppo.py` | CMDP collection, update, HIM preservation |
| `rsl_rl/rsl_rl/runners/quietpaw_stage1_runner.py` | Conditional logging/inference and checkpoint contract |
| `legged_gym/legged_gym/scripts/train_quietpaw.py` | Training CLI that honors resume |
| `tests/test_quietpaw_stage1.py` | Mathematical, information-flow and checkpoint tests |

Existing edits: add Stage 1 configs to `quietpaw_go1_config.py`, register the
task in `envs/__init__.py`, select its runner in `utils/task_registry.py`, and
add logging-value/checkpoint-dictionary extension methods in
`him_on_policy_runner.py`. Its collection/learning loop remains inherited.

For current commands, including the >=160-environment balanced smoke test,
use the correction document. The original `train.py` forcibly sets `args.resume=False`; it can start a fresh
Stage 1 run but cannot honor `--resume`. Use `train_quietpaw.py` for Stage 1.
The original training script remains unchanged.

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Balanced real training run. Omit overrides for full training.
WANDB_MODE=disabled python legged_gym/legged_gym/scripts/train_quietpaw.py \
  --task go1_quietpaw_stage1 --headless --num_envs 160 --max_iterations 3 \
  --seed 1 --run_name stage1_smoke

# Resume: use the SAME environment count and a corrected v2 checkpoint.
WANDB_MODE=disabled python legged_gym/legged_gym/scripts/train_quietpaw.py \
  --task go1_quietpaw_stage1 --headless --num_envs 160 --max_iterations 1 \
  --seed 1 --resume --load_run YOUR_CORRECTED_RUN_DIRECTORY \
  --checkpoint 3 --run_name stage1_resume

python -m unittest discover -s tests -p 'test_quietpaw_stage*.py' -v
python -m compileall -q rsl_rl/rsl_rl legged_gym/legged_gym tests
git diff --check
```

Checkpoints save actor/critics/HIM, PPO optimizer, HIM optimizer, iteration,
adaptive learning rate, epsilon grid, all PID state and the cost/budget/
algorithm contract. Training resume rejects a different epsilon grid or
changed contract. As HIM does not save simulator state, fresh environments
start fresh partial-episode cost accumulators on resume. This is a training
resume, not bitwise simulator continuation. Stage 0 checkpoints are rejected
with an explicit migration message rather than silently partially loaded.

Conditional inference is explicit:

```python
policy = runner.get_inference_policy(device=env.device)
epsilon = torch.full((env.num_envs, 1), 0.2, device=env.device)
with torch.no_grad():
    actions = policy(history_270, epsilon)
```

The legacy `play.py`/JIT exporter supplies only the 270-D history and is not
the Stage 1 deployment interface. The conditional evaluation launcher is now
`scripts/evaluate_quietpaw.py`; conditioned JIT export remains separate work.

## Diagnostics and scientific limits

TensorBoard/W&B record cost-value loss, raw mean noise cost, old/new lambda,
PID update counts, completed episodes, observed budget violation and HIM
minibatch update count. Raw mean noise cost and discounted budget violation
have different aggregation semantics; do not compare the former directly to
epsilon and call it a constraint violation.

`infos['quietpaw']` exposes noise cost plus fz_peak, f3d_peak, df3d_peak,
base_acceleration, touchdown_events, velocity_tracking_error, tilt and failure.
The peak quantities are maxima over feet at policy samples; df3d is
`||F_t-F_previous||/dt` in N/s, base acceleration is a world-velocity finite
difference in m/s². These diagnostics do not enter optimization. A later
evaluation should aggregate temporal peaks/events and episode success, with
identical commands, terrains, seeds and horizons across epsilon settings.

Risks to retain in research reports:

- The repaired proxy, budget units, sampling mask/timing and PID gains are
  assumptions; paper-level numerical reproduction is still unresolved.
- Saturating exponentials reduce sensitivity at very large impacts. A
  policy-rate proxy misses short collisions and is not a calibrated SPL model.
- Early failures truncate cost returns, potentially looking cheap. Preserve
  the locomotion reward, but always report survival/success alongside cost.
- Epsilon=0 may be infeasible; neither PID nor clipping guarantees constraint
  satisfaction. PID gain scale depends on cost-return units and reward scale.
- Per-environment completed episodes can be sparse/noisy, causing infrequent
  multiplier updates; short smoke tests cannot tune this behavior.
- Cohort-stratified assignment removes the monotone epsilon/environment-ID
  coupling. It cannot guarantee zero finite-sample correlation or balance of
  learned future terrain difficulty. Evaluate on matched terrain sets/seeds.
- Stage 1 direct-conditioned critics are the intermediate reproduction; no
  full CNCP/value-decomposition or quiet-performance result is claimed.

## Completed validation

All checks used the existing conda `himloco` environment. No dependencies or
environments were installed. On 2026-09-29:

| Check | Result |
| --- | --- |
| Original Stage 0 regression suite | 7/7 pass, including all six frozen source hashes |
| Stage 1 regression suite | 11/11 pass |
| Combined suite | 18/18 pass |
| Syntax/import and whitespace checks | Pass |
| Real Stage 1 training | 3 iterations ×100 steps ×16 environments =4,800 transitions, finite losses |
| Real Stage 1 resume | Loaded iteration 3/PID state; completed iteration 4 and saved `model_4.pt` |
| Original Go1 after Stage 1 integration | 1 iteration, 1,600 transitions, finite losses, checkpoint saved |

Stage 1 tests cover unchanged environment configuration; conditional actor/
critic dimensions; untouched HIM initialization and detached gradients; cost
monotonicity/bounds/contact/touchdown/reset masking; pre-reset cost preservation
and fresh timeout metadata; hand-computed GAE and lambda broadcasting; terminal
versus timeout bootstrap; PID direction, independence, missing-data handling
and checkpoint state; shuffled condition/target alignment; policy gradient sign
and PPO clipping; exact auxiliary replay equivalence; dual-critic updates;
resume and incompatible-contract rejection.

The three-iteration run's raw mean costs were 0.139526, 0.144333 and 0.134786;
cost-value losses were 0.0001476, 0.0001881 and 0.0001994. Each iteration recorded
20 HIM updates. The maximum lambda after iteration 3 was 0.00666960; after one
resumed iteration it was 0.01834681. These are startup diagnostics, not evidence
of meaningful noise control or improved locomotion.

Both Stage 1 checkpoints were loaded on CPU: all model/PID tensors finite,
actor input 65, both critic inputs 239, actor output 12. Every populated PPO
Adam state and HIM Adam state had exactly 60 steps after three iterations and
80 after the resumed fourth iteration. The PID update count summed over
environments continued from 19 to 25, rather than restarting.

Local artifacts (ignored, not portable fixtures):

- Tests: `/tmp/quietpaw_stage1_contracts.log`.
- Stage 1 training: `/tmp/quietpaw_stage1_smoke.log`;
  `legged_gym/logs/quietpaw_go1_stage1/Sep29_18-22-58_stage1_smoke_fixed_timeout/model_3.pt`.
- Verified resume: `/tmp/quietpaw_stage1_resume_verified.log`;
  `legged_gym/logs/quietpaw_go1_stage1/Sep29_18-24-54_stage1_resume_verified/model_4.pt`.
- Baseline: `/tmp/quietpaw_stage1_baseline_smoke.log`;
  `legged_gym/logs/rough_go1/Sep29_18-28-43_stage1_baseline_regression/model_1.pt`.

Two integration issues were observed rather than hidden: the first real
Stage 1 run rejected stale timeout flags (fixed in the QuietPaw subclass),
and an initial legacy `train.py --resume` attempt actually started iteration 0
because that script discards the flag. The dedicated Stage 1 entry point was
then used to verify actual iteration-3 loading and iteration-4 saving. The
failed/misrouted attempts are not counted as successful resume validation.
