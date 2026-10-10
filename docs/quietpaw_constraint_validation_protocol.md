# Stage 1 constraint validation protocol

This pilot diagnoses the existing CMDP learner before changing its value representation. It uses the original trained HIM policy rather than training locomotion from scratch. The experiment is authorized by the request to validate the proposed solution.

## Preserved components

The HIM architecture and auxiliary update, original locomotion reward, noise definition and weights (force 0.2, impact 0.8), robot, control, observation/history, terrain, domain randomization, GAE, combined advantage, and PID update frequency remain unchanged. Epsilon is the actual cost budget; it does not determine the noise cost. Stage 2 is not implemented in this experiment.

New options select PID gains, a fixed budget, and diagnostic frequency. Omitting these options preserves the existing configuration. Synthetic regression checks confirm that enabling diagnostics does not change random state, parameter gradients, model tensors, losses, or PID state after a PPO update.

## Matched training matrix

Every arm starts from `rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt` (SHA-256 `2de4c315944b5db672c3af79c09a0d25867e37ba4c4e3c03ba503b322a3143a9`), with training seed 4, 4096 environments, 100 transitions per environment per update, and 300 PPO updates. Each arm receives 122,880,000 transitions. Initial network tensor hashes must match across arms.

| Arm | Epsilon | PID Kp, Ki, Kd | Purpose |
| --- | --- | --- | --- |
| grid_g0 | Balanced grid [0.08, 0.12] | 0, 0, 0 | Architecture-matched reward-only continuation |
| grid_g1 | Balanced grid [0.08, 0.12] | 1, 0.1, 0.1 | Current gains |
| grid_g10 | Balanced grid [0.08, 0.12] | 10, 1, 1 | Intermediate constraint pressure |
| grid_g100 | Balanced grid [0.08, 0.12] | 100, 10, 10 | Stronger constraint pressure |
| fixed085_g100 | Fixed 0.085 | 100, 10, 10 | Strict-budget single-condition diagnostic |
| fixed110_g100 | Fixed 0.11 | 100, 10, 10 | Loose-budget single-condition diagnostic |

The fixed-budget arms allocate every environment to one condition. Their comparison with a grid arm consequently includes differences in sample allocation; it cannot isolate representation capacity by itself.

## Measurement and screening

Each final model is evaluated deterministically with 4096 environments, evaluation seed 11, command `(1,0,0)`, and a 1000-step horizon. Every epsilon uses a fresh simulator process. Initial terrain, robot, and randomization fingerprints must match both within a sweep and across arms before interpreting comparisons. GPU PhysX trajectories need not be bitwise deterministic after initialization.

Grid models use epsilon 0.08, 0.085, 0.095, 0.11, and 0.12. Fixed-budget models are evaluated at their training budget. Only the first episode counts; later automatically reset episodes are excluded. Failure truncation and horizon/timeout bootstrapping remain distinct.

Primary measurements are observed normalized discounted cost, budget violation, raw cost per surviving step, original reward return, observed discounted reward, tracking error, survival, contact frequency, and force/impact components. Observed returns omit a tail after the evaluation horizon. Critic-bootstrapped cost is recorded separately and must not substitute for the physical measurement.

Training diagnostics include raw reward/cost advantage standard deviations, lambda-weighted component standard deviations, pre-clipping actor loss gradient norms and their ratio/cosine, per-bin PPO probability-ratio clip fraction, PID multipliers, and completed episode costs. The gradient probe uses fixed collected samples and excludes value/entropy losses. It does not measure the final optimizer displacement.

Declared pilot screening criteria, relative to the same-epsilon `grid_g0` result, are at least 3% cost reduction, no more than 5 percentage points of survival loss, and no more than 15% tracking-error increase. These are experiment screening choices, not thresholds supplied by the QuietPaw paper. Raw cost per surviving step and survival are also checked to reject apparent improvements caused primarily by early failure.

## Limits and decision

This is initially a single-training-seed pilot with matched transitions. Evaluation seeds do not replace independent training seeds. A 300-update warm-start experiment can identify gross scale or training problems but cannot establish convergence, infeasible budgets, or faithful reproduction. Promising configurations require independent-seed confirmation and longer matched runs before selecting gains for a Stage 1 versus Stage 2 comparison.

The manifest at `legged_gym/logs/quietpaw_constraint_validation_pid300_seed4/manifest.json` records commands, immutable training-source hashes, contracts, and results. Checkpoints stay in their original run directories throughout training.

## Reproduction commands

```bash
LD_LIBRARY_PATH=/home/sjc/anaconda3/envs/himloco/lib \
MPLCONFIGDIR=/tmp/quietpaw_validation_mpl WANDB_MODE=disabled \
/home/sjc/anaconda3/envs/himloco/bin/python -u \
legged_gym/legged_gym/scripts/validate_quietpaw_constraints.py \
  --output legged_gym/logs/quietpaw_constraint_validation_pid300_seed4 \
  --him-checkpoint legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt \
  --iterations 300 --num-envs 4096 --seed 4
```

The study output must be new, or `--resume-study` must be added for a manifest with exactly matching configuration and source hashes. A failed training arm is restarted from the common HIM source; already verified complete arms are retained. This is study-level recovery rather than trajectory-continuous training recovery.

```bash
MPLCONFIGDIR=/tmp/quietpaw_validation_mpl \
/home/sjc/anaconda3/envs/himloco/bin/python \
legged_gym/legged_gym/scripts/analyze_quietpaw_constraints.py \
  --study legged_gym/logs/quietpaw_constraint_validation_pid300_seed4
```

Analysis can be refreshed while training runs; it explicitly marks incomplete results. The training matrix itself runs sequentially so the GPU holds only one simulator/training process at a time.

## Follow-up declared after the seed-4 outcomes

The six-arm seed-4 pilot completed on 2026-10-02. No arm passed the primary same-epsilon lambda-zero screen. The strict fixed-budget model showed a small cost reduction relative to the loose fixed-budget model, motivating two follow-up checks:

- Evaluation seed 12 at command `(1,0,0)`, and evaluation seed 11 at command `(0.5,0,0)`, for `grid_g0`, `fixed085_g100`, and `fixed110_g100`.
- Independent fine-tuning seed 5, repeating only `fixed085_g100` and `fixed110_g100` for the same 300 updates / 122,880,000 transitions each, from the same HIM checkpoint and with the same frozen algorithm sources.

These are exploratory follow-ups selected after the primary outcome; they do not replace the original success criterion. Evaluation seeds describe different initial evaluation distributions and do not count as independent training seeds. Both fine-tuning seeds share the same trained HIM backbone initialization.

Value prediction discrepancies in the evaluation compare the training-policy critic (a stochastic Gaussian policy) with deterministic deployment returns. They include this policy mismatch and must not be called a pure estimate of the critic's training-policy calibration error.

The primary study survived a Codex service restart. Completed primary arms were checked and retained; no completed training was repeated. The follow-ups began after recovery.

## Completion

All six primary arms and both seed-5 confirmation arms completed. The eight arms received 983,040,000 training transitions in total. Final model tensors and training scalars are finite, every iteration retains 20 HIM auxiliary updates, and the listed frozen training-source hashes remain unchanged. Initial network tensor hashes match across all eight arms; paired evaluation initial-state and environment-configuration hashes match.

The seed-5 strict-versus-loose result is weaker than seed 4: observed discounted cost decreases by 1.77%, raw cost per alive step by 1.15%, tracking error increases by 15.91%, and survival decreases by 3.91 percentage points. Both strict-budget deployment results remain above epsilon 0.085. These separate fixed-budget policies do not establish within-policy epsilon control. A fixed-epsilon 0.085 lambda-zero arm has not been run, so this comparison alone does not isolate the constraint term from the constant epsilon input.

Combined interpretation: [quietpaw_constraint_validation_findings.md](./quietpaw_constraint_validation_findings.md). Seed-5 details: [quietpaw_constraint_validation_seed5.md](./quietpaw_constraint_validation_seed5.md).

A post-completion source archive is stored at `legged_gym/logs/quietpaw_constraint_validation_pid300_seed4/source_snapshot_after_validation.tar.gz`, with its file hashes and package versions in `source_snapshot_record.json` beside it. This captures repository Python code, tests, and the analysis helpers for future recovery. Only files listed in the original study manifest were checked against hashes recorded before training; the archive is explicitly a snapshot captured after completion.
