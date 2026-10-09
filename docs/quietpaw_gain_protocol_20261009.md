# E3: stable-start PID gain experiment

Execution of the gain experiment specified in the follow-up protocol. This
document fixes endpoint and acceptance rules before E3 results are available.

- Original HIM8000 only: `rough_go1/Sep29_01-01-22_baseline_seed1/model_8000.pt`.
- Initial PPO LR=1e-5; original adaptive LR rule retained. HIM receives the same
  LR each minibatch, exactly as before. Fresh Adam in all arms.
- Speed-only cost and epsilon=0.037646 unchanged. No reward modification,
  independent cost-advantage normalization, estimator freezing, or Stage 2.
- PID gain factors 0,1,100,1000 multiply Kp,Ki,Kd together; EMA=.95 and initial
  lambda=0. Factors are diagnostic choices, not author-reported parameters.
- Seeds 4 and 5, 4096 environments, 100 rollout steps, 300 updates each. Each
  arm has 122,880,000 transitions and 6000 PPO/HIM updates. Eight arms total
  983,040,000 transitions. Run order: seed4 then seed5, factors ascending.
- Initial weights, both optimizer states and RNG/environment hashes must match
  within seed before training. Store initial/full final checkpoints and source
  hashes. Do not change settings or select an intermediate checkpoint by score.
- Primary evaluation: final update300, stochastic policy, training distribution,
  independent seeds111/112, 4096 complete first episodes each. Report observed
  discounted cost, timeout tail, cost-critic calibration, alive-time cost,
  survival, tracking, reward and touchdown rate.
- Secondary: deterministic command(1,0,0), seed11, 4096 envs, 1000 steps,
  existing 200-Hz observer, including runtime PD-input checks and censoring.

## Instrumentation and arithmetic checks

Before training, replay the stored per-environment normalized episode costs
(including timeout tails) from the two E2 low-LR seed4 evaluations through the
PID. This is an arithmetic check, not an on-policy counterfactual trajectory.
Check zero-gain lambda remains zero and positive scaling of all gains scales
lambda by that factor on identical input sequences. Verify against an explicit
reference recursion and also test invalid-event masking/integral decrease.

Store real per-env PID inputs from every E3 update for subsequent replay. Record
LR, policy KL and HIM-induced policy drift per minibatch. At iterations
0,50,100,150,200,250,299, measure reward/cost gradients separately for actor and
std, before/after the actual joint PPO clipping gate, at three predetermined
rollout blocks (steps0-1,49-50,98-99). Record actual reward/cost critic gradient
norms entering global clipping in every minibatch. Diagnostics must not consume
RNG, alter model/optimizer state, or write gradients. Test learning equivalence.

## Interpretation fixed before results

Report every arm. A candidate passes this diagnostic only if BOTH training
seeds have mean primary Jc<=epsilon, lower alive-time cost than paired gain0,
survival no more than 1 percentage point below gain0 and tracking error no more
than 10% higher. These are experiment acceptance tolerances, not statistical
equivalence claims. Report both eval seeds and physical metrics separately.
Do not claim acoustic improvement or a controllable multi-epsilon policy.

If no arm passes, state that outcome. Increased lambda alone is not success.
If effective cost gradients remain weak, inspect scales and critic estimates;
if they strengthen without cost reduction, investigate estimation/detection or
feasibility; if locomotion collapses, classify low cost as a failure. If adaptive
LR instability returns, test an LR cap separately, never patch this experiment
mid-run. Two training seeds support diagnosis, not publication-level robustness.

Artifacts: `legged_gym/logs/quietpaw_gain_20261009/`. Original source files and
the existing user edits remain intact. Only experiment wrappers are added.
