# Follow-up: separate update stability from constraint strength

This protocol is written before the new replay/short-training results. Keep
speed-only cost, epsilon=0.037646, robot/environment/reward and HIM architecture,
auxiliary objective and update count unchanged. Source is exclusively original
HIM `Sep29_01-01-22_baseline_seed1/model_8000.pt`.

## Hypotheses

1. No hard loss/sign/bootstrap bug has been established. Existing tests do not
   prove learning stability or correctness of the acoustic proxy.
2. Weight-only transfer discards Adam state and starts at LR=1e-3, whereas both
   saved HIM optimizer groups use 1e-5. The first update may damage the mature
   policy/representation before the adaptive schedule reduces LR.
3. The normalized cost advantage and small PID multipliers give weak effective
   cost pressure. This remains a separate issue from warm-start stability.

## E1: exact-data first-update replay (run first)

Seeds 4,5; collect one 4096 x 100 rollout with lambda=0 from HIM8000. Replay
the same rollout, weights, fresh optimizers and RNG twice: initial LR=1e-3
versus 1e-5. Keep adaptive schedule, 20 PPO/HIM minibatch updates unchanged.
No actor/HIM freezing, new objective, or changed PID formula.

Measure initial and final policy KL to the collection distribution, per-step
HIM-only and PPO-only changes on fixed, evenly spaced rollout observations,
action-mean drift relative to action std, LR per minibatch, parameter-change
norms and optimizer step counts. A large KL at lambda=0 is evidence of update
instability, not cost optimization. Probe forwards must consume no RNG and
must not populate gradients. This is an off-line update experiment, not an
on-policy training-performance comparison.

## E2: short on-policy stability check

If E1 shows a smaller first-update KL with 1e-5 in both seeds, run 20 updates
for both initial LRs with lambda=0, seeds 4,5. Only initial LR differs; both use
fresh optimizers and the original adaptive rule. Collect 8,192,000 transitions
per arm. Evaluate every final checkpoint on stochastic training conditions,
seeds 111,112, 4096 full first episodes; fixed checkpoint=20. Compare survival,
cost, tracking, training failures, terrain occupancy and HIM drift. E1 alone
does not justify a claim about stable continued training.

If the original adaptive rule subsequently increases LR and instability returns,
test an explicit LR cap as a separate follow-up, not an unreported change. If
lower initial LR fails, audit optimizer-state transfer and HIM/actor coupling
before increasing constraint pressure. A checkpoint LR match is a controlled
initialization choice, not a faithful full training resume.

## E3: constraint-strength test (conditional; not launched by E1/E2)

After stability is supported, fix that initialization across arms. Use PID gain
scales 0,1,100,1000 with all three gains scaled together, initial lambda=0,
EMA unchanged; seeds 4,5, 300 updates, same calibrated epsilon and evaluation
protocol. These are parameter ablations, not QuietPaw-author hyperparameters.
Do not normalize cost advantage independently or change cost/budget units.

Before committing the large sweep, replay gains on stored per-env episode
costs and verify multiplier arithmetic; this cannot predict changed policy
occupancy. Record actor and std gradients separately, at multiple rollout
time blocks, both before/after joint clipping, and the reward/cost critics'
contributions to global gradient clipping. Do not pick gains from final eval.

Primary evidence: paired Jc and Jc-epsilon, survival and alive-time cost at
the fixed endpoint. Larger cost pressure plus lower cost supports a scale
problem. Large pressure with no cost reduction points to advantage/critic,
proxy detection or infeasibility; lower cost with failure is not success.
Keep per-minibatch KL/HIM drift to detect stability loss from stronger pressure.

Only after a stable, effective fixed-budget loop should a multi-epsilon or
Stage 2 representation experiment begin. Two seeds are diagnostic only.
