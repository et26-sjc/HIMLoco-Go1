# Fixed-budget constraint mechanism test — preregistered 2026-10-09

Question: starting from exactly the same HIM8000 weights and fresh optimizer
states, does the existing PID-Lagrangian reduce the speed-only cost it actually
optimizes? This is Stage 1 diagnosis, not Stage 2 or a new cost experiment.

## Frozen choices (before calibration or new training results)

- Source: `legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_8000.pt`.
  Do not use later HIM weights. Oct07/Oct08 CMDP8000 models were trained from
  scratch and are historical comparisons only, never initialization sources.
- Unmodified speed-only `normalized_complement_v2`: force weight 0, impact
  weight 1, sigma_v=0.5, existing 50-Hz touchdown mask and timing.
- Actor, HIM, reward critic and std are transferred by the existing HIM warm
  start; cost critic hidden layers copy reward critic and output starts zero.
  PPO/HIM optimizer states start empty in both arms, not resumed from HIM.
- Calibration: seeds 101 and 102; 4096 environments each; random actions drawn
  from the loaded HIM8000 Gaussian distribution; original training commands,
  terrain, randomization and curriculum. No fixed forward command and no
  deterministic mean-action substitution. One complete first episode per
  environment, up to the actual timeout (1001 policy steps at gamma=.99).
- Budget rule: epsilon = round(0.90 * pooled mean observed normalized discounted
  episode cost, 6). Distribution quantiles are descriptive, not feasibility
  guarantees. Use no critic estimate to select epsilon. At timeouts, report
  the omitted continuation bound gamma^T (bounded c in [0,1]); require its
  maximum to be <=1e-4, otherwise stop and revise the protocol explicitly.
- Training: seeds 4 and 5; two arms per seed, current PID gains (1,.1,.1),
  EMA=.95, versus gains all zero and initial lambda=0. This disables constraint
  pressure while retaining cost critic and identical HIM update counts. No
  gain sweep or additional tuning based on results.
- Each arm: 4096 environments ×100 steps ×300 updates =122,880,000 transitions;
  four arms total 491,520,000 transitions. Five epochs ×four minibatches yield
  6000 PPO and 6000 HIM optimizer steps per arm. Same fixed epsilon in all arms.
- Primary checkpoint: update 300. Updates 100 and 200 are learning diagnostics
  only; do not select the most favorable checkpoint. No early success stopping.
- Save full initial checkpoint and simulator/RNG fingerprints. Verify actor,
  HIM source/target/prototypes, both critics, std, optimizer initial state and
  environment initialization are identical within each seed. PID gains are the
  only training intervention. Both arms have all-zero initial PID state except
  identical epsilon-dependent cost EMA. Later trajectories may diverge due to
  policy differences, resets and random-number consumption.

## Measurements

- Primary evaluation: independent seeds 111 and 112, random policy actions,
  training environment/command distribution, full first episodes, 4096 envs.
  Evaluate the initial policy as reference and both final policies, same
  conditions. Record directly observed discounted cost AND timeout-bootstrapped
  PID estimate; report tail separately and check critic calibration. Primary
  paired contrasts are PID minus lambda=0 within training seed.
- Deployment diagnostic: final checkpoint, deterministic actions, command
  (1,0,0), eval seed 11, 4096 environments, 1000 steps. Freeze evaluation
  curriculum equally across arms. Retain other baseline randomization.
- Report mean signed Jc-epsilon, positive violation, failure, alive duration,
  per-alive-step and per-alive-second cost, touchdown frequency, tracking error,
  locomotion reward. Never interpret failure-induced small cost as improvement.
- Gradient probes at iterations 0,50,100,150,200,250,299, before and after PPO
  update, same predetermined rollout samples: absolute reward and cost loss
  gradient norms, lambda-weighted ratio and cosine, raw unit-weight cost norm,
  PPO clipping-active fraction. Shared advantage standardization is retained.
  Clipped contributions use the actual joint-advantage clipping gate; they are
  not two independently clipped objectives. Autograd probes must not mutate
  model parameters, optimizer state, RNG or existing .grad buffers.
- 200-Hz observer is evaluation-only. Reuse the established 40-ms contact-window
  definitions, force/rate peaks, pre-contact downward speed and base acceleration.
  Verify that added force/root/rigid-body refresh leaves the PD joint-state
  inputs unchanged. Apply the same observer to both arms; no physical metrics
  become training costs. Report missing/censored events and 50/200-Hz timing
  limitations. No claim of acoustic SPL reduction.

## Interpretation rules

- Low lambda with satisfied budget is normal; low effective cost gradient with
  persistent violation is insufficient constraint pressure, not success.
- Report outcomes at fixed budget/compute even if PID fails. Do not automatically
  increase gains, change proxy, extend training or choose another checkpoint.
- A cost decrease with worse survival must be qualified using alive-time cost
  and survival metrics. Paired common-survivor comparisons are descriptive,
  because survival is an outcome.
- Two training seeds support mechanism diagnosis only. Thousands of simulator
  instances and evaluation bootstrap intervals do not replace training seeds.
- Stage 2 is a subsequent matched representation experiment; it is not run here.

Artifacts and exact source/checkpoint hashes will be saved under
`legged_gym/logs/quietpaw_fixed_budget_20261009/`. Existing user changes and
previous experiment artifacts are preserved.
