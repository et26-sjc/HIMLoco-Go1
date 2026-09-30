# Stage 1 correction experiments

This record answers: “How do we verify that the current Stage 1 corrections
have the intended effect?” It separates implementation correctness from policy
quality. Results below were run on 2026-09-30 in the existing `himloco`
environment on the `quietpaw-reproduction` branch. No reward, robot, PD,
terrain-generation or domain-randomization setting was changed.

## Experimental questions and acceptance tests

| Question | Experiment | Pass criterion |
| --- | --- | --- |
| C1: Are timeout targets state-correct? | Distinct current/terminal privileged states and ε; compare old HIMPPO target with corrected target | Timeout uses `gamma*V(terminal, ε)`; non-timeout is unchanged; old value remains only the GAE baseline |
| C2: Is ε independent of environment cohorts? | N=160 Go1 topology, old fixed-ID mapping vs seeded stratified mapping | Complete grid retained; all 20 joint terrain×command cohorts cover 8 bins; count imbalance ≤1; correlations with ID/terrain/command near zero |
| C2 training diagnostic | Two 2-iteration, same-seed runs: corrected assignment and temporary fixed-ID ablation | Report bin diagnostics as descriptive evidence; do not claim significance from one seed |
| C3: Are measurements paired and complete? | Fresh process per ε, fixed command/seed/initial state fingerprint | All ε files share initial fingerprint; JSON/CSV/plot include reward, cost, tracking, success and physical diagnostics |
| HIM preservation | Existing Stage 0/Stage 1 tests, real training and checkpoint optimizer states | HIM auxiliary update count equals PPO minibatch count; estimator remains detached from PPO; PPO/HIM optimizers resume together |

The C2 fixed-ID run is an in-memory monkeypatch used only as an ablation. It
is not a supported configuration and its checkpoint must not be used for final
research claims.

## C1 target experiment

The test uses a random conditional actor-critic (seed 2026), gamma=0.99, six
transitions, and three timeout flags. It evaluates distinct current and
terminal critic observations with their ε values.

```text
old timeout target mean       0.84972191
corrected timeout target mean 0.85284764
maximum absolute difference   0.01399028
non-timeout target differences 0
```

The corrected target path passes the terminal privileged observation to both
reward and cost critics. The stored `transition.values` remains `V(s_t, ε)`
for GAE/value clipping. A physical failure suppresses both timeout
bootstraps, including simultaneous failure/time-limit flags.

Artifact: `/tmp/quietpaw_c1_experiment.json`.

## C2 assignment experiment

For 160 environments, the Go1 layout yields 20 terrain columns with eight
environments each. The first 32 environment IDs are the high-command cohort,
matching the unchanged command sampler. The corrected assignment retains all
160 `linspace(0,1,160)` values and stratifies each joint cohort over eight
ε bins with a dedicated CPU seed.

| Correlation with | Fixed env-ID ε | Corrected ε |
| --- | ---: | ---: |
| Environment ID | 0.99375 | 0.02336 |
| Terrain type | 0.99253 | 0.02663 |
| Command cohort | −0.68850 | −0.01782 |

The range of joint-cohort mean ε falls from **0.95597** to **0.05267**. All
20 cohorts contain all eight bins and the per-bin count difference is at most
one. The sorted corrected values equal the original complete grid exactly.
The mapping does not consume the global PyTorch RNG. A cohort with fewer than
eight environments is rejected explicitly.

Artifact: `/tmp/quietpaw_c2_experiment.json`.

### Short training ablation

Both runs used seed 1, 160 environments, two iterations and 32,000
transitions. The fixed-ID run was a temporary monkeypatch; all other code was
the same. Latest per-bin diagnostics were:

| Diagnostic range across 8 bins | Corrected assignment | Fixed-ID ablation |
| --- | ---: | ---: |
| Noise cost mean | 0.01559 | 0.01578 |
| Reward mean | 0.01092 | 0.00956 |
| Velocity tracking error mean | 0.17156 | 0.14785 |
| fz peak mean | **5.14259 N** | **8.06445 N** |
| Completed episodes | 8 | 10 |

The corrected run has visibly lower force-peak bin spread in this one seed,
while other ranges do not uniformly improve. This is the expected use of the
ablation: it demonstrates that old assignment can imprint cohort differences
into ε-bin summaries, but it does not establish a treatment effect.

Runs:

- Corrected: `legged_gym/logs/quietpaw_go1_stage1/Sep29_22-26-37_stage1_correction/`.
- Fixed-ID ablation: `legged_gym/logs/quietpaw_go1_stage1/Sep30_17-00-18_stage1_unbalanced_ablation_20260930/`.
- Ablation log: `/tmp/quietpaw_unbalanced_ablation.log`.

## C3 paired evaluation

The evaluation launcher starts one fresh Isaac Gym process per ε. It fixes the
command `(1,0,0)`, seed, initial terrain/terrain levels, initial robot state
and available randomized parameters; it freezes terrain and command curricula.
It counts only each environment's first episode, so auto-reset samples after
a failure cannot inflate reward, success or physical peaks.

### Corrected four-iteration checkpoint

The sweep used 160 environments, 500 steps, seed 11, ε ∈
`{0, .25, .5, .75, 1}`. All five initial-state fingerprints and checkpoint
fingerprints matched within the sweep.

| ε | Return | Noise proxy | Tracking error | Success | fz peak max (N) | Touchdowns/episode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.747615 | 0.216404 | 1.003842 | 1.0 | 165.860 | 5.444 |
| .25 | 4.747639 | 0.216395 | 1.003839 | 1.0 | 165.729 | 5.425 |
| .5 | 4.747699 | 0.216386 | 1.003834 | 1.0 | 165.663 | 5.437 |
| .75 | 4.747768 | 0.216375 | 1.003831 | 1.0 | 165.795 | 5.419 |
| 1 | 4.747831 | 0.216358 | 1.003827 | 1.0 | 165.715 | 5.419 |

The small monotonic differences are below what can be interpreted as learned
noise preference after only four training iterations. They verify that the
conditional policy, cost measurement, aggregation and plotting path execute.

### Same-duration assignment comparison

Balanced and fixed-ID two-iteration checkpoints were evaluated with the same
seed 19, fixed command, 20 environments (one per terrain column), and 300
steps. Each sweep's three initial fingerprints matched. Both had success rate
1.0.

| Assignment | Mean return (ε=0,.5,1) | Mean noise proxy | Tracking error |
| --- | ---: | ---: | ---: |
| Corrected | 2.699429 / 2.699548 / 2.699572 | 0.212098 / 0.212095 / 0.212076 | 1.007763 / 1.007764 / 1.007771 |
| Fixed-ID | 2.688854 / 2.688934 / 2.688739 | 0.212435 / 0.212441 / 0.212440 | 1.008145 / 1.008143 / 1.008137 |

These values are not a statistical comparison: one seed, two updates and
short horizons cannot support a performance claim. The useful result is that
both policies can be evaluated on identical initial conditions, so future
multi-seed differences will not inherit the original ε-to-environment-ID
confound.

Artifacts:

- Five-level sweep: `legged_gym/logs/quietpaw_experiment_eval_20260930/`.
- Balanced two-iteration sweep: `legged_gym/logs/quietpaw_eval_balanced2_20260930/`.
- Fixed-ID two-iteration sweep: `legged_gym/logs/quietpaw_eval_fixedid2_20260930/`.

## Training and regression checks

- Corrected 160-environment training: two iterations, 32,000 transitions;
  two resumed iterations; PPO/HIM Adam steps 40 → 60 → 80.
- Original `go1`: one-iteration, 16-environment regression completed with a
  finite checkpoint.
- 22 tests pass, including C1 target math, C2 grid/cohort properties, C3
  first-episode arithmetic, Stage 0 hashes and exact HIM auxiliary replay.
- `compileall` and `git diff --check` pass.

## Recommended full experiment before scientific conclusions

The current experiments validate implementation and measurement. To test the
research hypothesis, train corrected and fixed-ID ablation policies for at
least three independent seeds, the same long horizon and total updates. Use
the same terrain/command seed sets for every policy, evaluate all eight paper
ε values on paired initial states, and report confidence intervals for reward,
tracking, success, raw proxy cost, discounted budget violation and each force /
impact diagnostic. Add an unconditioned HIM baseline and an oracle fixed-ε
PPO-Lagrangian baseline. Only then test monotone ε response and construct a
Pareto frontier; do not treat the current smoke plots as evidence of one.
