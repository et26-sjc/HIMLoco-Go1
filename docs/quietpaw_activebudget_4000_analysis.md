# Stage 1：4000 轮活跃预算实验分析

## 结论

**这次训练完整结束，保留了 HIM 的可用运动性能；低预算触发了 PID/Lagrangian，但尚未形成有用的 ε 控制降噪行为，也没有验证成功的 QuietPaw Pareto trade-off。**

ε 对网络动作存在微弱影响。在主命令 1 m/s 的配对闭环评估中，端点 return 与折扣成本差异的条件置信区间均包含零；0.5 m/s 下有幅度极小的成本变化。不能用“输入权重非零”或极小样本均值差替代有用的可控行为证据。这个结论仅覆盖本次训练 seed 4、所评估的命令和初始地形分布。

本次只读取训练日志/checkpoint、执行确定性评估并新增分析产物；未修改算法、reward、cost、ε 映射、HIM 或仿真配置，未更新模型参数。

## 1. 实验身份与完整性

- Run：`Oct01_15-41-29_him7500_stage1_activebudget_seed4`。
- HIM warm-start：`rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt`；保留训练前 `model_him_init.pt`。
- 最终 checkpoint 内部 `iter=4000`；TensorBoard 记录 0–3999，共 4000 个 PPO iteration。
- 4096 environments × 100 rollout steps × 4000 updates = **1,638,400,000 transitions**。日志跨度约 4.30 小时。
- 实际 ε：`[0.08,0.12]`，不是当前默认 `[0.2,1]`；cost 与 ε 无直接函数关系。
- Cost：`normalized_complement_v2`；force/impact 权重 `0.2/0.8`，γ=0.99。
- 4000 轮每轮 20 次 HIM auxiliary update，共 80,000 次；HIM 更新边界沿用 Stage 1。
- 训练 scalar 和最终 model tensors 没有检测到非有限数。
- 最终 checkpoint SHA256：`4e9b620985d338307b4f5a1d680095df6b1a6b804211f0dbfe4f97b3d9b45478`。

之前只有约 59 轮的中断训练与这次完整 4000 轮是不同的 run，未混用。此次失败的主要解释已不能归结为“从零训练 100 轮还不会走”；是否达到受约束优化的收敛仍需另外判断。

## 2. 训练趋势

| 区间 | Mean episode return | Episode steps | Terrain level | Step cost | Completed discounted cost | Mean λ |
|---|---:|---:|---:|---:|---:|---:|
| 0–99 | 8.741 | 869.0 | 3.77 | 0.087790 | 0.087442 | 0.00075 |
| 100–499 | 13.424 | 927.6 | 6.03 | 0.089195 | 0.091423 | 0.00696 |
| 500–999 | 13.348 | 921.1 | 6.06 | 0.088990 | 0.091231 | 0.01613 |
| 1000–1999 | 13.411 | 925.0 | 6.04 | 0.089188 | 0.091337 | 0.03078 |
| 2000–2999 | 13.628 | 924.8 | 6.05 | 0.089118 | 0.091285 | 0.05018 |
| 3000–3499 | 13.463 | 928.7 | 6.06 | 0.088983 | 0.091224 | 0.06429 |
| 3500–3999 | 13.770 | 928.9 | 6.06 | 0.089265 | 0.091390 | 0.07410 |

最后 500 轮平均 episode length 为 928.9 steps；完成 episodes 中 timeout 比例按 episode 数加权为 84.95%。这些是随机命令、随机动作和持续地形 curriculum 下的训练指标，与固定命令的部署评估不能直接数值比较。

成本从训练 100 轮后就基本停在同一水平。低 ε 的 λ 持续增加，说明长期超预算没有被解决，而非 Lagrange multiplier 已达到稳定约束平衡。

### 最后 500 轮按 ε bin 汇总

| ε bin 均值 | Completed discounted cost | 正超预算均值 | λ 均值 | Tracking error | Timeout rate |
|---:|---:|---:|---:|---:|---:|
| 0.0825 | 0.091575 | 0.009610 | 0.3702 | 0.2327 | 85.31% |
| 0.0875 | 0.091608 | 0.005378 | 0.1771 | 0.2338 | 84.53% |
| 0.0925 | 0.091203 | 0.002009 | 0.0416 | 0.2331 | 84.38% |
| 0.0975 | 0.091285 | 0.000480 | 0.0039 | 0.2339 | 85.56% |
| 0.1025 | 0.091506 | 0.000073 | 0.0000 | 0.2335 | 85.32% |
| 0.1075 | 0.091380 | 0.000004 | 0.0000 | 0.2347 | 84.69% |
| 0.1125 | 0.091192 | 0.000000 | 0.0000 | 0.2334 | 85.14% |
| 0.1175 | 0.091371 | 0.000000 | 0.0000 | 0.2333 | 84.71% |

Completed-cost 列是每轮对有效环境的均值再做时间平均；正超预算列先对每个环境截断负误差再平均，因此不是前两列均值的简单相减。训练 bin 属于不同环境且 curriculum 可产生不同访问分布；它只能诊断约束激活，不能单独证明 ε 的因果作用。

最终最低 bin 的平均 λ=0.3948，该 bin 100% 环境的 PID cost EMA 仍高于各自预算。全局平均预算 0.1 高于全局 cost 约 0.091，不代表低 ε 约束已经满足。

## 3. 配对闭环评估

每个 ε 单独启动 Isaac Gym 进程，4096 个环境，20 s/1000 policy steps，只统计首次 episode。固定命令，冻结 terrain/command curricula，保留现有随机化；策略使用确定性均值动作。每次 sweep 的初始 terrain、robot state 和记录的随机化参数 SHA256 一致。

初始化 hash 一致并不保证 GPU PhysX 闭环轨迹完全一致。因此额外评估了 ε 列为零的 HIM warm-start，并用另一组 evaluation seed 重复最终策略端点。seed 11/12 是评估随机种子，不是两次独立训练。

### 主评估：command=(1,0,0)，seed 11

| ε | First-episode return | Step noise proxy | Observed discounted cost | 超预算 | Tracking error | Survival |
|---:|---:|---:|---:|---:|---:|---:|
| 0.08 | 22.6823 | 0.092055 | 0.094326 | 0.014326 | 0.1422 | 86.96% |
| 0.09 | 22.7055 | 0.092059 | 0.094285 | 0.004285 | 0.1422 | 87.30% |
| 0.10 | 22.7944 | 0.092104 | 0.094352 | 0.000000 | 0.1415 | 87.82% |
| 0.11 | 22.6575 | 0.092095 | 0.094342 | 0.000000 | 0.1408 | 86.99% |
| 0.12 | 22.6631 | 0.092085 | 0.094351 | 0.000000 | 0.1417 | 86.99% |

ε=0.08 的折扣成本超预算 0.014326，约为预算的 17.9%。ε≥0.10 的预算更宽，原 HIM warm-start 的成本也已低于 0.10；宽预算满足不能证明策略学会降噪。

ε 从 0.08 到 0.12 的折扣成本差为 0.00002466，环境配对差的近似 95% 区间为 [-0.00013071, 0.00018003]，包含零。该区间条件于这组环境样本，不覆盖训练随机种子或 simulator-run 的整体不确定性。

### 命令与评估种子检查

| Command / evaluation seed | Cost ε=0.08 → 0.12 | Return ε=0.08 → 0.12 | Survival ε=0.08 → 0.12 |
|---|---:|---:|---:|
| (0.5,0,0), seed 11 | 0.089582 → 0.089648 | 26.2015 → 26.1753 | 97.68% → 97.34% |
| (1,0,0), seed 11 | 0.094326 → 0.094351 | 22.6823 → 22.6631 | 86.96% → 86.99% |
| (1,0,0), seed 12 | 0.094403 → 0.094380 | 22.8136 → 22.7812 | 87.96% → 87.33% |
| (2,0,0), seed 11 | 0.090204 → 0.090018 | 11.4853 → 11.4440 | 46.22% → 45.73% |

`0.5 m/s` 下存活率约 97.5%，`1 m/s` 下约 87%，`2 m/s` 下只有约 46%，所以本模型保留的运动能力具有明显命令依赖性。2 m/s 的每存活步 proxy 约 0.098，高于 1 m/s 的约 0.092；折扣成本反而较低，主要需要警惕真实失败截断所造成的下降，不能解释为更安静。跨命令、跨失败率比较时必须同时检查每存活步成本、存活率和首次 episode 持续时间。

0.5 m/s 的成本随 ε 从 0.08 到 0.12 增加约 0.00006551，约 0.073%。环境配对差的条件 95% 区间为 [0.00001333, 0.00011769]，没有包含零；因此不能断言所有命令下的成本响应都是纯测量噪声。这个很小的正方向响应尚未在该命令下做独立 simulator-run、训练 seed 或无条件策略对照，而且最低预算仍然超标约 0.00958；它不足以建立有用的静音控制或 Pareto trade-off。

评估器 `pareto_reward_noise` 只判断有限样本点的数值支配关系，且使用未折扣 return 与每存活步成本；几个点被标为 nondominated 并不证明 CMDP 目标下有稳定、可控的 Pareto frontier。

## 4. 相比训练前是否有收益？

同条件 command=(1,0,0)、seed 11、ε=0.10：

| Metric | HIM warm-start | Stage 1, 4000 | Change |
|---|---:|---:|---:|
| Episode return | 21.964104 | 22.794436 | +0.830332 |
| Reward per alive step | 0.023720 | 0.024088 | +0.000367 |
| Step noise proxy | 0.091137 | 0.092104 | +0.000967 |
| Discounted noise cost | 0.093828 | 0.094352 | +0.000524 |
| Tracking error | 0.155055 | 0.141462 | -0.013593 |
| Survival rate | 0.833008 | 0.878174 | +0.045166 |
| Mean sampled max-foot Fz (N) | 72.035309 | 75.378113 | +3.342804 |
| Mean sampled touchdown-speed metric | 0.269420 | 0.253207 | -0.016213 |

这说明存在运动性能方面的描述性收益：tracking 改善、survival 提高，成本却略增。没有同等 transitions 的 reward-only 继续训练对照，不能把这些收益归因于 CMDP。

触地速度指标下降也不能单独视为静音收益：force 指标和接触次数同时增加，最终定义的总 noise cost 没有下降。上述速度指标包含无触地帧的零值，并非仅在 touchdown events 上条件平均。force “peak” 是 50 Hz 时刻对脚取最大，再时间/episode 平均，不是连续时间冲击最大值。

## 5. 为什么 λ 激活但行为改变很弱？

### 5.1 条件通路工作，但动作响应很小

在真实 rollout 的同一 history 上对比 ε=0.08/0.12，平均 action RMS 差为 0.0010619；乘 action scale 0.25 后，PD 目标角 RMS 差为 0.0002655 rad（约 0.0152°）。这确认 ε 能影响 actor，但目前幅度不足以支持有用的条件步态结论。

### 5.2 Cost advantage 与 reward advantage 的尺度

在独立确定性 command=(1,0,0)、ε=0.08 rollout 上，用同一 γ/GAE λ、pre-reset critic 和失败截断规则重建未标准化两路 GAE：

- std(A_r) = 0.086315。
- std(A_c) = 0.003010。
- 以最终低 ε bin 平均 λ=0.3948 估算，std(λ A_c)/std(A_r) ≈ **1.38%**。

Cost 使用 `(1−γ)c` 的预算定义，而 reward 没有同样缩放；合成后一次标准化不会把两路优势自动变成同等量级。这是当前约束项可能偏弱的实证线索。分母 `(1+λ)` 还会改变不同条件样本的相对权重。

该尺度比是部署 rollout 的诊断，不是实际随机训练 minibatch 的梯度比或 PPO clip 后作用量；不能据此精确计算所需 PID gains 或断言唯一根因。

### 5.3 PID 的更新频率与未解决超预算

PID 仅在对应环境完成新 episode 时更新。最终每个环境累计约 399–532 次更新，远少于 4000 PPO iterations；小于 0.02 的预算误差配合 ki=0.1，使 λ 积分增长较慢。低预算的 λ 近似线性上升，而成本曲线保持平坦；当前未达到受约束平衡。

### 5.4 Critic 校准

ε=0.08 rollout 在首次动作后状态的 Vc 与未来实测折扣成本比较：prediction mean=0.094210，MC target mean=0.095256，bias=-0.001046，MAE=0.004286，RMSE=0.006582。MC 采用真实失败截断、timeout/horizon 尾项；1000 步后的尾项很小。

这里的 critic 在随机高斯训练策略下学习，而 MC target 来自确定性均值动作的部署评估。因此 bias、MAE 和 RMSE 同时包含策略不匹配、采样方差和估计误差，不能直接当作训练策略上的纯校准误差。Prediction 与部署 target 的量级相近，但本诊断不足以确认 critic 校准，或把 critic 误差判定为低预算超标的唯一原因。

### 5.5 Cost 的物理构成

最后 500 轮全 bin 平均 force 部分约 0.058446，touchdown-speed 部分约 0.030819，force 仍占约 65.5%。

虽然 impact 权重为 0.8，但它只在触地事件生效；force 部分在持续接触帧生效且包含支持力，时间平均贡献仍更大。当前没有实现“超过正常支持力”的 force cost。是否存在满足 0.08 且维持运动的可行策略，尚未证明；也不能因为当前策略超标就断言预算不可能达到。

## 6. 稳定性与结论边界

- 没有非有限训练 scalar 或模型 tensor，也没有持续运动崩溃。
- Reward critic 有明显瞬时尖峰：最大 1425.98（iteration 1217），iteration 3466 为 1168.44，整次训练 24 轮 value loss >10、6 轮 >100。末 500 轮中位数约 0.0101，均值受尖峰影响。
- 这些尖峰伴随自适应学习率降至 1e-5；不能把完整运行等同于始终平稳的优化。日志不足以定位尖峰来自哪个环境或哪个物理事件。
- 只有一个训练 seed；评估 seed 变化不替代独立训练复现。
- 评估地形来自默认初始 levels 0–5，训练后期平均 level 约 6；结果不自动覆盖最高难度。
- Cost 是 50 Hz 采样物理 proxy，没有声学测量，不能直接宣称声压或真实噪声降低。

## 7. 下一步判断

**可保留本 checkpoint 作为“完整 Stage 1、保留运动能力、约束已激活但效果弱”的实验结果；当前不支持宣称成功静音控制或 QuietPaw Pareto 复现。**

优先进行小规模、明确标注的约束强度/优势尺度诊断，保持 cost 预算语义一致，并设置同等 transitions 的 reward-only 继续训练对照。较强 λ 是否能降成本且保持 tracking/survival、以及最低预算是否可行，需要实验区分。不要仅把 ε 再调小或扩回 [0.2,1] 作为已验证解决方案。

若继续实现 Stage 2，因子化 critic 可以作为论文组件的结构复现，但需以当前结果作为未成功的 Stage 1 对照；不能预先把其效果解释为解决了本次约束强度问题。

## 8. 产物与复查

- [训练趋势图](quietpaw_activebudget_4000_training_curves.png)。
- [闭环评估对照图](quietpaw_activebudget_4000_evaluation.png)。
- [完整结果 JSON](quietpaw_activebudget_4000_results.json)。
- [训练分析 JSON](quietpaw_activebudget_4000_training_analysis.json)。
- [训练 bin 表](quietpaw_activebudget_4000_bins.csv)。
- [真实观测动作/优势/critic 诊断](quietpaw_activebudget_4000_value_diagnostics.json)。

各次 evaluation 的 `legged_gym/logs/quietpaw_activebudget_*` 目录包含逐环境 first-episode 数据、初始 hash、checkpoint hash、参数契约、stdout 和 summary。日志目录是本地忽略产物。

评估复查示例：

```bash
LD_LIBRARY_PATH=/home/sjc/anaconda3/envs/himloco/lib \
  /home/sjc/anaconda3/envs/himloco/bin/python -u \
  legged_gym/legged_gym/scripts/evaluate_quietpaw.py \
  --checkpoint legged_gym/logs/quietpaw_go1_stage1/Oct01_15-41-29_him7500_stage1_activebudget_seed4/model_4000.pt \
  --output legged_gym/logs/quietpaw_activebudget_recheck_cmd1 \
  --epsilons 0.08 0.09 0.10 0.11 0.12 \
  --num-envs 4096 --steps 1000 --seed 11 --command 1 0 0
```

优势量级和 critic 校准采用临时只读评估 instrumentation；参数没有更新。完整定义与数值保存在 value diagnostics JSON。
