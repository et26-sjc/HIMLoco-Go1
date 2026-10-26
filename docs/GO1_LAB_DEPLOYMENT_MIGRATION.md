# 将当前 Go1 部署代码用于 Isaac Lab 策略

建议保留 `him_go1_gym_deploy_20261026/him_go1_gym_deploy` 作为已验证实机基线，在 Lab 项目中建立新的部署版本。SDK/LCM、状态配对、按钮状态机、日志和目标发送可以复用；训练相关的数据约定通过显式的策略配置选择。部署程序本身无需依赖 Isaac Gym 或 Isaac Lab，也无需将训练环境安装到 Jetson。

本建议针对 Lab 的 `go1_rough_pd/2026-10-22_23-51-36_go1_pd_him_seed1/exported/policy.pt`。该文件来自 `model_10000.pt`，外部接口为 float32 `[1,270] → [1,12]`。其他模型应使用其训练快照确认约定。

## 推荐方案：让部署直接构造 Lab 原生观测

新部署代码已有 `JointMap(training_dof_names)`，可直接按 Lab 名称排列，将 SDK 反馈转换为 Lab 关节顺序，并将 Lab 位置目标转换回 SDK 顺序。因此，在部署入口中不需要先构造 Gym 顺序再重排一次。

| 项目 | 现有 Gym 部署 | Lab 原生策略部署 |
| --- | --- | --- |
| 关节顺序 | FL、FR、RL、RR，每腿 hip/thigh/calf | hip/thigh/calf，每类 FL、FR、RL、RR |
| 命令观测尺度 | `[2,2,0.25]` | `[1,1,1]` |
| gyro / 关节位置 / 关节速度尺度 | 0.25 / 1 / 0.05 | 当前 PD 训练配置相同 |
| 重力 | 机身坐标 `[0,0,-1]` 的投影 | 相同物理定义；保留 SDK wxyz 的正确解释 |
| 历史 | 45×6，当前帧在前 | 相同；Lab 配置 `history_length=5` 表示当前帧加 5 帧过去观测 |
| q0 | 由名称取得 | 当前 PD run 按名称相同；仍从训练快照导出 |
| 位置目标 | `q0 + 0.25 * applied_action` | 当前 PD run 相同；在 Lab 顺序中计算后转 SDK |
| Kp/Kd、周期 | 40/1、20 ms | 当前 PD run 相同 |

Lab 的 12 维顺序为：

```text
FL_hip, FR_hip, RL_hip, RR_hip,
FL_thigh, FR_thigh, RL_thigh, RR_thigh,
FL_calf, FR_calf, RL_calf, RR_calf
```

配置中的实际名称均带 `_joint` 后缀。`previous_action` 必须使用当前策略的关节顺序和无量纲动作定义。当前 Lab wrapper 将输入动作直接交给 `env.step()`；应对照训练时 action manager 的 `last_action`、动作项裁剪顺序和正常动作范围，而不能直接假定所有裁剪前后的值总是相同。接管渐入反馈实际应用动作仍是一项需要单独记录的部署处理。

需要修改的位置：

1. `himloco/contract.py`：将命令尺度、关节默认角、动作尺度和裁剪规则等固定训练常量，整理为显式的 `gym_legacy` / `lab_native` 策略约定。当前 gyro/速度尺度不需要额外重复乘除。
2. `himloco/config.py`：现有 schema 会校验固定官方 `source_commit` 并拒绝未知字段。为 Lab 建立新版本配置，分别记录接口版本、Lab 源码版本和训练 checkpoint；保留明确的兼容性校验，不将 Lab 模型伪记为旧官方模型。
3. `himloco/runner.py`：通过所选约定构建观测与位置目标，复用 `JointMap`。历史只推进一次，动作回填与输出目标使用同一策略约定。
4. 模型打包：生成新模型 SHA 和 manifest。旧包的完整性清单属于旧发布，不能复制旧摘要或已有模型实验结论作为新模型的验证结果。

## 较少改动的方案：导出 Gym 接口兼容的 Lab JIT 包装器

如果希望先复用当前部署的固定 45×6 输入约定，可另行导出 `policy_gym_compatible.pt`，把转换编进 TorchScript：

```text
Gym history
  → reshape 为 6×45
  → 每帧命令除以 [2,2,0.25]
  → 每帧 q、dq、previous_action 从 Gym 顺序转 Lab 顺序
  → 原 Lab JIT
  → Lab 关节目标转 Gym 顺序和无量纲动作
  → 现有部署的 Gym 关节映射与发送
```

零填充历史应保留为零。若两侧 q0 或动作尺度不同，位置和上一动作的转换应通过绝对关节位置/目标计算，不能只重排：

```text
q_absolute = q0_gym + relative_q_gym
relative_q_lab = reorder(q_absolute, gym→lab) - q0_lab

previous_target_gym = q0_gym + scale_gym * previous_action_gym
previous_action_lab = (reorder(previous_target_gym, gym→lab) - q0_lab) / scale_lab

target_lab = q0_lab + scale_lab * action_lab
action_gym = (reorder(target_lab, lab→gym) - q0_gym) / scale_gym
```

应保留训练定义的动作裁剪先后关系。当前这两个 Go1 配置的 q0 与 scale 相同，以上计算可简化。包装后 JSON 的外部关节顺序仍填写 Gym 顺序；如果又改为 Lab 顺序，会发生重复置换。原始 Lab 模型、包装器和发布模型分别记录摘要及训练来源。

此前 Gym 播放的 Python I/O 适配验证了这一方向：Lab 原始 JIT 直接接收 Gym 向量时表现异常；完成命令/关节转换后能前进。该结果并未验证包装后的新 JIT，也没有验证 Lab 策略在真机上的表现。

## 建议的迁移顺序

1. 从实际 Lab run 导出机器可读的 `deployment_contract.json`：输入/输出关节名称、q0、各段尺度和单位、动作裁剪及历史动作定义、历史顺序和长度、周期、执行器类型、Kp/Kd、力矩/速度限制，以及源码/checkpoint/JIT 摘要。不要只记录 270/12 两个维度。
2. 在 Lab 中录制同步的原始 q/dq、gyro、重力、命令、45 维单帧、270 维历史、原始动作和最终关节目标。用部署代码离线重放，先逐字段对齐观测，再对齐 JIT 动作和按名称排列的绝对目标。
3. 在 Gym 中用相同测试条件对比旧策略、新策略与所选转换实现，并记录速度轨迹、reset/跌倒、机身和膝部净空、目标/实测误差。末端速度和单张画面只能说明该时刻，不能代替这些轨迹。
4. 在现有 Go1 运行镜像中先离线加载实际新 JIT，核对 CPU 推理时延和完整性清单；随后沿用现有分阶段接管流程，保存独立的新发布目录和运行日志。

优先迁移 `go1_rough_pd`，其名义 PD 参数与当前部署相同。ActuatorNet 训练结果另有执行器模型约定，不应仅凭相同输出维度直接等同于当前 40/1 MCU PD。即便是 PD 策略，也需区分 Lab 显式力矩限幅、旧 Gym URDF 限幅与 SDK/MCU 的实际执行行为。
