# Go1 部署代码与当前 Gym 项目对照

审核日期：2026-10-26。部署包：`him_go1_gym_deploy_20261026/him_go1_gym_deploy`；实机基线：`20261025_forward_trial_3`。

本次源码审阅与离线数值核对未发现当前 HIM 入口存在核心观测、动作或名义控制参数错配。部署包采用 WTW 的 SDK/LCM 通信框架，但当前控制入口使用独立的 HIM 实现，未调用保留的 WTW 旧观测和控制循环。没有修改冻结包内文件或重新连接机器人。

结论适用于本项目当前源码默认参数，以及包内这一份模型。历史训练 run 的完整配置和运行时 DOF dump 没有因此补齐；实际步态和硬件表现的证据范围见部署包的 `docs/STATUS_AND_LIMITATIONS.md`。

## 名义参数与数据接口

| 项目 | 当前 Gym 实现 | 部署实现 | 对照结果 |
| --- | --- | --- | --- |
| 单帧/历史/动作维度 | 45 / 6 帧 / 12 | 45 / 6 帧 / 12 | 一致 |
| 历史排列 | 当前帧在前 | 当前帧在前，每周期推进一次 | 一致 |
| 策略周期 | 0.005 s × decimation 4 = 0.02 s | Python 0.02 s | 一致 |
| 命令尺度 | `[2, 2, 0.25]` | `[2, 2, 0.25]` | 一致 |
| 角速度/关节位置/关节速度尺度 | 0.25 / 1 / 0.05 | 0.25 / 1 / 0.05 | 一致 |
| 观测与动作裁剪 | ±100 / ±100 | ±100 / ±100 | 一致 |
| 关节默认角 | 左髋 0.1、右髋 −0.1；前腿 thigh 0.8、后腿 1.0；calf −1.5 rad | 按名称使用同样数值 | 一致 |
| 动作目标 | `q0 + 0.25 * action` | `q0 + 0.25 * applied_action` | 正常 RUNNING 一致 |
| 髋关节额外缩放 | `hip_reduction=1.0` | 无额外缩放 | 一致 |
| PD 名义增益 | Kp=40、Kd=1 | Kp=40、Kd=1 | 一致 |
| 关节位置限值 | 当前 Go1 URDF | Python 与 C++ 使用对应 URDF 限值 | 一致 |
| 关节排列 | 当前 sim2sim 配置为 FL、FR、RL、RR，每腿 hip/thigh/calf | 根配置相同；`JointMap` 双向映射 SDK 顺序 | 与当前显式配置一致 |
| JIT 内容 | encoder + actor | 单个 JIT，外部不重复估计速度或 latent | 一致 |

Gym 对照源码是 `legged_gym/legged_gym/envs/go1/go1_config.py`、`envs/base/legged_robot_config.py`、`envs/base/legged_robot.py`、`utils/helpers.py` 及 `mujoco_sim2sim/config.py`。部署对应实现是包内 `go1_gym_deploy/himloco/{contract,policy,runner,state,config}.py`。

当前部署模型 SHA-256 为 `71466584a0f500cc94a97ac3852b3287650ab326fcb957c7387624894cfb2456`。它与 `legged_gym/logs/rough_go1/exported/policies/policy.pt` 的文件字节哈希不同，但全部 state_dict 张量及 forward 代码相同，实际部署加载器的推理输出也相同。

## 已核对的 WTW 改造点

- 活动入口为 `go1_gym_deploy/scripts/deploy_policy.py`，使用 HIM 的 `contract.py`、`policy.py`、`runner.py` 和 `state.py`。保留的 `envs/lcm_agent.py`、旧 history wrapper 和 `utils/deployment_runner.py` 不在此入口的控制调用链中。
- 输入为 `[command, gyro, projected_gravity, q-q0, dq, previous_action]`，不混入 WTW 的 gait/clock/camera 特征。Gym 的地形高度和真实机身线速度属于训练时的 privileged/critic 信息，不需要补入这份 45 维 actor 观测。
- 使用机身原生 gyro；投影重力由 SDK wxyz 姿态计算。Gym 内部四元数为 xyzw，但代码没有将 SDK wxyz 直接当 xyzw 使用。
- `previous_action` 在 RUNNING 阶段记录裁剪后的无量纲动作，未误写成绝对目标角或再乘 0.25。关节反馈和目标均按名称转换。
- C++ 下发位置目标及 Kp/Kd，目标速度和前馈力矩为零，没有把完整外部 PD 力矩再叠加到 MCU 的 PD 上。

## 有意增加的部署行为与物理差异

这些差异不能按“所有环节与仿真完全一致”描述，也不构成本次发现的观测错配：

1. Gym episode reset 使用历史零填充；部署先完成标定并收集至少六帧真实历史，再接管策略。动作接管的 1 s 渐入期反馈实际应用的 `alpha * clipped_action`。
2. 部署遥控上限为 `[0.30, 0.10, 0.15]`，小于训练命令范围。部署直接输入 yaw rate；训练的 heading 模式也先算出 yaw rate 再进入前三维观测，因此不需要额外添加 heading 观测。
3. 仿真以 5 ms 物理步执行显式 PD 并按 URDF effort 裁剪；部署由 2 ms SDK 桥接发送目标，MCU 执行 PD，并保留 SDK 位置/功率保护。二者不能仅凭相同 40/1 增益认定力矩饱和和动态响应一致。
4. SDK 的可接受目标范围与训练 URDF 不完全相同。当前桥接在 SDK `PositionLimit` 修改目标时停止，不静默裁剪后继续。某个训练合法目标仍可能触发这一停止分支。
5. 训练中的观测噪声、随机延迟和增益随机化不在部署时人为重现。部署附加的状态新鲜度、跟踪误差、倾角和停止门控是运行参数，不是训练超参数。

包内记录仍包含低姿态、膝触地和目标/实测跟踪偏差。本次接口核对不能将这些表现归因于某一项控制代码，也没有通过增加姿态偏置或调高增益处理它们。

## 本次执行的离线检查

| 检查 | 结果 |
| --- | --- |
| 原 `manifest.json` 文件摘要 | 75/75 匹配 |
| `EXPORT_MANIFEST.json` 文件摘要 | 99/99 匹配 |
| 现有 Python 测试 | 共 77 项，75 通过、2 跳过；无失败 |
| 跳过的范围 | 固定官方源码提交的 2 项对照测试，未配置该快照 |
| 当前项目源码独立对照 | 20 项检查通过，包含默认角、尺度、周期、增益、URDF 限值、模型与观测 |
| 当前项目观测/历史 AST 表达式数值对照 | 100 个连续随机状态，逐元素相同 |
| 部署 JIT 与当前 rough_go1 JIT | 全部参数、forward 代码相同；20 组输入通过实际部署加载器的输出逐元素相同 |
| SDK/LCM 独立的 C++ bridge guard 测试 | 1/1 通过 |

Python 检查使用现有 `himloco` 环境；C++ 只编译无 SDK 依赖的门控测试，没有编译或运行机器人桥接。

在部署包目录可复现现有测试：

```bash
python -B -m unittest discover -s go1_gym_deploy/tests/himloco -p 'test_*.py'
cmake -S go1_gym_deploy/unitree_legged_sdk_bin -B /tmp/him-go1-guard -DHIM_GUARD_TEST_ONLY=ON
cmake --build /tmp/him-go1-guard
ctest --test-dir /tmp/him-go1-guard --output-on-failure
```
