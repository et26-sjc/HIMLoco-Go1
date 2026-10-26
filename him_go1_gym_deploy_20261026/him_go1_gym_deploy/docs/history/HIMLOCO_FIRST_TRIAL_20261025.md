# HIMLoco 首次悬空策略试运行

日期：2026-10-25。用户请求在现有 WTW 验证和 HIM 代码/物理粗核基础上开始 HIM 部署尝试。本记录用于首次有人看护的悬空实机试验，不声明已经通过 HIM 承重或行走验证。

后续更新：第二次标定因触地时约 0.179 rad 的稳定误差未达到原 0.15 rad 门槛。当前启动脚本已指向专用的 0.20 rad 固定支撑触地试运行配置，操作及依据见 `HIMLOCO_CONTACT_CALIBRATION_20261025.md`。下文保留为此前悬空方案的记录。

## 可以复用与本次要确认的部分

- WTW 经验证的 SDK 3.8、网口、LCM 及 Jetson Docker 镜像可以复用。HIM 使用新的协议桥接 `lcm_position_him`，不能启动旧 WTW `lcm_position` 或其 Python 控制器。
- HIM 模型、45×6 观测/推理接口、关节名称映射、物理身份/正负号和 IMU 轴向已经完成代码检查与人工粗核。实际训练运行的 DOF dump 未提供，仍按用户接受的默认 fork 配置解释模型。
- 本次确认 HIM 的 Kp/Kd=40/1 下实际标定跟踪、策略接管及停止表现。WTW 的成功不代替这部分，但无需再次从头做 SDK/单关节测试。

## 已准备的专用配置

源文件：`go1_gym_deploy/configs/himloco_go1.suspended_trial.json`。普通 `himloco_go1.json` 不改为通用已确认运行配置。

阈值审查记录：`go1_gym_deploy/configs/evidence/himloco_go1_suspended_runtime_review_20261025.json`，SHA-256 `18da6f7853004ec6713cdaf217c28298f3fce62a5ccd261bd74f6ba7c6de2b27`。其中 `safety_reviewed=true` 表示已完成本次悬空试验参数的静态评估，**不表示已通过实机试验**。

| 参数 | 首次悬空试验值 | 依据/用途 |
| --- | --- | --- |
| PD 增益 | Kp=40、Kd=1 | 保持 HIM Go1 训练合同，未替换为 WTW 的 20/0.5 |
| 标定位置/增益渐入 | 3 s / 0.5 s | 保留已经修复的启动过程 |
| 历史/策略渐入 | 至少 6 帧 / 1 s | 保留实测预热和接管渐入 |
| 遥控速度上限 | `[0.15,0.10,0.15]` | 单位 m/s、m/s、rad/s；第一轮摇杆保持居中 |
| 跟踪误差停止 | 0.5 rad | 比一般模板 1 rad 更严格；不是硬件力矩限制 |
| 倾角停止 | 0.5 rad | 比一般模板 0.7 rad 更严格，机身应接近水平 |
| 目标单步停止 | 0.08 rad / 20 ms | 保留，不静默裁剪策略动作 |
| SDK/通信保护 | 全部保留 | PositionLimit、PowerProtect、50/100 ms watchdog 等 |

首次准备没有修改策略、观测顺序、动作比例或关节符号。首轮标定遇到桥接命令超时后，修复了 C++ 桥接的多线程计时竞争，见 `HIMLOCO_BRIDGE_TIMEOUT_FIX_20261025.md`；50/100 ms 超时保护仍保留。若实际策略触发某项停止，读取日志分析，不自动删除所有保护。

当前发布包：`/home/unitree/himloco_releases/20261025_suspended_trial_2`，桥接使用该目录内重新编译的 `go1_gym_deploy/unitree_legged_sdk_bin/build/lcm_position_him`。启动时核验二进制源码摘要，不能再使用 `20261025_alignment_1` 的旧 HIM 桥接或 WTW 桥接。首次发布包 `20261025_suspended_trial_1` 保留用于故障回看。

目标 Jetson 原 Docker 镜像在 `--network none` 下，以合成名义姿态的六帧观测完成 100 次实际模型 CPU 推理：均值约 1.712 ms、P95 约 1.831 ms、最大约 1.991 ms；该输入的完整目标相对 q0 最大偏移约 0.163 rad，位于 URDF 内。未创建 LCM 或发布命令。这只是单个合成输入的离线检查，不代表真实反馈下的控制周期或策略轨迹已经验证。完整记录：`audit/20261025_fixes/jetson_suspended_trial_offline_benchmark.json`。

## 启动步骤

保持网线和两个 SSH 会话。首轮使用固定支架/吊带承托机身，四脚离地、腿部活动范围无接触。原厂运动控制结束；先人工分开靠拢的腿，随后手撤离活动范围。遥控器开机、摇杆居中，松开 L2/R2。

### 终端 A：电脑新开终端，SSH 登录后运行桥接

```bash
ssh unitree@192.168.123.15
sudo bash /home/unitree/himloco_alignment/start_him_trial_bridge.sh
```

出现 `Press Enter to start bridge communication...` 后，确认支撑和活动空间再按 Enter。此时桥接开始收发状态；尚无有效 Python 命令前不会主动下发伺服位置目标。终端保持打开。

### 终端 B：电脑再开一个终端，SSH 登录后运行 Python

```bash
ssh unitree@192.168.123.15
bash /home/unitree/himloco_alignment/start_him_trial_policy.sh
```

该脚本直接用既有 Docker 镜像创建一个独立临时容器，无需 `docker start foxy_controller`。模型和发布包只读挂载，日志持久写入 `/home/unitree/himloco_alignment/him_trial_logs/`。脚本不在后台运行，不自动重新启动。

1. 看到 `Fresh SDK state...R2 to calibrate` 后，松开再按第一次 R2，开始 3 s 标定。
2. 等待 `HIM phase: READY`。检查四条腿接近默认姿态、无碰撞；机身始终由支架/吊带承托。若标定不正常，按 L2 或在终端 B Ctrl+C；**此时不要按 R2 来停止，因为 READY 中的 R2 会启动策略**。
3. 姿态正常后，松开再按第二次 R2。程序依次显示 `RAMP`、`RUNNING`。第一秒为渐入，随后完整策略接管。
4. 第一轮摇杆保持居中，策略运行约 2～3 s 后按 R2 或 L2 停止。**脚本没有自动 3 s 结束功能**，需要操作者停止。首轮不降低承托、不尝试行走。

## 停止与下一次尝试

- **任何阶段 L2 均请求停止**，C++ 桥接也独立读取 L2。
- R2 在 WAIT_R2 开始标定，在 READY 开始策略；在 CALIBRATE/SETTLE/RAMP/RUNNING 中再按 R2 为停止。
- 终端 B 的 Ctrl+C 也请求停止；然后在终端 A Ctrl+C 正常结束桥接。不要用 Ctrl+Z 暂停控制进程。
- 停止为 mode 0、零增益请求，机身仍需支撑；不是电气断电。故障或停止锁存后，每次都重新启动 A、B 两个程序。
- `R2 stop during RUNNING`、`L2 emergency stop` 或桥接 `remote L2 stop` 是此次人工停止的预期原因；若输出其他错误，保留原始日志分析。

本次只准备、传输和离线检查这些文件，未代用户启动电机。首轮结束后提供终端 B 的退出原因与日志文件路径，再判断能否进入承托触地试运行。
