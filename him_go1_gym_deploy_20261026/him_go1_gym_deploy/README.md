# HIMLoco Go1 EDU 部署代码

导出日期：2026-10-26。基线：已经在本机 Go1 上运行的 `20261025_forward_trial_3`。

**当前可以基本确认部署链路可用，可以作为后续修正策略和开展对照实验的基线。** 已有代码审查、离线接口核对、实物通道/坐标粗核，以及承重、前后行走的实机证据。仍存在低姿态、膝部触地及承重跟踪误差；这份导出不将它们标记为已解决，也不等于任意权重或运动工况均已验证。

## 1. 导出内容与版本

```text
him_go1_gym_deploy/
├── README.md                         本说明与快速开始
├── himloco.json                      当前实机配置，默认入口
├── manifest.json                     原冻结包 75 个文件的校验清单
├── EXPORT_MANIFEST.json              本次全部导出静态文件的校验清单
├── models/policy.pt                  当前实际使用的 TorchScript 权重
├── start_bridge.sh                   终端 A，宿主机 SDK 桥接
├── start_policy.sh                   终端 B，Docker 中策略控制
├── scripts/
│   ├── prepare_bridge.sh             复用匹配的已编译桥接，或在 Jetson 编译
│   ├── check_environment.sh          原镜像中离线加载配置和权重
│   └── common.sh                     按脚本所在位置定位导出目录
├── go1_gym_deploy/                   原冻结包完整源码，保持包名以保留导入路径
│   ├── himloco/                      观测、映射、推理、状态机、发送
│   ├── lcm_types/                    Python/SDK 共用消息定义
│   ├── configs/                      原历史配置和对齐记录
│   └── unitree_legged_sdk_bin/        C++ 桥接源码与 CMake
├── docs/                             操作说明、接口说明、已知问题和历史记录
├── evidence/                         末次行走日志、增益核对和低姿态分析
└── logs/                             此导出后续运行日志
```

控制核心、LCM 定义、权重和根目录 `himloco.json` 原样取自实机基线；新加的是可迁移的启动脚本和文档。没有更改 PD、动作、观测、历史、策略或当前保护阈值。

**默认只使用根目录 `himloco.json`。** `go1_gym_deploy/configs/`、内部旧 README 及 `docs/history/` 保留阶段性记录，其中的阈值、状态、路径可能属于早期实验；实际使用以本 README 和根配置为准。

ARM64 桥接二进制没有包含在导出中：导出时 Go1 未能通过 SSH 连接，且本地没有该二进制副本。完整 C++ 源码已包含。原机可直接复用已通过版本/源码摘要核验的程序，见下文；SDK 库和 Docker 镜像也使用原机已有安装。

历史记录中的 `audit/...`、旧 release 和原工程绝对路径是原始证据出处；本包在 `evidence/` 中仅附最后一轮行走及其分析，不包含所有阶段的原始采集。导出检查范围见 [导出核对记录](docs/EXPORT_REPORT_20261026.md)。

## 2. 当前固定接口

| 项目 | 当前值 |
| --- | --- |
| 策略输入/输出 | float32 `[1,270]` → `[1,12]`，JIT 内含 estimator 和 actor |
| 观测历史 | 45 维 × 6 帧，最新帧在前 |
| 控制周期 | Python 20 ms；SDK 桥接 2 ms |
| 目标角 | `q_des = q0 + 0.25 * applied_action`，髋关节不再额外缩小 |
| 增益 | 全部关节 Kp=40、Kd=1；仅标定最初 0.5 s 增益渐入 |
| 目标速度/前馈力矩 | 均为 0，不额外叠加整份外部 PD 力矩 |
| 机身坐标 | x 前、y 左、z 上；SDK 四元数 wxyz，安装变换保留单位矩阵 |
| 策略关节顺序 | FL、FR、RL、RR，每腿 hip、thigh、calf |
| SDK 关节顺序 | FR、FL、RR、RL，每腿 hip、thigh、calf |
| 速度上限 | 前后 ±0.30 m/s，左右 ±0.10 m/s，偏航 ±0.15 rad/s |
| 当前额外步长停止 | `max_joint_target_step_rad=null`，按此前请求关闭；变化量仍记录 |

模型 SHA-256：

```text
71466584a0f500cc94a97ac3852b3287650ab326fcb957c7387624894cfb2456
```

## 3. 复制到原 Go1

在电脑终端执行（归档已随本目录一并生成）：

```bash
scp /home/sjc/him_go1_gym_deploy_20261026.tar.gz unitree@192.168.123.15:/home/unitree/
ssh unitree@192.168.123.15
```

下列步骤在 Go1 的 SSH 终端执行。首次解压到新目录；如果已有同名工作目录，先另行保留后续改动和日志，再选择新的解压目录，避免覆盖。

```bash
tar -xzf /home/unitree/him_go1_gym_deploy_20261026.tar.gz -C /home/unitree

bash /home/unitree/him_go1_gym_deploy/scripts/prepare_bridge.sh --reuse \
  /home/unitree/himloco_releases/20261025_forward_trial_3/go1_gym_deploy/unitree_legged_sdk_bin/build/lcm_position_him

sudo bash /home/unitree/him_go1_gym_deploy/scripts/check_environment.sh
```

`--reuse` 会检查旧二进制、校验凭据和当前源码摘要，只执行它的 `--version`；不会开启 UDP 或电机控制。目标二进制会复制到新目录，之后不用继续依赖原 release 路径。旧程序不存在时，在 Jetson 上编译：

```bash
bash /home/unitree/him_go1_gym_deploy/scripts/prepare_bridge.sh --build \
  /home/unitree/unitree_legged_sdk-go1
```

离线检查成功应显示 `configuration_valid=true`、`model_loaded=true`、`deployment_ready=true`、`motor_commands_published=false`。这里的 ready 是配置声明门控通过，不是对运动表现的自动鉴定。

## 4. 每次启动：两个 SSH 终端

操作前让机身获得可承接停止的可靠支撑，避开腿部活动范围；结束原厂运动控制。沿用这台机器此前确认的低层开发模式操作，模式切换完成后松开 L2，再启动桥接。不要同时运行 WTW、单关节脚本或另一套 HIM 控制器。

**终端 A：** 登录 Go1，启动桥接，看到提示后按 Enter。

```bash
ssh unitree@192.168.123.15
sudo bash /home/unitree/him_go1_gym_deploy/start_bridge.sh
```

看到 `UDP Initialized`、端口 8090 及循环启动信息后，保持这个终端运行。

**终端 B：** 另开电脑终端，也 SSH 到同一台 Go1，然后启动策略。

```bash
ssh unitree@192.168.123.15
sudo bash /home/unitree/him_go1_gym_deploy/start_policy.sh
```

不需要 `docker start foxy_controller`，也不需要进入旧容器。终端 B 使用原有 `jetson-model-deployment` 镜像新建临时容器 `him_go1_policy`，代码只读挂载，日志写回导出目录。

## 5. R2、摇杆与退出

1. 出现 `WAIT_R2`：摇杆居中，**按下再松开 R2 一次**，开始约 3 s 的姿态标定。
2. 等待 `CALIBRATE → SETTLE → READY`。当前标定容差为全部关节 0.20 rad，并要求连续 6 帧满足条件。不要一直按 R2，也不要在等待中反复按。
3. 确认姿态和支撑后，在 `READY` 中再次按下、松开 R2：进入 1 s `RAMP`，随后 `RUNNING`。
4. `RUNNING` 中用摇杆下发速度；本版本不自动按秒数退出。

| 操作 | 作用 |
| --- | --- |
| 左摇杆前后 | 前进/后退，最大 ±0.30 m/s |
| 左摇杆左右 | 左右横移，最大 ±0.10 m/s；左右均以机器人自身为准 |
| 右摇杆左右 | 偏航，最大 ±0.15 rad/s；当前 yaw 符号为 -1 |
| 摇杆回中 | 零速度命令，不保证原地位置保持，也不等于停止伺服 |
| L2 | 每个阶段均请求停止，C++ 桥接也独立检查 |
| R2 | `READY` 中启动；`RAMP/RUNNING` 中停止；标定等其他主动阶段再次按下也会停止 |
| 终端 B 的 Ctrl+C | 请求停止策略和桥接伺服，然后还需结束终端 A |

准备好承接机身后，按 L2 或运行阶段的 R2。终端 B 退出后，在终端 A 按 **Ctrl+C**。不要按 Ctrl+Z；它会暂停发送程序而不是完成退出。停止采用 mode 0、零增益请求，不是保持站立或确认电气断电。

停止或保护触发后会锁存。排查原因后，重新启动终端 A 和 B，再从两次 R2 流程开始。不要用 `fg` 恢复暂停的旧控制器继续运动。

## 6. 日志与后续维护

日志位于 `/home/unitree/him_go1_gym_deploy/logs/him_*.jsonl`。先保存日志再修改配置、权重或控制代码。当前证据与未完成项见 [部署状态与已知问题](docs/STATUS_AND_LIMITATIONS.md)，算法接口见 [观测与执行链路](docs/INTERFACE_AND_CHANGES.md)，故障处理、环境复用及网络说明见 [环境和排查](docs/ENVIRONMENT_AND_TROUBLESHOOTING.md)。

本导出采用前台双终端运行，**操作时保留网线和两个 SSH 会话**。无线遥控命令通过 Go1 的 SDK 状态传给桥接，电脑不执行策略；但前台进程不具备断开 SSH 后持续运行的生命周期管理。拔掉电脑网线不是经过验证的停止方式。若要脱离电脑运行，需要另行实现并验证后台启动、日志、掉线与重启行为。

本包不要求安装 HIM 训练工程、Isaac Gym、ROS，也不使用 WTW 的 `parameters.pkl`。它复用原机已验证的运行镜像；不要根据随包保留的旧 WTW Dockerfile直接重建并假定兼容。
