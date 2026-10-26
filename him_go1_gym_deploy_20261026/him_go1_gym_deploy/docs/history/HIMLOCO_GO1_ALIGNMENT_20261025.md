# HIMLoco Go1 启动修复与实物对齐记录

日期：2026-10-25。项目实际根目录为 `/home/sjc/HIMLoco_WTW_deploy/HIMLoco_WTW_deploy`。

## 当前状态

两处启动问题已修复，部署 JSON 已填入可确定的信息；模型在 Jetson 原 WTW Docker 镜像内通过离线加载和 forward。经用户确认机身已承托、四腿离地并结束原厂运动控制后，启动了专用 `--observe-only` 桥接。桥接不订阅电机目标，持续输出 SDK mode 0、零增益；本次尚未运行 HIM 标定、策略伺服或行走。

静止、三轴运动和十二关节数据均已采集。补采后三个 IMU 动作的方向、gyro 与起止静止检查通过。关节严格自动端点检查 7 项通过、5 项未通过，后者有被动联动或提前回位。用户明确补充：右后 hip 项先动错为左后 hip，发现后在**同一项采集中改做右后 hip**；完整曲线呈现同样先左后右的变化，疑点已解决，无需补采两个后髋。

按用户接受的粗核范围，已完成人工关节身份/方向及 IMU 轴向复核，JSON 已将 `alignment.verified`、`joint_mapping_verified` 置为 true，并引用带范围说明和原始文件 SHA 的人工复核记录。严格自动报告仍保留 false；这两种判定不混称。`runtime.safety_reviewed=false` 保留，完整主动运行仍被阻止：尚需运行阈值评估和 HIM 悬空实机标定。

用户接受粗略坐标与关节身份核对，无需将重力联动误当成机械故障或整套重采。尚未用水平仪确认机身刚性参考面；保留单位 IMU 变换，不使用手动三轴拟合的小角度修正。本次未执行 HIM 实机标定或策略。

修改前备份：`audit/20261025_fixes/before_184232.tar.gz`。早先的审查报告保留为修改前快照：`/home/sjc/walk-these-ways/docs/himloco_deployment_review_20261025.md`。

## 修改内容

| 文件 | 处理及原因 |
| --- | --- |
| `go1_gym_deploy/scripts/verify_bridge_build.py` | 将 Python 3.6 不支持的 subprocess 参数换为 PIPE、universal_newlines；保留二进制和源码摘要校验 |
| `go1_gym_deploy/himloco/runner.py` | 修复趴卧反馈 -2.73～-2.80 rad 导致等待/首条标定目标被拒绝；分阶段容差、合法首目标、增益渐入 |
| `go1_gym_deploy/himloco/config.py` | 对三个新增标定参数做显式范围校验 |
| `go1_gym_deploy/configs/himloco_go1.example.json` | 补齐新增参数，继续保留空白来源/确认字段 |
| `go1_gym_deploy/configs/himloco_go1.json` | 新增实际模型配置，记录 SHA、默认训练来源、显式关节顺序及待确认项 |
| `go1_gym_deploy/unitree_legged_sdk_bin/bridge_guard.hpp` | 用 float32 端点与 float32 电机目标比较，避免合法端点因 float/double 舍入被拒绝 |
| `go1_gym_deploy/unitree_legged_sdk_bin/lcm_position.cpp` | 增加 `--observe-only`，该分支不订阅目标通道，不能通过 R2 或 LCM 目标激活伺服 |
| `go1_gym_deploy/himloco/state.py` | 保存原始加速度用于对照；策略重力仍来自姿态四元数 |
| `go1_gym_deploy/himloco/alignment.py` | 新增离线姿态/gyro/关节分析；生成候选变换，不写确认开关 |
| `go1_gym_deploy/scripts/check_alignment.py` | 新增只订阅采集和逐步提示；保留完整性标记、时间、tick、四元数/RPY/gyro/加速度/关节原始数据 |
| `go1_gym_deploy/tests/himloco/` | 补充趴卧启动、增益、Python 3.6 调用、端点及坐标对齐的合成检查 |

### 标定修复的具体语义

- `startup_joint_margin_rad=0.2`：WAIT_R2 可接受训练 URDF 外最多 0.2 rad 的实测角；标定中逐渐降至运行期 0.02 rad 容差。更大的异常实测角仍拒绝。
- 首目标为 `clip(q_measured, lower+0.001, upper-0.001)`，避免原样发送训练边界外的趴卧读数。位置在 3 s 内插值到 q0。
- 首帧 Kp/Kd 为零；`calibration_gain_ramp_s=0.5` 内渐增至 40/1。这是实际新增的启动逻辑，尚需悬空实机标定验证。
- 正常策略继续按 45×6 最新帧在前的观测历史、`q_des=q0+0.25*action`、Kp/Kd=40/1 工作。动作目标范围、0.08 rad 单步拒绝、SDK PositionLimit/PowerProtect、watchdog 保留。

## 实际配置依据

- 模型 SHA-256：`71466584a0f500cc94a97ac3852b3287650ab326fcb957c7387624894cfb2456`。
- 用户指定训练默认来源：HIMLoco-Go1 `c59f76349c4718da19aa9b055a830ac1a383f038`。接口基准：官方 `ef289acaa62795009363b7b819c9186690630441`。
- 策略顺序按该 fork 的显式 sim2sim Go1 配置采用 FL、FR、RL、RR；每腿 hip、thigh、calf。实际训练运行的资产 DOF 导出记录仍缺失，这一来源假设保留在 JSON 中。
- SDK 顺序 FR、FL、RR、RL；默认置换 `[3,4,5,0,1,2,9,10,11,6,7,8]`。十二关节物理身份和正负号已经被动动作粗核；保留原置换，没有交换后腿。
- SDK 头文件明确为 wxyz，实际 JSON 选择 `quaternion_wxyz`；R_BS 经三轴动作粗核保留单位矩阵。静止姿态不自动当成 IMU 偏差减掉，也不声明精确安装角标定完成。

## 已执行的检查与证据

1. 本机 73 项 Python 检查通过，包括直接对照固定官方源码的数值检查。日志：`audit/20261025_fixes/python_tests.log`。
2. 本机和 Jetson 的 C++ guard 检查通过。新桥接在 Jetson g++ 7.5、CMake 3.10 下完整编译；宿主 Python 3.6 实际版本/源码摘要校验通过。日志：`audit/20261025_fixes/jetson_bridge_verification.log`。
3. 坐标分析器在 Jetson 原镜像内的检查日志：`audit/20261025_fixes/jetson_alignment_tests.log`。包括非单位安装变换、q/-q、RPY 错误、gyro 反号、物理水平缺失、错误关节及负向运动拒绝等。
4. 原 `jetson-model-deployment` 镜像内实际模型 `[1,270]→[1,12]` 通过。离线容器使用 `--network none --runtime nvidia`；即使 CPU 推理，这个镜像仍需要 NVIDIA runtime 提供依赖库。`foxy_controller` 原容器保持停止；新代码用单独挂载运行。
5. 实际 SDK 静态 `Safety::PositionLimit` 离线探针不创建 UDP/LCM。结果 hip≈±1.047，thigh≈[-0.663,2.966]，calf≈[-2.721,-0.837] rad；修复后的首条示例标定目标未被改写。源码/输出：`audit/20261025_fixes/sdk_limits_offline.cpp`、`.log`。SDK 大腿范围比训练 URDF 窄，未来策略目标超出时桥接仍会停止。

### 静止实物采集

用户确认支架承托后，以专用观测桥接采集 5 s，共 238 个新鲜配对状态。证据：`audit/20261025_fixes/baseline.jsonl`、`baseline_summary.json`。

| 项目 | 结果 |
| --- | --- |
| roll / pitch 均值 | -0.2225° / -0.4785° |
| roll / pitch 标准差 | 0.00734° / 0.00403° |
| 四元数模长 | 0.999999971～1.000000030 |
| RPY 与 wxyz 旋转最大差 | 约 4.12e-7° |
| gyro 均值 rad/s | [-0.000115,-0.000290,0.000183] |
| 加速度均值 m/s² | [0.0844,-0.0396,9.7996] |
| 姿态投影重力候选 | [-0.00835,0.00388,-0.99996] |

可据此确认实际数据链路、wxyz 解析与 SDK RPY 的数值一致性；无法仅凭该数据确定物理轴向、水平偏差来源或关节身份。用户说明“支架撑着就是水平”；尚未提供水平仪测量，因此不加 `--level-confirmed`。

## 首轮采集流程（留档，已完成）

Jetson 上观测桥接正在前台 SSH 会话中运行：

`/home/unitree/himloco_releases/20261025_alignment_1/go1_gym_deploy/unitree_legged_sdk_bin/build/lcm_position_him --observe-only`

当前工具发布目录：`/home/unitree/himloco_releases/20261025_alignment_3`。两者 C++/协议源码摘要相同，已用版本校验器交叉核验。采集脚本只订阅，不会开启另一个桥接。

在另一个登录 Jetson 的终端执行：

```bash
bash /home/unitree/himloco_alignment/run_passive_alignment.sh imu
```

每项先摆好起始姿态，按 Enter；先静止 2 s，听从屏幕文字缓慢完成动作，最后 2 s 保持终点。程序显示该项结束后再恢复机身。四项依次为静止、左侧抬高、机头抬高、机头向左转。左右以机器人自身为准；每次转动约 10～15°，持续承托机身并避开腿和线缆。

收到并审查 IMU 报告后，再执行关节组：

```bash
bash /home/unitree/himloco_alignment/run_passive_alignment.sh joints
```

十二项按 SDK FR、FL、RR、RL，每腿 hip/thigh/calf 提示，只有轻动指定关节，幅度约 5～15°。定义正向：hip 向机器人左侧摆；thigh 大腿向后摆；calf 伸展膝盖。无需按 R2。无法轻动或存在碰撞则输入 q 中止，不能强扳。

结果持久保存于 `/home/unitree/himloco_alignment/20261025_passive/physical/`。IMU 组结束因缺少十二关节和物理水平测量而仍显示未全部通过，这是预期状态。发生中断时保留原文件，不能将未完成采集当作有效记录。

完成数据核对后，才能按证据填写 R_BS、关节和 IMU 的确认引用，再评估运行阈值、悬空标定、短时策略及承重试运行。观测程序退出或桥接结束不等于电气断电。

## 第一轮人工 IMU 数据复核

用户已完成四项。Jetson 原文件已复制到 `audit/20261025_fixes/physical_imu_1/`，SHA 与用户报告一致；新增 `raw_diagnostics.json` 和 `alignment_report_detailed.json` 记录复核细节。

| 项目 | 起始 gyro RMS rad/s | 末段 gyro RMS rad/s | 单位 R_BS 下轴误差 | 结论 |
| --- | ---: | ---: | ---: | --- |
| roll_left_up | 0.00875 | 0.04649 | 3.45° | 末段未达到 0.03 rad/s 静止标准 |
| pitch_nose_up | 0.01234 | 0.03756 | 3.91° | 末段未达到静止标准 |
| yaw_left | 0.01077 | 0.01006 | 1.25° | 动态及起止静止检查通过 |

RPY/四元数最大差约 0.00253°，三个旋转的角度约 14.08°、11.39°、10.67°，gyro 积分/姿态转角比约 1.03、1.05、0.94。方向与读数初步匹配；roll/pitch 的失败来自结束段晃动，不是本轮发现了轴反号。静止捕获在单位变换下的倾角约 0.53°；拟合后的 1.95° 不能直接当作实际机身倾斜，手动旋转轴的不纯会影响拟合。

因此保持配置 R_BS=I，不将手动拟合的约 1.8° 小旋转写入配置；确认开关继续保持 false。原数据保留，复采只重做两项。新增工具参数 `--only`、`--still-seconds`，增加末段诊断字段，不改变 0.03 rad/s、1° 等判定门槛。本机全部 77 项检查通过，记录 `python_tests_retake.log`。

补采工具发布：`/home/unitree/himloco_releases/20261025_alignment_4`。新工作数据目录为 `/home/unitree/himloco_alignment/20261025_passive/physical_followup_1/`，已原样复制第一轮 neutral/yaw。仍使用原专用观测桥接，不重新启动控制器。

```bash
# 仅补采 roll 和 pitch，每项 14 s：开始/结束各 4 s 静止
bash /home/unitree/himloco_alignment/run_alignment_followup.sh imu

# 两项结束后可继续十二关节被动采集，结果汇总在同一新目录
bash /home/unitree/himloco_alignment/run_alignment_followup.sh joints
```

动作达到约 10～15° 后立即保持；最后提示“保持”后不再调整或回位，等本项结束再恢复。物理水平参考仍待实际测量，不能通过更改命令参数把未测量的支架姿态声明为已经测量。

## 补采与误操作复核

原始文件已复制至 `audit/20261025_fixes/physical_followup_1/`。`alignment_report_joints.json` 保留原自动结果，不更改判定阈值。新增 `physical_alignment_review.json`、`joint_trace_review.png` 记录人工粗核；生成脚本为 `audit/20261025_fixes/review_physical_followup.py`，不会连接机器人或写入部署配置。可随部署目录打包的记录为 `go1_gym_deploy/configs/evidence/himloco_go1_physical_review_20261025.json`。

### IMU 粗核

| 动作 | 实际姿态转角 | 单位安装变换下的轴误差 | 动态/静止检查 |
| --- | ---: | ---: | --- |
| 左侧抬高 | 13.956° | 3.533° | 通过 |
| 机头抬高 | 7.558° | 4.582° | 通过 |
| 机头左转 | 10.672° | 1.247° | 通过 |

RPY 与 wxyz 四元数最大旋转差约 0.000882°。单位变换下静止倾角约 0.533°。这支持 wxyz 及 x 前/y 左/z 上的粗略轴向约定；不等于精密 IMU 安装角或水平零点标定。手动动作存在混合轴分量，拟合矩阵反而使静止倾角变为约 3.73°，因此不写入拟合矩阵，继续保留 R_BS=I。自动报告的 `imu_verified=false` 原样保留。

### 关节曲线与已知错误

| 记录标签 | 指定 SDK 通道的端点变化 rad | 曲线说明 |
| --- | ---: | --- |
| FR hip | +0.1035 | 自动端点检查通过 |
| FR thigh | +0.1669 | 同腿 calf -0.0656，未达到严格独立运动条件 |
| FR calf | +0.1917 | 同腿 thigh -0.1565，伴随髋联动 |
| FL hip | +0.2927 | 自动端点检查通过 |
| FL thigh | +0.3018 | 自动端点检查通过 |
| FL calf | +0.0953 | FL hip -0.0545，伴随联动 |
| RR hip | +0.5109 | 用户先动左后，后在同项中纠正为右后；曲线一致，见下方时间窗口 |
| RR thigh | +0.4789 | 自动端点检查通过 |
| RR calf | -0.0743 | 6～10 s 中段 +0.0837，约 12 s 后回位至起点另一侧；不能只凭负端点认定反号 |
| RL hip | +0.3624 | 自动端点检查通过，与上一项先左后右的变化相互印证 |
| RL thigh | +0.3521 | 自动端点检查通过 |
| RL calf | +0.0797 | 同腿 thigh +0.1026，幅度更大；需要结合指定动作理解 |

用户说明两腿会因重力靠拢、需要人工摆开，这与部分被动联动相容。该现象不直接说明关节映射有误。FR thigh/calf、FL calf 和 RL calf 的指定通道正向变化均可见，同腿其他关节也发生变化；按人工指定动作和跨项曲线接受粗略身份与方向核对，不将它们伪写为“严格独立单关节自动通过”。不改变关节零位，不据这些手动角度校准编码器偏移。

### 右后髋误操作已通过同项后段核对解决

同一份 `joint_RR_hip_joint.jsonl`，相对起始均值：

| 采集内时段 | RR hip 变化 | RL hip 变化 | 与用户说明的对应 |
| --- | ---: | ---: | --- |
| 约 6～7 s | 约 -0.0077 rad | 约 +0.234 rad | 先误动左后髋 |
| 约 9～10 s | 约 +0.460 rad | 约 +0.024 rad | 已改做右后髋，左后大致回位 |

单独的 `joint_RL_hip_joint.jsonl` 中，RL hip 变化约 +0.362 rad，RR hip 端点仅约 -0.011 rad。两项与操作说明共同支持两个后髋身份正确，保留原始关节索引和符号，无需因此重采。

结论范围为物理通道/正负号与 IMU 轴向的人工粗核。HIM 主动标定、Kp/Kd=40/1 下真实跟踪、停止链路、完整策略及承重表现均尚未验证。下一步是承托机身、四腿无障碍下的受控标定；被动阶段两腿重力靠拢时可以人工摆开，但进入主动阶段前手必须撤离活动范围，不能通过持续用手拽腿完成标定。

### 本轮记录更新后的核验

- 本机全部 77 项 Python 检查通过，含固定官方源码数值对照：`audit/20261025_fixes/python_tests_post_physical_review.log`。
- 新发布目录 `releases/20261025_alignment_5` 已同步至 Jetson 的 `/home/unitree/himloco_releases/20261025_alignment_5`，目标端 65 个 manifest 文件校验通过；人工记录随包携带。
- 现有观测桥接二进制与 release 5 的 C++/协议源码摘要一致：`86672aa3728ca7defd4d331ab8d058da868a28282779f4e4aa5b70f4d2dfa02d`。本轮未修改或重新启动伺服桥接。
- 目标 Jetson 原 Docker 镜像在 `--network none` 下实际模型加载/forward `[1,270]→[1,12]` 通过，未发布任何电机命令。日志：`audit/20261025_fixes/jetson_post_physical_review_preflight.log`。
- 离线入口如实显示 `deployment_ready=false`，目前记录门控只剩 `runtime thresholds require an explicit review record for this robot`。人工粗核不是完整部署通过声明；主动运行所需阈值审查和实机标定另行完成。
- 采集结束后已用 Ctrl+C 正常退出专用 `--observe-only` 桥接，收到 `HIM bridge stopped`；随后检查未见 HIM 桥接/策略进程及 8090 绑定。未启动任何 HIM 主动标定或策略。

## 后续：首次悬空策略试运行已准备

用户随后请求开始 HIM 部署尝试。已完成首次悬空试验参数的静态审查，另建 `configs/himloco_go1.suspended_trial.json`，其运行记录门控已填写；普通 JSON 保留原状态。未改观测、策略、动作映射或 C++ 保护；首次配置收紧遥控范围、跟踪误差和机身倾角阈值。

新包 `20261025_suspended_trial_1` 以及终端 A/B 两个启动脚本已同步至 Go1，校验及目标镜像离线推理通过，尚未启动电机。完整操作说明见 `docs/HIMLOCO_FIRST_TRIAL_20261025.md`。第一次 R2 标定，READY 中第二次 R2 接管；全程承托悬空、摇杆居中，运行 2～3 秒后由操作者停止。
