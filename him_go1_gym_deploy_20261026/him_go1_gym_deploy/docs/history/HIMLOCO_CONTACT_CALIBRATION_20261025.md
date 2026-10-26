# HIM 触地标定容差调整

后续更新：本配置已在实机进入 RUNNING，约 12.5 秒后因一次 69.6 ms 控制间隔停止。当前启动脚本指向 `20261025_supported_trial_2`，运行参数相同，增加精简日志与分段计时，详情见 `HIMLOCO_POLICY_CYCLE_GAP_20261025.md`。以下保留容差调整的原始依据。

## 依据

2026-10-25 第二次标定进入 SETTLE 后等待 5 秒退出。用户确认当时四脚触地或机身在承重。日志为 Jetson 上的 `him_suspended_20261025_181240_14678.jsonl`，本地副本在 `audit/20261025_fixes/him_second_calibration_settle_timeout.jsonl`。

最后一秒左后小腿关节平均为 −1.678977 rad，目标 −1.5 rad，绝对误差约 0.178977 rad；左前小腿误差约 0.153121 rad。它们超过原来 0.15 rad 的就绪门槛，因此当时没有进入 READY，也没有执行 HIM 策略。桥接记录的 `stop command` 是 Python 退出后的停止请求。

用户要求接受这次测得的小幅静态误差继续尝试，不再用双手抱住机身标定。根据现有记录准备单独的固定支撑触地配置，不把承重误差直接当作编码器零位偏差。

## 本次调整

- 专用配置 `go1_gym_deploy/configs/himloco_go1.supported_trial.json` 将标定容差从 0.15 改为 **0.20 rad**，约 11.46°。这是全部关节的绝对位置误差门槛，不是增加关节指令或修改零点。
- 必须连续取得 6 帧满足容差的实际观测，才进入 READY。5 秒等待期限保留。
- 增加 SETTLE 的剩余秒数、最大误差关节及容差提示，每秒最多一次。超时错误包含该关节、实际角度、目标和误差，JSONL 记录相同诊断字段。
- 模型、观测历史顺序、四元数解释、物理映射、动作比例、名义姿态、Kp/Kd=40/1 以及其余停止条件均保留。
- 审查依据保存在 `go1_gym_deploy/configs/evidence/himloco_go1_supported_runtime_review_20261025.json`。此前普通配置和悬空配置保留。

新发布目录为 `/home/unitree/himloco_releases/20261025_supported_trial_1`。C++ 源码没有变化，复制上一发布包中已修复线程计时竞争的桥接二进制及校验凭据，并核对当前源码摘要。

同步后在 Jetson 完成 69 个清单文件的完整性核对，桥接协议为 20261023，嵌入源码 SHA-256 为 `7a5cb1c4c3629f1b18c7543be180e5f1030c869d3bcbb83707992358cbe1b851`。Docker 使用 `--network none` 执行 `--check`，模型加载成功，输入 `[1,270]`、输出 `[1,12]`，没有发布电机命令。这是离线配置及模型接口检查。

## 操作

机身由固定支架或吊带承接，腿部活动范围无接触；停止后的零增益不能承托机身。无需人抱着标定。保持网线和 SSH，摇杆居中。

先在旧终端 A 按 **Ctrl+C** 结束已锁停的桥接，不能用 Ctrl+Z。两个终端均 SSH 登录 Go1 后：

终端 A：

```bash
sudo bash /home/unitree/himloco_alignment/start_him_trial_bridge.sh
```

按桥接提示 Enter，再启动终端 B：

```bash
bash /home/unitree/himloco_alignment/start_him_trial_policy.sh
```

1. 第一次 R2 松开再按一次，开始标定，无需一直按住。
2. 等 `HIM phase: READY`，观察姿态；如果未就绪，读取 `HIM settle` 的最大误差提示。
3. 只有在 READY 后松开再按第二次 R2，才开始 1 秒渐入和 HIM 策略。首轮居中运行约 2–3 秒再人工停止，暂不降低支撑或尝试行走。
4. L2 任意阶段停止；R2 在 RAMP/RUNNING 停止。终端 B Ctrl+C 也请求停止，然后终端 A Ctrl+C。所有停止都需要重启 A、B 才能再次接管。

本次文件准备和离线检查不证明 HIM 承重或行走已经通过；需要这一轮实际反馈。
