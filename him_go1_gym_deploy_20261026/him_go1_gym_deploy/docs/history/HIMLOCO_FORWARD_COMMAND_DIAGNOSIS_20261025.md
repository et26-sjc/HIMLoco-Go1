# HIM 前进检查：站姿前倾、尚未形成交替迈步

## 本轮证据

机器人日志：`/home/unitree/himloco_alignment/him_trial_logs/him_supported_20261025_183655_16565.jsonl`。
本地原始副本：`audit/20261025_fixes/him_first_unsupported_forward_attempt.jsonl`。
曲线：`audit/20261025_fixes/him_forward_calf_tracking.png`。

- 已经进入 RUNNING，记录约 35 秒；最终桥接因遥控器 L2 停止而锁定。`-20261023` 是锁定状态标识，需要结合桥接具体原因判断，并非本轮发生了协议不一致。
- 用户确认撤去支架后可以承重站立，但四脚大多贴地，主要是机身前倾或缓慢挪动。尚未验证行走，也没有足够证据判断某个电机损坏。
- 247 个 RUNNING 帧的前进命令超过 0.05 m/s，上限达到原试运行配置的 0.15 m/s；命令实际进入策略，并非摇杆输入丢失。
- 这段前进采样中，左前小腿实际角度极差约 0.0013 rad，Python 目标极差约 0.0637 rad；右后小腿实际角度极差约 0.0017 rad，也存在类似现象。左前、右后小腿的平均目标减反馈误差约 0.248、0.202 rad。该段受载跟踪偏差值得继续检查，不能把“左前不抬”简单归因为单腿指令未发送。
- 策略目标未呈现明显周期性交替迈步。HIM 没有 WTW 试运行脚本固定的 3 Hz 小跑命令，零速时不应据此要求 HIM 原地碎步。
- RUNNING 周期最大约 20.34 ms，纯模型调用最大约 5.17 ms。本轮没有出现上次的周期超时；这不能排除以后偶发调度延迟。
- 旧日志只有 Python 请求目标，没有 MCU 力矩或逐帧 post-Safety UDP 输出，因此不能用旧日志确定电流限制、外部负载、驱动问题或物理位置零偏。

## 训练命令范围与部署配置

用户指定默认训练 fork 的固定提交为 `c59f76349c4718da19aa9b055a830ac1a383f038`。训练代码 `_resample_commands` 中：

```python
self.commands[env_ids, :2] *= (
    torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
).unsqueeze(1)
```

[固定版本源码](https://github.com/et26-sjc/HIMLoco-Go1/blob/c59f76349c4718da19aa9b055a830ac1a383f038/legged_gym/legged_gym/envs/base/legged_robot.py#L477)。这是训练时的命令采样规则。它不是导出策略内部的运行时清零条件，部署的 0.15 m/s 仍会构成有效非零观测。按照这份默认配置，非零平移采样模长大于 0.20 m/s；原部署的直行上限 0.15 m/s 处于该采样间隙，因此无法充分验证模型在训练命令范围内的行走行为。

这是一项有源码依据的排查假设，不是已经证实的唯一根因。实际 checkpoint 的完整训练日志缺失；超过 0.20 m/s 也不保证一定迈步。

## 本次准备的独立版本

新配置 `go1_gym_deploy/configs/himloco_go1.forward_trial.json`，新发布目录 `20261025_forward_trial_1`。

1. 只把前进／后退摇杆速度范围从 ±0.15 调为 ±0.30 m/s；横移 ±0.10 m/s、转向 ±0.15 rad/s 不变。回中仍为零，保持线性缩放及原死区，不添加强制最低速度。
2. 每帧补记 MCU `tau_est`（换成策略关节顺序）和原始摇杆值，每秒打印实际送入策略的速度命令。MCU 力矩是估计值，并不证明驱动实际力矩精度。
3. 模型、45 维观测顺序、6 帧历史、动作缩放、关节映射、IMU 变换、KP=40/KD=1、标定容差、目标步长与跟踪限制、状态/遥控/命令超时、L2/R2 停止逻辑全部保留。
4. 二进制桥接及其已修复的锁内计时逻辑复用，源码及构建收据继续校验。

旧的 supported_trial_2 发布目录和日志保留，可以回退。记录文件 `configs/evidence/himloco_go1_forward_runtime_review_20261025.json` 明确标记行走尚未验证。

## 人工执行顺序

准备能够承接机身、完全避开腿部活动范围的支撑和前方空地。程序不能在已经 L2 锁定的桥接上继续接管；先在旧终端 A 按 Ctrl+C，然后重新启动 A、B。本文准备过程不启动电机。

终端 A：

```bash
sudo bash /home/unitree/himloco_alignment/start_him_trial_bridge.sh
```

按桥接提示 Enter。终端 B：

```bash
bash /home/unitree/himloco_alignment/start_him_trial_policy.sh
```

摇杆回中，第一次 R2 标定；READY 后松开再按第二次 R2；RUNNING 后短时推前进，确认屏幕 `HIM command: vx=0.250` 到 `0.300 m/s`，只观察 1–2 秒并回中。L2 停止，随后终端 A Ctrl+C。该版本不自动按 2 秒结束，回中也不结束策略。

若仍四脚贴地前倾、某腿明显不能跟随，或触发保护，应立即停止并保留日志。下一步对比该段动作目标、实际角度、速度和力矩；必要时在腿部完全卸载时检查同一关节的跟随情况，再区分负载/驱动与策略是否真正给出迈步目标。不要靠继续加大速度、增益或撤掉保护强迫迈步。
