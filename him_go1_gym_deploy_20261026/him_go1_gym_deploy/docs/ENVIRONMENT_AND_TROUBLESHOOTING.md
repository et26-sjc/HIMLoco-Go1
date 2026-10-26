# 环境复用、网络和故障排查

## 一、运行环境

当前流程分两部分：

- Jetson 宿主机：C++ SDK3.8 桥接，ARM64，使用已安装的 `unitree_legged_sdk-go1`、LCM 和 SDK 库。
- Docker：现有 `jetson-model-deployment` 镜像中执行 Python/Torch/NumPy/LCM。CPU 推理也保留 `--runtime nvidia`，以提供该 Jetson 镜像依赖的库。

本版本不需要启动或进入 `foxy_controller`，不需要安装整个训练仓库、Isaac Gym 或 ROS。旧 `go1_gym_deploy/docker/Dockerfile` 为 WTW 历史环境，不能据此宣称重新构建后能加载当前 JIT。运行镜像是否兼容应由实际权重 `--check` 判定。

环境读取命令（不启动电机）：

```bash
uname -m
docker image inspect jetson-model-deployment --format '{{.Id}} {{.Architecture}} {{.Os}}'
bash /home/unitree/him_go1_gym_deploy/scripts/prepare_bridge.sh --describe
sudo bash /home/unitree/him_go1_gym_deploy/scripts/check_environment.sh
```

若要另存现有镜像，人工执行 `docker save`；镜像体积较大，本代码导出不附带镜像归档：

```bash
sudo docker save -o /home/unitree/jetson-model-deployment.tar jetson-model-deployment
# 在兼容的 Jetson 环境恢复时：sudo docker load -i 镜像归档路径
```

## 二、网络和终端定义

- **终端 A**：电脑的第一个 SSH 会话，登录 Go1 后运行 `start_bridge.sh`。
- **终端 B**：电脑的第二个 SSH 会话，登录同一台 Go1 后运行 `start_policy.sh`。
- 网络 IP 使用已经部署成功的原机设置：电脑通过网线访问 Jetson `192.168.123.15`。C++ 桥接在宿主 UDP 8090 上与 MCU `192.168.123.10:8007` 通信，保留原代码设置。
- Python 和桥接运行在 Jetson；Docker 使用 `--network host`，与宿主共享 LCM 多播网络。
- LCM URL 由根 JSON 的 `runtime.lcm_url` 决定。若显式设置 `HIM_LCM_URL`，它必须与根 JSON 一致，启动脚本会检查。
- 原桥接及原厂运动控制不可与本桥接争用控制通道。终端 A 在 Enter 提示前没有绑定 UDP 8090；先按 Enter 再启动 B。

默认前台流程中保留网线和 SSH。用户遥控器的命令通过机载 SDK 上报；SSH 并不发送速度控制量，但它承载进程的终端。未设置 nohup/systemd 或验证断线行为，不能将拔网线视为可靠停止或可靠继续运行。

## 三、常见故障

| 现象 | 含义及处理 |
| --- | --- |
| 找不到 `foxy_controller` / 容器未运行 | 本导出不用该容器，使用根目录两个启动脚本 |
| 终端 A 提示 Enter、B 提示 8090 未绑定 | 在 A 按 Enter，看到 UDP/循环启动后再启动 B |
| `A bridge or controller is already present` | 查看下列进程/端口命令，先处理旧会话，不同时开两套控制器 |
| `Bridge binary ... missing` | 先运行 `prepare_bridge.sh --reuse ...` 或 `--build ...` |
| checksum/source digest 不匹配 | 二进制与源码不配套，使用匹配版本或重新编译，不能简单重写 checksum 绕过 |
| `-20261023` | 桥接已锁存停止；查看终端 A 的 `HIM bridge latched: ...` 具体原因 |
| `policy command timeout` | 有效 Python 命令中断超过 watchdog，检查 B、推理/日志/主机负载；排查后重启 A/B |
| `remote L2 stop` / `stop command` | 按钮或 Python 请求停止，通常为预期停止原因；确认是否由操作者触发 |
| `Nominal pose did not settle ...` | 当前标定误差未连续满足容差，检查触地承重、腿部干涉和目标/实际曲线，不等同于 R2 没按住 |
| `Policy cycle gap` / deadline | 实际控制周期超时，读取故障前分段耗时，检查 CPU 调度、推理和日志开销 |
| `SDK PositionLimit` / 功率保护 | SDK 保护触发，检查目标及反馈；不能让两份不同限位的动作历史静默继续 |
| `Target step exceeds ...` | 根默认配置已关闭此额外门控；若仍出现，核对是否启动了旧路径/旧 JSON |
| `Stopped` / 按过 Ctrl+Z | 进程被暂停，不能当成正常退出；不要直接恢复旧控制程序继续动作 |
| 模型加载错误 | 核对模型 SHA、原镜像和架构，先运行离线 `--check`，无需开桥接 |

读取进程、端口和资源占用：

```bash
pgrep -af 'lcm_position|deploy_policy.py|wtw_supported_trial.py|go1_fr.*sdk38'
sudo ss -uanp 'sport = :8090'
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
sudo docker stats --no-stream him_go1_policy
top
```

`top` 中单进程 `%CPU` 可能按一个逻辑 CPU 计量；它不能单独证明 20 ms 周期满足要求，应结合 JSONL 的 `period_s`、`policy_inference_s` 和分段时序。程序中断后按操作说明正常退出 A，排查后手动重新启动，不配置自动重启继续控制。

## 四、配置、模型和版本维护

根 `manifest.json` 保护当前 75 个冻结文件。编辑其覆盖的源码、JSON 或模型后，启动校验会拒绝继续，这有助于区分本次基线与修改版本。

后续开发请复制一份工作目录修改，完成对照记录后，用原打包工具生成新的目录；不要原地覆盖这份基线或只更改清单来跳过检查。打包入口：

```bash
# 在已修改的开发副本中，明确指定配置和权重，输出目录必须尚不存在
python3 -B go1_gym_deploy/scripts/build_deployment_bundle.py \
  --config /absolute/path/to/reviewed_config.json \
  --model /absolute/path/to/new_policy.pt \
  --output /absolute/path/to/new_release
```

该工具保留模型哈希检查。新模型须更新真实来源、配置、SHA、关节顺序和接口核对，不只替换文件名。若 C++ 或 LCM 定义变化，应重新编译桥接，不能复用旧二进制。根目录额外的通用启动脚本可在新发布目录复用，内部 Python 包名仍是 `go1_gym_deploy`。

完整性校验不会检查清单之外的额外文件，也不是数字签名。导出时的额外脚本和文档哈希在 `EXPORT_MANIFEST.json` 中；运行生成的日志和桥接 build 目录不属于这份静态导出清单。
