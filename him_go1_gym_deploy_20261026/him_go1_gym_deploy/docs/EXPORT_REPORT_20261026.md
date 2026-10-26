# 导出核对记录

导出日期：2026-10-26，Asia/Shanghai。

## 来源和目标

- 原工程：`/home/sjc/HIMLoco_WTW_deploy/HIMLoco_WTW_deploy`。
- 原冻结发布：`releases/20261025_forward_trial_3`。
- 导出：`/home/sjc/him_go1_gym_deploy`。
- 归档：`/home/sjc/him_go1_gym_deploy_20261026.tar.gz`。
- 原机曾使用：`/home/unitree/himloco_releases/20261025_forward_trial_3`。

本次读取本地冻结包，并检查原工程对应文件；75 个原 manifest 文件均与冻结包逐字节一致。随后原样复制这 75 个文件及原 manifest 到导出目录。没有通过编辑 manifest 放行控制代码变化。

## 新增内容

- 可从自身目录定位根配置和模型的 `start_bridge.sh`、`start_policy.sh`。
- 桥接复用/编译、Docker 离线检查及公共路径脚本，共五个新增 shell 文件。
- 根 README、当前接口、流程、环境、已知问题和本记录；原阶段性文档放入 `docs/history/`。
- 最后取得的完整行走 JSONL、增益核对、低姿态分析和关节曲线，放入 `evidence/`。
- 覆盖全部导出静态文件的 `EXPORT_MANIFEST.json`；它与原 manifest 的覆盖范围不同。

新启动包装保持控制参数、状态机、观测、权重、关节/坐标映射和 C++ 核心原样。容器名由历史 `him_forward_trial` 改为导出目录专用的 `him_go1_policy`；日志写入本目录 `logs/`。新的控制器启动前确认本导出路径的活动桥接和端口，拒绝 observe-only 桥接及已存在的控制器。

## 本次完成的离线检查

| 检查 | 结果 |
| --- | --- |
| 原冻结包 75 个文件的 SHA-256 | 全部匹配 |
| 导出原文件与冻结包逐字节比较 | 全部一致 |
| 原工程对应代码与冻结包比较 | 无变化 |
| 原工具 `build_deployment_bundle.py --verify` | 75 个文件通过 |
| 40 个 Python 源码 AST 语法解析 | 通过，未执行测试套件 |
| 5 个新增 shell 文件 `bash -n` | 通过 |
| 四个操作脚本 `--describe` | 正常返回，不创建机器人连接或电机命令 |
| 本机 HIM Python 环境 `deploy_policy.py --check` | 配置与权重加载通过，float32 `[1,270]` → `[1,12]` |
| 离线输出 | `deployment_ready=true`，`motor_commands_published=false` |

本机模型检查产生了 CUDA 初始化警告，但实际模型由既有代码以 CPU 加载并完成推理；未因该警告改动控制代码。该检查不替代 Jetson 上的实时周期核对。

## 本次未能重新取得的目标端内容

Go1 `192.168.123.15:22` 在导出时连接超时，因此没有重新读取机上最新文件、Docker image ID 或桥接 ELF。ARM64 二进制与镜像未附入包；已提供在原机复用或编译的方法。此前目标端曾核验桥接协议号及源码摘要：

```text
HIM_BRIDGE protocol=20261023 source_sha256=7a5cb1c4c3629f1b18c7543be180e5f1030c869d3bcbb83707992358cbe1b851
```

本次未启动桥接、状态接收或电机控制，未在目标端安装/替换文件，未运行单元测试或新增测试文件。此前单元检查和实机实验仅作为历史证据；当前低姿态和膝触地问题保持未解决状态。

## 文件完整性使用方法

原控制基线仍由原入口校验：

```bash
python3 -B /absolute/path/to/him_go1_gym_deploy/go1_gym_deploy/scripts/build_deployment_bundle.py \
  --verify /absolute/path/to/him_go1_gym_deploy
```

`EXPORT_MANIFEST.json` 记录新增脚本、文档和所附证据的 SHA-256；它排除自身以及之后生成的 `logs/`、`build/` 和 Python 缓存。归档旁的 `.sha256` 文件可用 `sha256sum -c` 校验压缩包。两份清单都是完整性记录，不是签名或实机运行保证。
