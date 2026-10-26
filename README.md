## Learning-based Locomotion Control from OpenRobotLab
This repository contains learning-based locomotion control research from OpenRobotLab, currently including [Hybrid Internal Model](/projects/himloco/README.md) & [H-Infinity Locomotion Control](/projects/h_infinity/README.md).

## Go1 deployment baseline

The current Go1 deployment snapshot is [him_go1_gym_deploy_20261026/him_go1_gym_deploy](him_go1_gym_deploy_20261026/him_go1_gym_deploy/README.md), exported from the robot deployment baseline `20261025_forward_trial_3`. It includes the policy, configuration, SDK/LCM bridge source, startup scripts, and recorded walking evidence. Its README is the deployment entry point.

See the [comparison with this project's Go1 implementation](docs/GO1_DEPLOYMENT_AUDIT_20261026.md) and [recommendations for deploying Isaac Lab policies](docs/GO1_LAB_DEPLOYMENT_MIGRATION.md).

## 🔥 News
- [2024-04] Code of HIMLoco is released.
- [2024-04] We release the [paper](https://arxiv.org/abs/2404.14405) of H-Infinity Locomotion Control. Please check the :point_right: [webpage](https://junfeng-long.github.io/HINF/) :point_left: and view our demos! :sparkler:
- [2024-01] HIMLoco is accepted by ICLR 2024.
- [2023-12] We release the [paper](https://arxiv.org/abs/2312.11460) of HIMLoco. Please check the :point_right: [webpage](https://junfeng-long.github.io/HIMLoco/) :point_left: and view our demos! :sparkler:

## 📝 TODO List
- \[x\] Release the training code of HIMLoco, please see `rsl_rl/rsl_rl/algorithms/him_ppo.py`.
- \[ \] Release deployment guidance of HIMLoco.
- \[ \] Release the training code of H-Infinity Locomotion Control.
- \[ \] Release deployment guidance of H-Infinity Locomotion Control.

## 📚 Getting Started

### Installation

We test our codes under the following environment:

- Ubuntu 20.04
- NVIDIA Driver: 525.147.05
- CUDA 12.0
- Python 3.7.16
- PyTorch 1.10.0+cu113
- Isaac Gym: Preview 4

1. Create an environment and install PyTorch:

  - `conda create -n himloco python=3.7.16`
  - `conda activate himloco`
  - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

2. Install Isaac Gym:
  - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
  - `cd isaacgym/python && pip install -e .`

3. Clone this repository.

  - `git clone https://github.com/OpenRobotLab/HIMLoco.git`
  - `cd HIMLoco`


4. Install HIMLoco.
  - `cd rsl_rl && pip install -e .`
  - `cd ../legged_gym && pip install -e .`

**Note:** Please use legged_gym and rsl_rl provided in this repo, we have modefications on these repos.

### Tutorial

1. Train a policy:

  - `cd legged_gym/legged_gym/scripts`
  - `python train.py`

2. Play and export the latest policy:
  - `cd legged_gym/legged_gym/scripts`
  - `python play.py`


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{long2023him,
  title={Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response},
  author={Long, Junfeng and Wang, ZiRui and Li, Quanyi and Cao, Liu and Gao, Jiawei and Pang, Jiangmiao},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}

@misc{long2024hinf,
  title={Learning H-Infinity Locomotion Control}, 
  author={Junfeng Long and Wenye Yu and Quanyi Li and Zirui Wang and Dahua Lin and Jiangmiao Pang},
  year={2024},
  eprint={2404.14405},
  archivePrefix={arXiv},
}
```

## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements
- [legged_gym](https://github.com/leggedrobotics/legged_gym): Our codebase is built upon legged_gym.











## 基本流程



- Python 3.7.16
- PyTorch 1.10.0+cu113
- Isaac Gym: Preview 4

1. Create an environment and install PyTorch:

  - `conda create -n himloco python=3.7.16`
  - `conda activate himloco`
  - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

2. Install Isaac Gym:
  - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
  - （下载安装包后解压，只运行下面的命令，不需要安装其他依赖）
  - `cd isaacgym/python && pip install -e .`

3. Clone this repository.

  - `git clone https://github.com/OpenRobotLab/HIMLoco.git`
  - `cd HIMLoco`


4. Install HIMLoco.
  - `cd rsl_rl && pip install -e .`
  - `cd ../legged_gym && pip install -e .`

**Note:** Please use legged_gym and rsl_rl provided in this repo, we have modefications on these repos.

### Tutorial

1. Train a policy:

  - `cd legged_gym/legged_gym/scripts`
  - `python train.py`
  - 会经历一些报错：1.ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory
  - 解决方法：find /home /usr /opt -name "libpython3.7m.so.1.0" 2>/dev/null，找到路径，没有就下载；
  - 找到路径后nano ~/.bashrc；把搜索到的路径添加到最后一行：export LD_LIBRARY_PATH=路径:$LD_LIBRARY_PATH。source ~/.bashrc

  - 报错2：没有安装tensorboard。解决：conda activate himloco ; pip install tensorboard

  - 报错3：AttributeError: module 'distutils' has no attribute 'version'
  - 解决方法：在anaconda里找到himloco环境的torch源文件，一般是：/anaconda3/envs/himloco/lib/python3.7/site-packages/torch
  - 在torch文件夹的子文件夹utils中找到tensorboard文件夹，打开__init__.py
  - 找到代码：from distutils import version（或：from distutils.version import LooseVersion）； 
  - LooseVersion = version.LooseVersion
  - 修改为：from packaging.version import Version；LooseVersion = Version
  - 这样一般就可以解决因为版本导致的报错。


2. Play and export the latest policy:
  - `cd legged_gym/legged_gym/scripts`
  - `python play.py`


3. 参数修改：
  - HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py里是基本的配置；

  - 默认使用机器人：aliengo

  - 修改配置参数时不能只修改基类的配置，还要修改所使用机器人的相应配置。或者运行时直接使用命令行参数如： --num_envs=2048

  - wandb和tensorboard的记录都在HIMLoco/legged_gym/logs里。

  - play.py默认使用最后一次记录的训练权重，但是我试验发现4000~5000轮次之后性能可能发生急剧下降，因此查看结果时需要提前看一下什么时
  - 候收敛，在HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py基础配置的最后一个类中修改调用的权重。

  - rsl_rl强化学习库中带him_ 前缀的文件是HIMLoco (Hybrid Internal Model) 的实现版本,而不带前缀的是标准 PPO 的基础实现。
