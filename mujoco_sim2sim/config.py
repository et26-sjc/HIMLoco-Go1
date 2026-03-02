from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Go1PolicyConfig:
    """Policy/observation settings aligned with HIM Go1 training config."""

    num_actions: int = 12
    num_one_step_obs: int = 45
    history_len: int = 6

    # from legged_robot_config.normalization.obs_scales
    obs_scale_ang_vel: float = 0.25
    obs_scale_dof_pos: float = 1.0
    obs_scale_dof_vel: float = 0.05
    obs_scale_lin_vel: float = 2.0

    # from go1_config.control
    action_scale: float = 0.25
    hip_reduction: float = 1.0
    kp: float = 40.0
    kd: float = 1.0

    # timing from legged_robot_config.sim + go1_config.control.decimation
    sim_dt: float = 0.005
    decimation: int = 4

    # use exact Go1 default joint order expected by policy/action head
    dof_names: tuple[str, ...] = (
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    )
    default_dof_pos: tuple[float, ...] = (
        0.1, 0.8, -1.5,
        -0.1, 0.8, -1.5,
        0.1, 1.0, -1.5,
        -0.1, 1.0, -1.5,
    )

    # command scaling used in LeggedGym observation:
    # commands[:3] * [lin_vel, lin_vel, ang_vel] scales -> [2.0, 2.0, 0.25] for Go1
    commands_scale: tuple[float, float, float] = (2.0, 2.0, 0.25)


@dataclass
class MujocoSimConfig:
    """MuJoCo runtime settings for sim2sim rollout."""

    model_xml: Path = Path("legged_gym/resources/robots/go1/xml/go1.xml")
    policy_path: Path = Path("policy.pt")
    device: str = "cpu"

    viewer: bool = True
    steps: int = 4000

    # debug logging
    debug: bool = False
    print_every: int = 20

    cmd_x: float = 1.0
    cmd_y: float = 0.0
    cmd_yaw: float = 0.0  # used only when heading_command=False

    # match training: heading_command=True means command[3]=target heading (rad),
    # and yaw-rate is computed each step from heading error and clipped.
    heading_command: bool = True
    cmd_heading: float = 0.0  # target heading in rad when heading_command=True

    gravity_vec_world: tuple[float, float, float] = (0.0, 0.0, -1.0)

    policy: Go1PolicyConfig = field(default_factory=Go1PolicyConfig)
