"""Flat-ground quiet adaptation with physics-step contact measurements."""

import math

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.quiet_contact import QuietContactTracker, touchdown_cost_rate


class Go1QuietRobot(LeggedRobot):
    # The existing MuJoCo runner clips at this boundary. This is a deployment
    # comparison, not the Isaac Gym action limit or a joint/torque limit.
    deployment_action_limit = 1.0

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.quiet
        self.quiet_tracker = QuietContactTracker(
            self.num_envs, len(self.feet_indices), self.device,
            self.sim_params.dt, self.cfg.control.decimation,
            enter_force=cfg.contact_enter_force,
            exit_force=cfg.contact_exit_force,
            min_air_time=cfg.min_air_time,
            startup_time=cfg.startup_time,
            impact_window=cfg.impact_window)
        self.quiet_episode_time = torch.zeros(self.num_envs, device=self.device)
        self.quiet_previous_position = self.root_states[:, :3].clone()
        self.quiet_reference = cfg.touchdown_reference
        self.quiet_target_weight = float(cfg.target_weight)
        self.quiet_progress = 0.0
        self.quiet_evaluation_command = None
        self.quiet_action_clip_fraction = torch.zeros(self.num_envs, device=self.device)
        self.quiet_deployment_limit_exceeded_fraction = torch.zeros_like(self.quiet_action_clip_fraction)
        if self.quiet_reference is not None:
            self.set_quiet_reward(self.quiet_reference, self.quiet_target_weight, 0.0)

    def set_quiet_reward(self, reference, target_weight, progress=1.0):
        """Install instance-local calibration; checkpoint metadata persists it."""
        reference = float(reference)
        target_weight = float(target_weight)
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError("Quiet reward requires a finite, positive baseline reference")
        if not math.isfinite(target_weight) or target_weight < 0:
            raise ValueError("Quiet reward weight must be finite and nonnegative")
        self.quiet_reference = reference
        self.quiet_target_weight = target_weight
        self.set_quiet_progress(progress)

    def step(self, actions):
        # Measure the raw policy sample before the base class clips it. These
        # diagnostics never transform actions or enter the reward.
        magnitude = actions.detach().abs()
        self.quiet_action_clip_fraction = (
            magnitude > self.cfg.normalization.clip_actions).float().mean(dim=1).to(self.device)
        self.quiet_deployment_limit_exceeded_fraction = (
            magnitude > self.deployment_action_limit).float().mean(dim=1).to(self.device)
        return super().step(actions)

    def set_quiet_progress(self, progress):
        progress = float(progress)
        if not math.isfinite(progress):
            raise ValueError("Quiet reward progress must be finite")
        self.quiet_progress = min(1.0, max(0.0, progress))

    def set_evaluation_command(self, command):
        """Use one deployable velocity command for every environment and reset."""
        if command is None:
            self.quiet_evaluation_command = None
        else:
            value = torch.as_tensor(command, dtype=torch.float, device=self.device)
            if value.shape != (3,) or not bool(torch.isfinite(value).all()):
                raise ValueError("Evaluation command must contain finite vx, vy and yaw")
            self.quiet_evaluation_command = value.clone()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = 0.0
        evaluation_command = getattr(self, "quiet_evaluation_command", None)
        if evaluation_command is not None:
            self.commands[env_ids, :3] = evaluation_command
            return
        speeds = torch.as_tensor(self.cfg.quiet.training_speeds,
                                 dtype=torch.float, device=self.device)
        if speeds.numel() == 0:
            raise ValueError("At least one training speed is required")
        # Each environment belongs to a fixed speed group. This overrides the
        # baseline's special high-speed/environment-subset resampling path.
        self.commands[env_ids, 0] = speeds[env_ids.long() % speeds.numel()]

    def _begin_control_step(self):
        self.quiet_tracker.begin_step()
        self.quiet_previous_position.copy_(self.root_states[:, :3])

    def _after_physics_substep(self, substep):
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        foot_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        self.quiet_episode_time.add_(self.sim_params.dt)
        self.quiet_tracker.observe(
            self.contact_forces[:, self.feet_indices, 2],
            foot_states[:, self.feet_indices, 9],
            self.quiet_episode_time)

    def _reward_quiet_touchdown(self):
        if self.quiet_reference is None or self.quiet_progress == 0.0:
            return torch.zeros(self.num_envs, device=self.device)
        cost = self.quiet_tracker.touchdown_speed.square().sum(dim=(1, 2))
        return (self.quiet_target_weight * self.quiet_progress
                * touchdown_cost_rate(cost, self.dt, self.quiet_reference))

    def compute_reward(self):
        super().compute_reward()
        # This method runs after termination detection and before reset_idx.
        # Clone every tensor so failed episodes retain their final measurements.
        data = self.quiet_tracker.snapshot(self.reset_buf.bool())
        data.update({
            "commands": self.commands[:, :3].clone(),
            "base_velocity": self.base_lin_vel.clone(),
            "base_angular_velocity": self.base_ang_vel.clone(),
            "projected_gravity": self.projected_gravity.clone(),
            "root_position": self.root_states[:, :3].clone(),
            "foot_velocity": self.feet_vel.clone(),
            "contact": self.quiet_tracker.contact.clone(),
            "torques": self.torques.clone(),
            "dof_velocity": self.dof_vel.clone(),
            "actions": self.actions.clone(),
            "target_offset": (self.actions * self.cfg.control.action_scale).clone(),
            "track_error": (self.commands[:, :2] - self.base_lin_vel[:, :2]).square().sum(dim=1),
            "yaw_error": (self.commands[:, 2] - self.base_ang_vel[:, 2]).square(),
            "distance": (self.root_states[:, :2] - self.quiet_previous_position[:, :2]).norm(dim=1),
            "done": self.reset_buf.bool().clone(),
            "time_out": self.time_out_buf.bool().clone(),
            "failed": (self.reset_buf.bool() & ~self.time_out_buf.bool()).clone(),
            "episode_time": self.quiet_episode_time.clone(),
            "orientation": self.projected_gravity[:, :2].square().sum(dim=1),
            "power": (self.torques * self.dof_vel).abs().sum(dim=1),
            "slip": (self.feet_vel[:, :, :2].square().sum(dim=2)
                     * self.quiet_tracker.contact.float()).sum(dim=1),
            "action_clip_fraction": self.quiet_action_clip_fraction.clone(),
            "action_clip_limit": float(self.cfg.normalization.clip_actions),
            "deployment_action_limit_exceeded_fraction": self.quiet_deployment_limit_exceeded_fraction.clone(),
            "deployment_action_limit": self.deployment_action_limit,
            "torque_saturation": (self.torques.abs() >= 0.98 * self.torque_limits).float().mean(dim=1),
        })
        tracking_rate = torch.zeros(self.num_envs, device=self.device)
        for name in ("tracking_lin_vel", "tracking_ang_vel"):
            scale = self.reward_scales.get(name, 0.0) / self.dt
            if scale > 0:
                tracking_rate.add_(getattr(self, "_reward_" + name)() * scale)
        data["positive_tracking_rate"] = tracking_rate
        data["quiet_weight"] = self.quiet_target_weight * self.quiet_progress
        self.extras["quiet"] = data

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) and hasattr(self, "quiet_tracker"):
            self.quiet_tracker.reset(env_ids)
            self.quiet_episode_time[env_ids] = 0.0
            self.quiet_previous_position[env_ids] = self.root_states[env_ids, :3]
