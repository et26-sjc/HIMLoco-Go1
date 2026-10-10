"""Read-only noise measurements at the existing pre-reset reward boundary."""

import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from .quietpaw_noise import noise_cost_components


class QuietPawLeggedRobot(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        shape = (self.num_envs, len(self.feet_indices))
        self.quietpaw_contact = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.quietpaw_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.quietpaw_velocity = torch.zeros(*shape, 3, device=self.device)
        self.quietpaw_force = torch.zeros(*shape, 3, device=self.device)
        self.quietpaw_base_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg.noise_cost.definition not in ("normalized_complement_v1", "normalized_complement_v2"):
            raise ValueError("Unknown noise definition; use an explicitly documented reproduction assumption.")

    def compute_reward(self):
        # Parent post_physics_step refreshed force/body/root state already;
        # this is after physics and BEFORE reset_idx overwrites terminal state.
        # HIM only refreshes extras["time_outs"] on reset steps. Its previous
        # tensor can otherwise remain stale after time_out_buf is reassigned.
        # Stage 1 must publish THIS step's flags even if no environment resets.
        self.extras["time_outs"] = self.time_out_buf.clone()
        force = self.contact_forces[:, self.feet_indices].clone()
        cfg = self.cfg.noise_cost
        force_cost, impact_cost, contact, touchdown = noise_cost_components(
            force, self.quietpaw_velocity, self.quietpaw_contact, self.quietpaw_valid,
            cfg.sigma_force, cfg.sigma_velocity, cfg.contact_threshold,
            cfg.force_weight, cfg.impact_weight)
        cost = force_cost + impact_cost
        magnitude = torch.linalg.vector_norm(force, dim=-1)
        difference = torch.linalg.vector_norm(force - self.quietpaw_force, dim=-1) / self.dt
        acceleration = torch.linalg.vector_norm(
            self.root_states[:, 7:10] - self.quietpaw_base_velocity, dim=-1) / self.dt
        self.extras["quietpaw"] = {
            "noise_cost": cost.clone(),
            "noise_force_cost": force_cost.clone(),
            "noise_impact_cost": impact_cost.clone(),
            # Diagnostic peaks are across feet at a 50-Hz sample, NOT 200-Hz
            # contact-impact peaks. They never enter optimization.
            "fz_peak": force[..., 2].clamp_min(0).amax(dim=-1),
            "f3d_peak": magnitude.amax(dim=-1),
            "df3d_peak": difference.amax(dim=-1) * self.quietpaw_valid,
            "base_acceleration": acceleration * self.quietpaw_valid,
            "touchdown_events": touchdown.sum(dim=-1),
            "touchdown_speed_peak": ((-self.quietpaw_velocity[..., 2]).clamp_min(0) * touchdown).amax(dim=-1),
            "velocity_tracking_error": torch.linalg.vector_norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=-1),
            "tilt": torch.linalg.vector_norm(self.projected_gravity[:, :2], dim=-1),
            # Preserve the actual failure signal even if a time limit happens
            # on the same step; do not change the environment's reset decision.
            "failure": torch.any(torch.linalg.vector_norm(
                self.contact_forces[:, self.termination_contact_indices], dim=-1) > 1., dim=1),
        }
        self.quietpaw_contact.copy_(contact)
        self.quietpaw_velocity.copy_(self.feet_vel)
        self.quietpaw_force.copy_(force)
        self.quietpaw_base_velocity.copy_(self.root_states[:, 7:10])
        self.quietpaw_valid.fill_(True)
        # Original reward, scales, dt multiplication and episode sums unchanged.
        super().compute_reward()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "quietpaw_valid"):
            self.quietpaw_valid[env_ids] = False
            self.quietpaw_contact[env_ids] = False
            self.quietpaw_velocity[env_ids] = 0
            self.quietpaw_force[env_ids] = 0
            self.quietpaw_base_velocity[env_ids] = 0
