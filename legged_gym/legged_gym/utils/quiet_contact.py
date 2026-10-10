"""Physics-sample contact measurements without an Isaac Gym dependency.

Velocity is the preceding sample in world coordinates. It is a discrete
touchdown proxy, not an exact pre-collision velocity. Impact measurements are
20 ms sampled windows, not acoustic measurements or isolated impulses.
"""

import math

import torch


class QuietContactTracker:
    def __init__(self, num_envs, num_feet, device, physics_dt, decimation,
                 enter_force=2.0, exit_force=0.5, min_air_time=0.01,
                 startup_time=0.5, impact_window=0.02):
        if physics_dt <= 0 or decimation < 1:
            raise ValueError("Physics dt and decimation must be positive")
        if not 0 <= exit_force < enter_force:
            raise ValueError("Contact exit force must be below enter force")
        if min_air_time < 0 or startup_time < 0 or impact_window <= 0:
            raise ValueError("Contact timing values are invalid")
        self.physics_dt = float(physics_dt)
        self.decimation = int(decimation)
        self.min_air_time = float(min_air_time)
        self.startup_time = float(startup_time)
        self.enter_force = float(enter_force)
        self.exit_force = float(exit_force)
        self.window_steps = max(1, int(math.ceil(impact_window / physics_dt - 1e-9)))
        shape = (num_envs, num_feet)
        self.contact = torch.zeros(shape, device=device, dtype=torch.bool)
        self.initialized = torch.zeros_like(self.contact)
        self.air_time = torch.zeros(shape, device=device)
        self.previous_velocity = torch.zeros_like(self.air_time)
        self.window_remaining = torch.zeros(shape, device=device, dtype=torch.long)
        self.window_peak = torch.zeros_like(self.air_time)
        self.window_load = torch.zeros_like(self.air_time)
        event_shape = (num_envs, self.decimation, num_feet)
        self.touchdown_event = torch.zeros(event_shape, device=device, dtype=torch.bool)
        self.touchdown_speed = torch.zeros(event_shape, device=device)
        self.impact_event = torch.zeros_like(self.touchdown_event)
        self.impact_peak_samples = torch.zeros_like(self.touchdown_speed)
        self.impact_load_samples = torch.zeros_like(self.touchdown_speed)
        self.impact_invalid = torch.zeros(shape, device=device, dtype=torch.long)
        self._next_substep = 0

    def begin_step(self):
        """Clear control-step outputs. Keep contact and active windows."""
        self.touchdown_event.zero_()
        self.touchdown_speed.zero_()
        self.impact_event.zero_()
        self.impact_peak_samples.zero_()
        self.impact_load_samples.zero_()
        self.impact_invalid.zero_()
        self._next_substep = 0

    @torch.no_grad()
    def observe(self, force_z, velocity_z, episode_time):
        """Record one physics sample. Inputs have shape (N,F), (N,F), (N)."""
        substep = self._next_substep
        if substep >= self.decimation:
            raise RuntimeError("Call begin_step before the next control period")
        if force_z.shape != self.contact.shape or velocity_z.shape != self.contact.shape:
            raise ValueError("Foot sample shape does not match the tracker")
        next_contact = torch.where(self.contact,
                                   force_z > self.exit_force,
                                   force_z >= self.enter_force)
        event = (self.initialized & ~self.contact & next_contact
                 & (self.air_time + 1e-8 >= self.min_air_time)
                 & (episode_time.unsqueeze(1) + 1e-8 >= self.startup_time))
        speed = torch.clamp(-self.previous_velocity, min=0.0)
        self.touchdown_event[:, substep].copy_(event)
        self.touchdown_speed[:, substep].copy_(torch.where(event, speed, torch.zeros_like(speed)))

        # An overlapping new contact invalidates the previous window. Normally
        # hysteresis and the air-time threshold prevent such overlap.
        self.impact_invalid.add_((event & (self.window_remaining > 0)).long())
        self.window_remaining[event] = self.window_steps
        self.window_peak[event] = 0.0
        self.window_load[event] = 0.0
        active = self.window_remaining > 0
        positive_force = torch.clamp(force_z, min=0.0)
        self.window_peak.copy_(torch.where(active,
                                          torch.maximum(self.window_peak, positive_force),
                                          self.window_peak))
        self.window_load.add_(positive_force * self.physics_dt * active.float())
        self.window_remaining.sub_(active.long())
        complete = active & (self.window_remaining == 0)
        self.impact_event[:, substep].copy_(complete)
        self.impact_peak_samples[:, substep].copy_(torch.where(complete, self.window_peak,
                                                             torch.zeros_like(self.window_peak)))
        self.impact_load_samples[:, substep].copy_(torch.where(complete, self.window_load,
                                                             torch.zeros_like(self.window_load)))

        self.air_time.copy_(torch.where(next_contact | ~self.initialized,
                                        torch.zeros_like(self.air_time),
                                        self.air_time + self.physics_dt))
        self.contact.copy_(next_contact)
        # A copy is essential: Isaac Gym refreshes its source tensors in place.
        self.previous_velocity.copy_(velocity_z)
        self.initialized.fill_(True)
        self._next_substep += 1
        return event

    def snapshot(self, terminal_mask=None):
        """Return independent outputs, including incomplete terminal windows."""
        invalid = self.impact_invalid.clone()
        if terminal_mask is not None:
            invalid.add_(((self.window_remaining > 0)
                          & terminal_mask.unsqueeze(1)).long())
        return {
            "touchdown_cost": self.touchdown_speed.square().sum(dim=(1, 2)),
            "touchdown_count": self.touchdown_event.sum(dim=1),
            "touchdown_event": self.touchdown_event.clone(),
            "touchdown_speed": self.touchdown_speed.clone(),
            "impact_count": self.impact_event.sum(dim=1),
            "impact_event": self.impact_event.clone(),
            "impact_peak_samples": self.impact_peak_samples.clone(),
            "impact_load_samples": self.impact_load_samples.clone(),
            "impact_peak": self.impact_peak_samples.max(dim=1).values.clone(),
            "impact_load": self.impact_load_samples.sum(dim=1),
            "impact_invalid": invalid,
        }

    def reset(self, env_ids):
        """Clear persistent history only for the reset environments."""
        for name in ("contact", "initialized", "air_time", "previous_velocity",
                     "window_remaining", "window_peak", "window_load"):
            getattr(self, name)[env_ids] = 0


def touchdown_cost_rate(cost, control_dt, reference):
    """Rate supplied to a reward system that multiplies scales by control_dt."""
    if control_dt <= 0 or reference <= 0 or not math.isfinite(reference):
        raise ValueError("A finite, positive reference and control dt are required")
    return cost / (control_dt * reference)
