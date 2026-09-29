"""Independent PID multipliers, following Stooke et al. (2020).

Gains and EMA factors are reproduction assumptions, not QuietPaw-reported
hyperparameters. State only advances when that environment has new episodes.
"""

import torch


class QuietPawPID:
    def __init__(self, epsilon, kp=1.0, ki=0.1, kd=0.1, ema=0.95, initial=0.0):
        if min(kp, ki, kd, initial) < 0 or not 0 <= ema < 1:
            raise ValueError("Invalid PID configuration")
        self.epsilon = epsilon.clone()
        self.kp, self.ki, self.kd, self.ema = kp, ki, kd, ema
        self.multiplier = torch.full_like(epsilon, initial)
        self.integral = self.multiplier.clone()
        self.proportional = torch.zeros_like(epsilon)
        self.cost_ema = epsilon.clone()  # no artificial derivative kick
        self.updates = torch.zeros_like(epsilon, dtype=torch.long)

    @torch.no_grad()
    def update(self, cost, valid):
        if cost.shape != self.epsilon.shape or valid.shape != self.epsilon.shape:
            raise ValueError("PID cost/valid shape mismatch")
        if not torch.isfinite(cost[valid]).all():
            raise ValueError("Non-finite PID cost")
        error = cost - self.epsilon
        integral = (self.integral + self.ki * error).clamp_min(0)
        proportional = self.ema * self.proportional + (1 - self.ema) * error
        cost_ema = self.ema * self.cost_ema + (1 - self.ema) * cost
        derivative = (cost_ema - self.cost_ema).clamp_min(0)
        multiplier = (integral + self.kp * proportional + self.kd * derivative).clamp_min(0)
        for name, value in (("integral", integral), ("proportional", proportional),
                            ("cost_ema", cost_ema), ("multiplier", multiplier)):
            getattr(self, name)[valid] = value[valid]
        self.updates[valid] += 1

    def state_dict(self):
        return {name: getattr(self, name).clone() for name in
                ("epsilon", "multiplier", "integral", "proportional", "cost_ema", "updates")}

    def load_state_dict(self, state):
        if not torch.equal(state["epsilon"].to(self.epsilon), self.epsilon):
            raise ValueError("PID resume requires the same environment count and epsilon grid.")
        for name, value in state.items():
            target = getattr(self, name)
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError("Invalid PID checkpoint: " + name)
            target.copy_(value.to(target))
