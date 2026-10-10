"""First-episode statistics: never count auto-reset episodes as survivors."""

import torch


class EpsilonEvaluation:
    def __init__(self, num_envs, epsilon, gamma, device):
        self.epsilon, self.gamma = float(epsilon), gamma
        self.cost_budget = self.epsilon
        self.active = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.steps = torch.zeros(num_envs, device=device)
        self.reward = torch.zeros_like(self.steps)
        self.discounted_reward = torch.zeros_like(self.steps)
        self.cost = torch.zeros_like(self.steps)
        self.discount = torch.ones_like(self.steps)
        self.discounted_cost = torch.zeros_like(self.steps)
        self.budget_estimate = torch.zeros_like(self.steps)
        self.failed = torch.zeros_like(self.active)
        self.timed_out = torch.zeros_like(self.active)
        self.sums, self.peaks, self.discounted_components = {}, {}, {}

    @torch.no_grad()
    def add(self, rewards, dones, metrics, timeout, terminal_cost_value, last_step=False):
        active = self.active.clone()
        self.steps += active
        self.reward += rewards * active
        self.discounted_reward += self.discount * rewards * active
        self.cost += metrics["noise_cost"] * active
        self.discounted_cost += self.discount * (1 - self.gamma) * metrics["noise_cost"] * active
        for name in ("noise_force_cost", "noise_impact_cost"):
            if name in metrics:
                if name not in self.discounted_components:
                    self.discounted_components[name] = torch.zeros_like(self.steps)
                self.discounted_components[name] += self.discount * (1 - self.gamma) * metrics[name] * active
        self.discount[active] *= self.gamma
        failures = metrics["failure"].bool() & active
        self.failed |= failures
        self.timed_out |= timeout.bool() & active & ~failures
        finished = active & (dones.bool() | last_step)
        bootstrap = finished & ~failures
        self.budget_estimate[finished] = self.discounted_cost[finished]
        self.budget_estimate[bootstrap] += self.discount[bootstrap] * terminal_cost_value.flatten()[bootstrap]
        for name, value in metrics.items():
            value = value.float()
            if name not in self.sums:
                self.sums[name] = torch.zeros_like(self.steps)
                self.peaks[name] = torch.zeros_like(self.steps)
            self.sums[name] += value * active
            self.peaks[name][active] = torch.maximum(self.peaks[name][active], value[active])
        self.active[finished] = False

    def summary(self):
        if self.active.any() or (self.steps == 0).any():
            raise ValueError("Evaluation must finish every first episode or its horizon")
        result = dict(epsilon=self.epsilon, cost_budget=self.cost_budget, num_envs=len(self.steps),
                      locomotion_reward=self.reward.mean().item(),
                      observed_discounted_reward=self.discounted_reward.mean().item(),
                      reward_per_step=(self.reward / self.steps).mean().item(),
                      noise_cost=(self.cost / self.steps).mean().item(),
                      observed_discounted_cost=self.discounted_cost.mean().item(),
                      observed_budget_violation=(self.discounted_cost.mean() - self.cost_budget).clamp_min(0).item(),
                      estimated_budget_cost=self.budget_estimate.mean().item(),
                      estimated_budget_violation=(self.budget_estimate.mean() - self.cost_budget).clamp_min(0).item(),
                      mean_episode_violation=(self.budget_estimate - self.cost_budget).clamp_min(0).mean().item(),
                      survival_success_rate=(~self.failed).float().mean().item(),
                      timeout_completion_rate=self.timed_out.float().mean().item(),
                      mean_survival_steps=self.steps.mean().item())
        for name, total in self.sums.items():
            result[name + "_mean"] = (total / self.steps).mean().item()
            result[name + "_max"] = self.peaks[name].max().item()
        for name, total in self.discounted_components.items():
            result[name + "_discounted"] = total.mean().item()
        result["touchdown_events_per_episode"] = self.sums["touchdown_events"].mean().item()
        return result

    def episodes(self):
        return dict(reward=self.reward.tolist(), steps=self.steps.tolist(),
                    observed_discounted_reward=self.discounted_reward.tolist(),
                    observed_discounted_cost=self.discounted_cost.tolist(),
                    estimated_budget_cost=self.budget_estimate.tolist(), failure=self.failed.tolist())
