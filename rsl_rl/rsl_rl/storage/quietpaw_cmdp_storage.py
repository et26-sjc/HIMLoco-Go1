"""Separate reward/cost GAE and consistently indexed conditional PPO batches."""

import torch
from .quietpaw_him_rollout_storage import QuietPawRolloutStorage
from .him_rollout_storage import HIMRolloutStorage


class QuietPawCMDPStorage(QuietPawRolloutStorage):
    class Transition(QuietPawRolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.cost_reward = None  # (1-gamma)*cost plus timeout bootstrap

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_rewards = torch.zeros_like(self.rewards)
        self.cost_returns = torch.zeros_like(self.rewards)
        self.cost_advantages = torch.zeros_like(self.rewards)
        self.reward_advantages = torch.zeros_like(self.rewards)
        self.combined_advantages = torch.zeros_like(self.rewards)

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        for name in ("epsilon", "noise_cost", "cost_value", "cost_reward"):
            value = getattr(transition, name)
            if value is None or value.shape != (self.num_envs, 1) or not torch.isfinite(value).all():
                raise ValueError("Invalid CMDP transition field: " + name)
        for name in ("epsilon", "noise_cost"):
            if ((getattr(transition, name) < 0) | (getattr(transition, name) > 1)).any():
                raise ValueError(name + " must be within [0, 1].")
        for name in ("epsilon", "noise_cost", "cost_value"):
            getattr(self, name)[self.step].copy_(getattr(transition, name))
        self.cost_rewards[self.step].copy_(transition.cost_reward)
        # Bypass only the Stage 0 unavailable-field guard.
        HIMRolloutStorage.add_transitions(self, transition)

    def compute_cost_returns(self, last_cost_values, gamma, lam, multipliers):
        if self.step != self.num_transitions_per_env:
            raise ValueError("CMDP returns require a full rollout.")
        advantage = torch.zeros_like(last_cost_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_value = last_cost_values if step == self.num_transitions_per_env - 1 else self.cost_value[step + 1]
            nonterminal = 1.0 - self.dones[step].float()
            delta = self.cost_rewards[step] + gamma * nonterminal * next_value - self.cost_value[step]
            advantage = delta + gamma * lam * nonterminal * advantage
            self.cost_returns[step] = advantage + self.cost_value[step]
        self.cost_advantages.copy_(self.cost_returns - self.cost_value)
        # Undo the reward-only normalization in inherited compute_returns.
        self.reward_advantages.copy_(self.returns - self.values)
        self.combined_advantages.copy_(
            (self.reward_advantages - multipliers * self.cost_advantages) / (1 + multipliers))
        # One common scale only: never independently normalize cost advantage.
        self.advantages.copy_((self.combined_advantages - self.combined_advantages.mean()) /
                              (self.combined_advantages.std() + 1e-8))

    def cmdp_mini_batch_generator(self, num_mini_batches, num_epochs):
        count = self.num_envs * self.num_transitions_per_env
        if count % num_mini_batches:
            raise ValueError("CMDP rollout size must divide evenly into minibatches.")
        fields = dict(observations=self.observations, critic=self.privileged_observations,
                      next_critic=self.next_privileged_observations, actions=self.actions,
                      values=self.values, returns=self.returns, advantages=self.advantages,
                      log_prob=self.actions_log_prob, mu=self.mu, sigma=self.sigma,
                      epsilon=self.epsilon, cost_value=self.cost_value, cost_returns=self.cost_returns)
        flat = {name: value.flatten(0, 1) for name, value in fields.items()}
        # Match HIM: one permutation reused across the PPO epochs.
        indices = torch.randperm(count, device=self.device)
        for _ in range(num_epochs):
            for batch in indices.split(count // num_mini_batches):
                yield {name: value[batch] for name, value in flat.items()}
