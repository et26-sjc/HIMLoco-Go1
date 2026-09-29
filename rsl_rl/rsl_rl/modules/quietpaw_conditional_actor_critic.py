"""Stage 1 conditioning; HIM history and estimator remain untouched."""

import copy
import torch
from torch import nn
from torch.distributions import Normal

from .quietpaw_him_actor_critic import QuietPawHIMActorCritic


class QuietPawConditionalActorCritic(QuietPawHIMActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve the original initialized weights and start with no dependence
        # on epsilon. All other actor/reward-critic layers retain their weights.
        for network in (self.actor, self.critic):
            old = network[0]
            new = nn.Linear(old.in_features + 1, old.out_features)
            with torch.no_grad():
                new.weight[:, :-1].copy_(old.weight)
                new.weight[:, -1].zero_()
                new.bias.copy_(old.bias)
            network[0] = new
        self.cost_critic = copy.deepcopy(self.critic)
        # No evidence initially for either positive or negative expected cost.
        nn.init.zeros_(self.cost_critic[-1].weight)
        nn.init.zeros_(self.cost_critic[-1].bias)
        print("QuietPaw Stage 1: actor 65; reward/cost critics 239; actions 12")

    @staticmethod
    def checked_epsilon(epsilon, reference):
        if epsilon is None:
            raise ValueError("Stage 1 requires explicit epsilon [batch, 1].")
        if epsilon.shape != (reference.shape[0], 1):
            raise ValueError("epsilon must have shape [batch, 1].")
        if not torch.isfinite(epsilon).all() or ((epsilon < 0) | (epsilon > 1)).any():
            raise ValueError("epsilon must be finite and within [0, 1].")
        return epsilon.to(device=reference.device, dtype=reference.dtype)

    def act_inference(self, obs_history, epsilon=None):
        epsilon = self.checked_epsilon(epsilon, obs_history)
        with torch.no_grad():
            velocity, latent = self.estimator(obs_history)
        return self.actor(torch.cat((obs_history[:, :self.num_one_step_obs],
                                     velocity, latent, epsilon), dim=-1))

    def act(self, obs_history, epsilon=None, **kwargs):
        mean = self.act_inference(obs_history, epsilon)
        self.distribution = Normal(mean, mean * 0. + self.std)
        return self.distribution.sample()

    def evaluate(self, critic_observations, epsilon=None, **kwargs):
        epsilon = self.checked_epsilon(epsilon, critic_observations)
        return self.critic(torch.cat((critic_observations, epsilon), dim=-1))

    def evaluate_cost(self, critic_observations, epsilon):
        epsilon = self.checked_epsilon(epsilon, critic_observations)
        return self.cost_critic(torch.cat((critic_observations, epsilon), dim=-1))
