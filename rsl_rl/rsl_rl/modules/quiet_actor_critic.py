"""Frozen HIM locomotion with adaptations inside the actor's linear layers.

The Gaussian and its sampled action are unchanged. Only its mean network is
adapted. The independent HIM estimator optimizer must be disabled by the runner.
"""

import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .him_actor_critic import HIMActorCritic


class LoRALinear(nn.Linear):
    """A linear layer retaining the original ``weight``/``bias`` state keys."""

    def __init__(self, source, rank):
        super().__init__(source.in_features, source.out_features,
                         bias=source.bias is not None)
        if rank < 1 or rank > min(source.in_features, source.out_features):
            raise ValueError("LoRA rank must be in [1, min(input, output)].")
        self.to(device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if self.bias is not None:
                self.bias.copy_(source.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.rank = rank
        # alpha = rank, so scaling is exactly one for every selected layer.
        self.scaling = 1.0
        self.lora_A = nn.Parameter(source.weight.new_empty(rank, source.in_features))
        self.lora_B = nn.Parameter(source.weight.new_zeros(source.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs):
        original = F.linear(inputs, self.weight, self.bias)
        update = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return original + self.scaling * update

    def merged_weight(self):
        return self.weight + self.scaling * self.lora_B.matmul(self.lora_A)


class DeltaLinear(nn.Linear):
    """Unrestricted output-weight/bias increment around a frozen linear head."""

    def __init__(self, source):
        super().__init__(source.in_features, source.out_features,
                         bias=source.bias is not None)
        self.to(device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if self.bias is not None:
                self.bias.copy_(source.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.delta_weight = nn.Parameter(torch.zeros_like(source.weight))
        self.delta_bias = (nn.Parameter(torch.zeros_like(source.bias))
                           if source.bias is not None else None)

    def forward(self, inputs):
        return (F.linear(inputs, self.weight, self.bias)
                + F.linear(inputs, self.delta_weight, self.delta_bias))

    def merged_weight(self):
        return self.weight + self.delta_weight


class QuietHIMActorCritic(HIMActorCritic):
    """HIM policy with explicit frozen/head/lora/full experiment modes.

    ``base_actor`` and ``base_std`` are checkpointed snapshots. Disabling the
    adaptation selects these snapshots, including after full actor fine-tuning.
    All modes freeze every estimator parameter, including its target/prototypes.
    """

    def __init__(self, num_actor_obs, num_critic_obs, num_one_step_obs, num_actions,
                 actor_hidden_dims=(512, 256, 128),
                 critic_hidden_dims=(512, 256, 128), activation='elu',
                 init_noise_std=1.0, adaptation='lora', lora_ranks=(8, 16, 8),
                 **kwargs):
        if adaptation not in ('frozen', 'head', 'lora', 'full'):
            raise ValueError("Unknown quiet adaptation mode: {}".format(adaptation))
        if num_actor_obs % num_one_step_obs:
            raise ValueError("Observation history must contain complete frames.")
        if len(lora_ranks) != len(actor_hidden_dims):
            raise ValueError("Specify one LoRA rank per actor hidden layer.")
        super().__init__(num_actor_obs, num_critic_obs, num_one_step_obs, num_actions,
                         actor_hidden_dims=list(actor_hidden_dims),
                         critic_hidden_dims=list(critic_hidden_dims),
                         activation=activation, init_noise_std=init_noise_std,
                         **kwargs)
        self.adaptation = adaptation
        self.lora_ranks = tuple(int(rank) for rank in lora_ranks)
        self.adapter_enabled = True
        self.policy_spec = {
            'num_actor_obs': num_actor_obs,
            'num_critic_obs': num_critic_obs,
            'num_one_step_obs': num_one_step_obs,
            'num_actions': num_actions,
            'actor_hidden_dims': list(actor_hidden_dims),
            'critic_hidden_dims': list(critic_hidden_dims),
            'activation': activation,
            'init_noise_std': float(init_noise_std),
            'adaptation': adaptation,
            'lora_ranks': list(self.lora_ranks),
        }
        self.base_actor = copy.deepcopy(self.actor)
        self.base_actor.requires_grad_(False)
        self.register_buffer('base_std', self.std.detach().clone())
        self.estimator.requires_grad_(False)
        self.actor.requires_grad_(adaptation == 'full')
        linear_indices = [i for i, layer in enumerate(self.actor)
                          if isinstance(layer, nn.Linear)]
        if adaptation == 'lora':
            for index, rank in zip(linear_indices[:-1], self.lora_ranks):
                self.actor[index] = LoRALinear(self.actor[index], rank)
        if adaptation in ('head', 'lora'):
            self.actor[linear_indices[-1]] = DeltaLinear(self.actor[linear_indices[-1]])

    def adapter_parameters(self):
        """Return mean-network parameters learned by this experiment mode."""
        return (parameter for parameter in self.actor.parameters()
                if parameter.requires_grad)

    def set_adapter_enabled(self, enabled):
        self.adapter_enabled = bool(enabled)
        # The next forward must rebuild a distribution for the selected mode.
        self.distribution = None

    def update_distribution(self, obs_history):
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        inputs = torch.cat((obs_history[:, :self.num_one_step_obs], vel, latent), dim=-1)
        actor = self.actor if self.adapter_enabled else self.base_actor
        mean = actor(inputs)
        std = self.std if self.adapter_enabled else self.base_std
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act_inference(self, obs_history, observations=None):
        vel, latent = self.estimator(obs_history)
        inputs = torch.cat((obs_history[:, :self.num_one_step_obs], vel, latent), dim=-1)
        actor = self.actor if self.adapter_enabled else self.base_actor
        return actor(inputs)

    def load_base_model(self, state_dict):
        """Load an ordinary HIM model with audited keys and fresh adaptation.

        Validate before copying any tensor. Ordinary actor/critic/estimator/std
        keys are all required, and no unexpected key or shape is accepted.
        Optimizer state is deliberately not part of baseline initialization.
        """
        current = self.state_dict()
        new_keys = self._adaptation_state_keys()
        required = set(current) - new_keys
        missing = required - set(state_dict)
        unexpected = set(state_dict) - required
        bad_shapes = [key for key in required.intersection(state_dict)
                      if current[key].shape != state_dict[key].shape]
        if missing or unexpected or bad_shapes:
            raise RuntimeError("Invalid baseline model: missing={}, unexpected={}, "
                               "shape_mismatch={}".format(sorted(missing),
                                                         sorted(unexpected),
                                                         sorted(bad_shapes)))
        if any(not torch.isfinite(value).all().item() for value in state_dict.values()
               if value.is_floating_point()):
            raise ValueError("Baseline model contains non-finite tensors.")
        if torch.any(state_dict['std'] <= 0).item():
            raise ValueError("Baseline action standard deviation must be positive.")
        self.load_state_dict(state_dict, strict=False)
        with torch.no_grad():
            for layer in self.actor:
                if isinstance(layer, LoRALinear):
                    nn.init.kaiming_uniform_(layer.lora_A, a=math.sqrt(5))
                    layer.lora_B.zero_()
                if isinstance(layer, DeltaLinear):
                    layer.delta_weight.zero_()
                    if layer.delta_bias is not None:
                        layer.delta_bias.zero_()
            original = {name[len('actor.'):]: value
                        for name, value in state_dict.items() if name.startswith('actor.')}
            self.base_actor.load_state_dict(original, strict=True)
            self.base_std.copy_(self.std)
        self.set_adapter_enabled(True)
        return self

    def _adaptation_state_keys(self):
        return {key for key in self.state_dict()
                if key == 'base_std' or key.startswith('base_actor.')
                or key.endswith(('.lora_A', '.lora_B', '.delta_weight', '.delta_bias'))}

    def merged_actor(self, adapter_enabled=True):
        """Copy effective weights into ordinary Linear layers for deployment."""
        if not adapter_enabled:
            return copy.deepcopy(self.base_actor).eval()
        result = []
        for source in self.actor:
            if isinstance(source, (LoRALinear, DeltaLinear)):
                target = nn.Linear(source.in_features, source.out_features,
                                   bias=source.bias is not None)
                target.to(device=source.weight.device, dtype=source.weight.dtype)
                with torch.no_grad():
                    target.weight.copy_(source.merged_weight())
                    if source.bias is not None:
                        bias = source.bias
                        if isinstance(source, DeltaLinear):
                            bias = bias + source.delta_bias
                        target.bias.copy_(bias)
                result.append(target)
            else:
                result.append(copy.deepcopy(source))
        return nn.Sequential(*result).eval()

    def export_policy(self, path, adapter_enabled=True):
        return export_quiet_policy(self, path, adapter_enabled=adapter_enabled)


class QuietPolicyExporter(nn.Module):
    """Scriptable deployment policy; no critic, target encoder, or prototypes."""

    __constants__ = ['num_one_step_obs']

    def __init__(self, model, adapter_enabled=True):
        super().__init__()
        self.actor = model.merged_actor(adapter_enabled=adapter_enabled).cpu()
        self.encoder = copy.deepcopy(model.estimator.encoder).cpu()
        self.num_one_step_obs = model.num_one_step_obs

    def forward(self, history):
        parts = self.encoder(history)
        velocity = parts[:, :3]
        latent = F.normalize(parts[:, 3:], dim=-1, p=2.0)
        inputs = torch.cat((history[:, :self.num_one_step_obs], velocity, latent), dim=-1)
        return self.actor(inputs)


def export_quiet_policy(model, path, adapter_enabled=True):
    """Write a JIT policy file; preserve the live model and adapter state."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    exporter = QuietPolicyExporter(model, adapter_enabled=adapter_enabled).eval()
    scripted = torch.jit.script(exporter)
    scripted.save(path)
    return path
