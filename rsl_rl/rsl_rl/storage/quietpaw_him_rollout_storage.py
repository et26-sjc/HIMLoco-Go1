"""Reserved QuietPaw data interfaces; PPO still consumes the HIM batches."""

import torch

from .him_rollout_storage import HIMRolloutStorage


class QuietPawRolloutStorage(HIMRolloutStorage):
    """Unpopulated scalar side channels have shape [steps, envs, 1].

    NaN means unavailable, NOT a measured zero cost or a cost-value prediction.
    These buffers are deliberately excluded from the inherited minibatches,
    returns and advantages. Populating them requires a Stage 1 implementation.
    """

    class Transition(HIMRolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.epsilon = None
            self.noise_cost = None
            self.cost_value = None

    def __init__(self, num_envs, num_transitions_per_env, obs_shape,
                 privileged_obs_shape, actions_shape, device="cpu"):
        super().__init__(num_envs, num_transitions_per_env, obs_shape,
                         privileged_obs_shape, actions_shape, device)
        shape = (num_transitions_per_env, num_envs, 1)
        self.epsilon = torch.full(shape, float("nan"), device=self.device)
        self.noise_cost = torch.full(shape, float("nan"), device=self.device)
        self.cost_value = torch.full(shape, float("nan"), device=self.device)

    def add_transitions(self, transition):
        for name in ("epsilon", "noise_cost", "cost_value"):
            if getattr(transition, name, None) is not None:
                raise NotImplementedError(
                    "Stage 0 cannot consume populated {}: implement Stage 1 "
                    "before enabling QuietPaw data.".format(name)
                )
        super().add_transitions(transition)
