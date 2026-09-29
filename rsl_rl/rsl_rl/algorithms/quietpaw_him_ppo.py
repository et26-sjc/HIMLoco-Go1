"""Stage 0 uses unmodified HIMPPO optimization with extended storage."""

from .him_ppo import HIMPPO
from rsl_rl.storage.quietpaw_him_rollout_storage import QuietPawRolloutStorage


class QuietPawHIMPPO(HIMPPO):
    """Only storage selection differs; act/returns/update remain inherited."""

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        self.storage = QuietPawRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape,
            critic_obs_shape, action_shape, self.device,
        )
        self.transition = QuietPawRolloutStorage.Transition()
